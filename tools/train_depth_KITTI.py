from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import json
import torchvision
import argparse
import multiprocessing as mp
import torch.nn as nn
import threading
from random import choice
import random
import os
from distutils.version import LooseVersion
import argparse
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import *
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from model.diffusers.models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.depth_module import Depthmodule
import torch.optim as optim
import torch.nn.functional as F
from model.segment.criterion import SetCriterion
from model.segment.matcher import HungarianMatcher
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
import yaml
from tools.utils import mask_image
from torch.optim.lr_scheduler import StepLR
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
LOW_RESOURCE = False 

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            if self.activate:
                self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.activate:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if self.activate:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.activate = True
        
def freeze_params(params):
    for param in params:
        param.requires_grad = False
        


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="./config/",
        help="config for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--image_limitation",
        type=int,
        default=5,
        help="image_limitation",
    )
    parser.add_argument(
        "--dataset", type=str, default="Cityscapes", help="dataset: VOC/Cityscapes/MaskCut"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default="Test"
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
#     opt.dataset = cfg.DATASETS.dataset
    opt.batch_size = cfg.DATASETS.batch_size
#     opt.image_limitation = cfg.DATASETS.image_limitation
    
    # dataset
    if opt.dataset == "VOC":
        dataset = Semantic_VOC(
            set="train",
        )
    elif opt.dataset == "Cityscapes":
        dataset = Semantic_Cityscapes(
            set="train",image_limitation = opt.image_limitation
        )
    elif opt.dataset == "COCO":
        dataset = Instance_COCO(
            set="train",image_limitation = opt.image_limitation
        )   
    elif opt.dataset == "KITTI":
        dataset = KITTI(
            is_train="train",image_limitation = opt.image_limitation, depth_scale=cfg.Depth_Decorder.max_depth
        ) 
        loss_fn = SiLogLoss()
    elif opt.dataset == "VirtualKITTI":
        dataset = VirtualKITTI2(
            is_train="train",image_limitation = opt.image_limitation, depth_scale=cfg.Depth_Decorder.max_depth
        ) 
        loss_fn = SiLogLoss()
    else:
        return
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print('***********************   begin   **********************************')
    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    learning_rate = cfg.SOLVER.learning_rate
    adam_weight_decay = cfg.SOLVER.adam_weight_decay
    total_epoch = cfg.SOLVER.total_epoch
    
    ckpt_dir = os.path.join(save_dir, opt.save_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    tokenizer = CLIPTokenizer.from_pretrained("./dataset/ckpts/imagenet/", subfolder="tokenizer")
    
    #VAE
    vae = AutoencoderKL.from_pretrained("./dataset/ckpts/imagenet/", subfolder="vae")
    freeze_params(vae.parameters())
    vae=vae.to(device)
    vae.eval()
    
    unet = UNet2D.from_pretrained("./dataset/ckpts/imagenet/", subfolder="unet")
    freeze_params(unet.parameters())
    unet=unet.to(device)
    unet.eval()
    
    text_encoder = CLIPTextModel.from_pretrained("./dataset/ckpts/imagenet/text_encoder")
    freeze_params(text_encoder.parameters())
    text_encoder=text_encoder.to(device)
    text_encoder.eval()
    
    
    
    depth_module=Depthmodule(max_depth=cfg.Depth_Decorder.max_depth).to(device)
    
    noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    print("learning_rate:",learning_rate)
    g_optim = optim.Adam(
            [{"params": depth_module.parameters()},],
            lr=learning_rate
          )
    scheduler = StepLR(g_optim, step_size=350, gamma=0.1)
    
    
    start_code = None
    
    LOW_RESOURCE = cfg.Diffusion.LOW_RESOURCE
    NUM_DIFFUSION_STEPS = cfg.Diffusion.NUM_DIFFUSION_STEPS
    GUIDANCE_SCALE = cfg.Diffusion.GUIDANCE_SCALE
    MAX_NUM_WORDS = cfg.Diffusion.MAX_NUM_WORDS
    
    
    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
    for j in range(total_epoch):
        print('Epoch ' +  str(j) + '/' + str(total_epoch))
        
        for step, batch in enumerate(dataloader):
            g_cpu = torch.Generator().manual_seed(random.randint(1, 10000000))
            
            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()
            
            
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            prompts = batch["prompt"]
            original_depth = batch["original_depth"]
            original_image = batch["original_image"]
            
            batch_size = image.shape[0]
            latents = vae.encode(image.to(device)).latent_dist.sample().detach()
            latents = latents * 0.18215
            
            # Sample noise that we'll add to the latents
            noise = torch.randn(latents.shape).to(latents.device)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            
            # set timesteps
            noise_scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
            stepss = noise_scheduler.timesteps[-1]
            timesteps = torch.ones_like(timesteps) * stepss
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            start_code = noisy_latents.to(latents.device)
    
            
            images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,noise_scheduler, prompts, controller, latent=start_code, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, low_resource=LOW_RESOURCE, Train=True)

                           
            
            if step%100 ==0:
                ptp_utils.save_images(images_here,out_put = (os.path.join(ckpt_dir,  'training/'+'viz_sample_{0:05d}'.format(j)+".png")))

                Image.fromarray(original_depth.cpu().numpy()[0].astype(np.uint8)).save(os.path.join(ckpt_dir, 'training/'+ 'original_depth_{0:05d}'.format(j)+".png"))
                Image.fromarray(original_image.cpu().numpy()[0].astype(np.uint8)).save(os.path.join(ckpt_dir, 'training/'+ 'original_image_{0:05d}'.format(j)+".png"))
                
                
            # train segmentation
            diffusion_features=get_feature_dic()
            pred_depth=depth_module(diffusion_features,controller,prompts,tokenizer)
            
            loss = []
            for b_index in range(batch_size):
                train_class_index=0
                pred_depth=torch.unsqueeze(pred_depth[b_index,train_class_index,:,:],0).unsqueeze(0)
                depth=depth[b_index].float().unsqueeze(0).unsqueeze(0).cuda()

                loss.append(loss_fn(pred_depth, depth))
            
            if len(loss)==0:
                pass
            else:
                total_loss=0
                for i in range(len(loss)):
                    total_loss+=loss[i]
                total_loss/=batch_size
                g_optim.zero_grad()
                total_loss.backward()
                g_optim.step()
        
        
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}, lr: {3:0.6f}, prompt: ".format(step, len(dataloader), total_loss ,float(g_optim.state_dict()['param_groups'][0]['lr'])),prompts)

            
            if step%100 ==0:
                annotation_pred_gt = depth[0][0].cpu()
                pred_depth = pred_depth[0][0].cpu()
                annotation_pred_gt = annotation_pred_gt/annotation_pred_gt.max()*255
                pred_depth = pred_depth/pred_depth.max()*255
                viz_tensor2 = torch.cat([annotation_pred_gt, pred_depth], axis=1)

                torchvision.utils.save_image(viz_tensor2, os.path.join(ckpt_dir, 
                                                        'training/'+ str(b_index)+'viz_sample_{0:05d}_seg'.format(j)+'.png'))
                    

        print("Saving latest checkpoint to",ckpt_dir)
        torch.save(depth_module.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        if j%10==0:
            torch.save(depth_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
        scheduler.step()


if __name__ == "__main__":
    main()