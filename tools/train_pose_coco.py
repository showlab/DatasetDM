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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
from model.pose_module import PoseModule
import torch.optim as optim
import torch.nn.functional as F
from model.segment.criterion import SetCriterion
from model.segment.matcher import HungarianMatcher
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
import yaml
from model.loss import NMTCritierion
from tools.utils import mask_image
from torch.optim.lr_scheduler import StepLR
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
LOW_RESOURCE = False 

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
            

# Chunhua Style
# (R,G,B)
color2 = [(252,176,243),(252,176,243),(252,176,243),
    (0,176,240), (0,176,240), (0,176,240),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
    (255,255,0), (255,255,0),(169, 209, 142),
    (169, 209, 142),(169, 209, 142)]

link_pairs2 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
        ]

point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)


def plot_pose(image,dt_joints):
    # Plot
    w = h = 512
    ref = 512
    fig = plt.figure(figsize=(512/100, 512/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(image)
    bk.set_zorder(-1)
                
                
    gt_joints_dict = map_joint_dict(dt_joints)
    
    # stick 
    for k, link_pair in enumerate(chunhua_style.link_pairs):

        if gt_joints_dict[link_pair[0]][0] < 0.1 or gt_joints_dict[link_pair[1]][0] < 0.1 or gt_joints_dict[link_pair[0]][1] < 0.1 or gt_joints_dict[link_pair[1]][1] < 0.1:
            continue
        
        if k in range(6,11):
            lw = 1
        else:
            lw = ref / 100.
        line = mlines.Line2D(
                np.array([gt_joints_dict[link_pair[0]][0],
                          gt_joints_dict[link_pair[1]][0]]),
                np.array([gt_joints_dict[link_pair[0]][1],
                          gt_joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2],)
        line.set_zorder(0)
        ax.add_line(line)

    # black ring
    for k in range(dt_joints.shape[0]):
#         if dt_joints[k,2] < joint_thres \
#             or vg[link_pair[0]] == 0 \
#             or vg[link_pair[1]] == 0:
#             continue
        if dt_joints[k,0] > w or dt_joints[k,1] > h or dt_joints[k,1] < 0.1 or dt_joints[k,0] < 0.1:
            continue

        if k in range(5):
            radius = 1
        else:
            radius = ref / 100

        circle = mpatches.Circle(tuple(dt_joints[k,:2]), 
                                 radius=radius, 
                                 ec='black', 
                                 fc=chunhua_style.ring_color[k], 
                                 alpha=1, 
                                 linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)
    return plt
                
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
        
def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

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
    
    opt.dataset = cfg.DATASETS.dataset
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
    elif opt.dataset == "NYU":
        dataset = nyudepthv2(
            is_train="train",image_limitation = opt.image_limitation
        ) 
        loss_fn = SiLogLoss()
    elif opt.dataset == "coco_pose":
        dataset = COCODataset(
            cfg.DATASETS, is_train="train",image_limitation = opt.image_limitation
        ) 
        loss_fn = NMTCritierion(label_smoothing = 0.1).cuda()
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
    
    
    
    pose_module=PoseModule(cfg).to(device)
    
    noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    print("learning_rate:",learning_rate)
    g_optim = optim.Adam(
            [{"params": pose_module.parameters()},],
            lr=learning_rate
          )
    scheduler = StepLR(g_optim, step_size=400, gamma=0.1)
    
    
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
            
            image, target, target_weight, meta, prompts, original_image = batch
            
            
            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()

            
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

                Image.fromarray(original_image.cpu().numpy()[0].astype(np.uint8)).save(os.path.join(ckpt_dir, 'training/'+ 'original_image_{0:05d}'.format(j)+".png"))
                
                
            # train pose model
            diffusion_features=get_feature_dic()
            output_x, output_y=pose_module(diffusion_features,controller,prompts,tokenizer)

            target = target.cuda(non_blocking=True).long()
            target_weight = target_weight.cuda(non_blocking=True).float()

            total_loss = loss_fn(output_x, output_y, target, target_weight)
            g_optim.zero_grad()
            total_loss.backward()
            g_optim.step()
            
        
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}, lr: {3:0.6f}, prompt: ".format(step, len(dataloader), total_loss ,float(g_optim.state_dict()['param_groups'][0]['lr'])),prompts)

            
            if step%100 ==0 or total_loss<50:
                # pred
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)
                
                max_val_x, preds_x = output_x.max(2,keepdim=True)
                max_val_y, preds_y = output_y.max(2,keepdim=True)
                
                
                output = torch.ones([1,preds_x.size(1),2])
                output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, 1.0))
                output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, 1.0))
                print(output)
                print(target[0].cpu().numpy())
                plt = plot_pose(original_image.cpu().numpy()[0].astype(np.uint8).copy(),output[0].cpu().numpy())
                plt.savefig(os.path.join(ckpt_dir, 'training/pred_sample_{0:05d}_seg'.format(j)+'.png'), 
               format='png', dpi=100)
                
                
                plt = plot_pose(original_image.cpu().numpy()[0].astype(np.uint8),target[0].cpu().numpy())
                plt.savefig(os.path.join(ckpt_dir, 'training/gt_sample_{0:05d}_seg'.format(j)+'.png'), 
               format='png', dpi=100)
                
                
                
#                 dt_joints = 
                
                
#                 annotation_pred_gt = depth[0][0].cpu()
#                 pred_depth = pred_depth[0][0].cpu()
#                 annotation_pred_gt = annotation_pred_gt/annotation_pred_gt.max()*255
#                 pred_depth = pred_depth/pred_depth.max()*255
#                 viz_tensor2 = torch.cat([annotation_pred_gt, pred_depth], axis=1)

#                 torchvision.utils.save_image(viz_tensor2, os.path.join(ckpt_dir, 
#                                                         'training/'+ str(b_index)+'viz_sample_{0:05d}_seg'.format(j)+'.png'))
                    

        print("Saving latest checkpoint to",ckpt_dir)
        torch.save(pose_module.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        if j%10==0:
            torch.save(pose_module.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
        scheduler.step()


if __name__ == "__main__":
    main()