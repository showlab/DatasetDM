from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
# from diffusers import StableDiffusionPipeline
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
from model.seg_module import segmodule
import torch.optim as optim
import torch.nn.functional as F
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
        
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
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
        "--dataset", type=str, default="VOC", help="dataset: VOC/MaskCut"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default="Test"
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    
    # dataset
    if opt.dataset == "VOC":
        dataset = Semantic_VOC(
            set="train",
        )
#     elif opt.dataset == "MaskCut":
#         dataset = Semantic_MaskCut(
#             set="train",
#         )
    else:
        return
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print('***********************   begin   **********************************')
    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    learning_rate = 1e-5 
    adam_weight_decay = 1e-4
    total_epoch = 5000
    
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
    
    
    
    
    seg_model=segmodule().to(device)
    
    noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    print("learning_rate:",learning_rate)
    g_optim = optim.Adam(
            [{"params": seg_model.parameters()},],
            lr=learning_rate
          )
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    start_code = None
    
    
    
#     MY_TOKEN = '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'
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
#     tokenizer = ldm_stable.tokenizer
    
    
    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
    for j in range(total_epoch):
        print('Epoch ' +  str(j) + '/' + str(total_epoch))
        
        for step, batch in enumerate(dataloader):
            g_cpu = torch.Generator().manual_seed(random.randint(1, 10000000))
            
            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()
            
            
            image = batch["image"]
            mask = batch["mask"]
            prompts = batch["prompt"]
            classs = batch["class"]
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
#             extra_set_kwargs = {"offset": 1}
#             noise_scheduler.set_timesteps(50, **extra_set_kwargs)
            noise_scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
            stepss = noise_scheduler.timesteps[-1]
            timesteps = torch.ones_like(timesteps) * stepss
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            start_code = noisy_latents.to(latents.device)
    
            
                           
            images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,noise_scheduler, prompts, controller, latent=start_code, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, low_resource=LOW_RESOURCE, Train=True)
                           
            
#             images_here, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=start_code, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=g_cpu, low_resource=LOW_RESOURCE, Train=True)
            
            if step%100 ==0:
                ptp_utils.save_images(images_here,out_put = (os.path.join(ckpt_dir,  'training/'+'viz_sample_{0:05d}'.format(step)+classs[0]+".png")))
                Image.fromarray(original_image.cpu().numpy()[0].astype(np.uint8)).save(os.path.join(ckpt_dir, 'training/'+ 'original_sample_{0:05d}'.format(step)+classs[0]+".png"))
                
                
            # train segmentation
#             query_text="a photograph of a "+classs[0]
            query_text=classs[0]
            
            text_input = tokenizer(
            query_text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            )
            text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]

    
            c_split = tokenizer.tokenize(query_text)

#             sen_text_embedding=tokenizer.tokenize(query_text)

#             class_embedding=text_embeddings[:,5:len(c_split)+1,:]
            class_embedding=text_embeddings

            if class_embedding.size()[1] > 1:
                class_embedding = torch.unsqueeze(class_embedding.mean(1),1)
            
            diffusion_features=get_feature_dic()
            seg=mask.unsqueeze(0).float().cuda()
            total_pred_seg=seg_model(diffusion_features,controller,prompts,tokenizer,classs,class_embedding)
            
#             loss = cross_entropy2d(total_pred_seg,seg)
            loss = loss_fn(total_pred_seg, seg)
            
            g_optim.zero_grad()
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}, lr: {3:0.6f}".format(step, len(dataloader), loss, float(g_optim.state_dict()['param_groups'][0]['lr'])))
            loss.backward()
            g_optim.step()
            
            if step%100 ==0:
#                 label_pred_prob = F.log_softmax(total_pred_seg[0], dim=0)
#                 label_pred_prob = torch.argmax(label_pred_prob, axis=0)
                
                label_pred_prob = torch.sigmoid(total_pred_seg)
                label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
                label_pred_mask[label_pred_prob>0.5] = 1
                annotation_pred = label_pred_mask[0][0]
                
                annotation_pred_gt = mask[0].cuda().float()
                
#                 print(annotation_pred_gt.shape, label_pred_prob.shape)
                viz_tensor2 = torch.cat([annotation_pred_gt, annotation_pred], axis=1)

                torchvision.utils.save_image(viz_tensor2, os.path.join(ckpt_dir, 
                                                        'training/'+ 'viz_sample_{0:05d}_seg'.format(step)+classs[0]+'.png'), normalize=True, scale_each=True)
                
#             print(total_pred_seg.shape)
        print("Saving latest checkpoint to",ckpt_dir)
        torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        if j%20==0:
            torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))                                    


if __name__ == "__main__":
    main()