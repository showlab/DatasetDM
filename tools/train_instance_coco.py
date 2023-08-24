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
from model.segment.transformer_decoder import seg_decorder
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
        
def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
#         print(mask_cls.shape)
#         mask_cls = F.softmax(mask_cls, dim=-1)
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    for i in range(1,semseg.shape[0]):
        if (semseg[i]*(semseg[i]>0.5)).sum()<5000:
            semseg[i] = 0

    return semseg

def instance_inference(mask_cls, mask_pred,class_n = 2,test_topk_per_image=20,query_n = 100):
    # mask_pred is already processed to have the same shape as original input
    image_size = mask_pred.shape[-2:]

    # [Q, K]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(class_n , device=mask_cls.device).unsqueeze(0).repeat(query_n, 1).flatten(0, 1)
    # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // class_n
    # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
#     print(topk_indices)
    mask_pred = mask_pred[topk_indices]


    result = Instances(image_size)
    # mask (before sigmoid)
    result.pred_masks = (mask_pred > 0).float()
    result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    # Uncomment the following to get boxes from masks (this is slow)
    # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    # calculate average mask prob
    mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image
    return result
    
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
    
    # dataset
    if opt.dataset == "VOC":
        dataset = Semantic_VOC(
            set="train",
        )
    elif opt.dataset == "Cityscapes":
        dataset = Semantic_Cityscapes(
            set="train",
        )
    elif opt.dataset == "COCO":
        dataset = Instance_COCO(
            set="train",image_limitation = opt.image_limitation
        )   
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
    
    # segmentation decorder
    # building criterion
    matcher = HungarianMatcher(
        cost_class=cfg.SEG_Decorder.CLASS_WEIGHT,
        cost_mask=cfg.SEG_Decorder.MASK_WEIGHT,
        cost_dice=cfg.SEG_Decorder.DICE_WEIGHT,
        num_points=cfg.SEG_Decorder.TRAIN_NUM_POINTS,
    )
        
    losses = ["labels", "masks"]
    weight_dict = {"loss_ce": cfg.SEG_Decorder.CLASS_WEIGHT, "loss_mask": cfg.SEG_Decorder.MASK_WEIGHT, "loss_dice": cfg.SEG_Decorder.DICE_WEIGHT}
    criterion = SetCriterion(
            cfg.SEG_Decorder.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.SEG_Decorder.no_object_weight,
            losses=losses,
            num_points=cfg.SEG_Decorder.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.SEG_Decorder.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.SEG_Decorder.IMPORTANCE_SAMPLE_RATIO,
        )
    
    
    seg_model=seg_decorder(num_classes=cfg.SEG_Decorder.num_classes, 
                           num_queries=cfg.SEG_Decorder.num_queries).to(device)
    
    noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    print("learning_rate:",learning_rate)
    g_optim = optim.Adam(
            [{"params": seg_model.parameters()},],
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
            
            
            image = batch["image"]
            instances = batch["instances"]
            prompts = batch["prompt"]
            class_name = batch["classes_str"]
            original_image = batch["original_image"]
            
            # [1, 19, 512, 512]
#             print(instances["gt_masks"].shape)
            
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
    
            
            try:               
                images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,noise_scheduler, prompts, controller, latent=start_code, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, low_resource=LOW_RESOURCE, Train=True)
            except:
                continue
                           
            
            if step%100 ==0:
                ptp_utils.save_images(images_here,out_put = (os.path.join(ckpt_dir,  'training/'+'viz_sample_{0:05d}'.format(step)+".png")))
                Image.fromarray(original_image.cpu().numpy()[0].astype(np.uint8)).save(os.path.join(ckpt_dir, 'training/'+ 'original_sample_{0:05d}'.format(step)+".png"))
                
                
            # train segmentation
            
#             query_text=class_name[0]
#             text_input = tokenizer(
#             query_text,
#             padding="max_length",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt",
#             )
#             text_embeddings = text_encoder(text_input.input_ids.to(latents.device))[0]
#             print(text_embeddings.shape)
#             if text_embeddings.size()[1] > 1:
#                 text_embeddings = torch.unsqueeze(text_embeddings.mean(1),1)
                
            diffusion_features=get_feature_dic()
            outputs=seg_model(diffusion_features,controller,prompts,tokenizer)
            
            # bipartite matching-based loss
    
            losses = criterion(outputs, batch)
            loss = losses['loss_ce']*cfg.SEG_Decorder.CLASS_WEIGHT + losses['loss_mask']*cfg.SEG_Decorder.MASK_WEIGHT + losses['loss_dice']*cfg.SEG_Decorder.DICE_WEIGHT
#             loss = loss_fn(total_pred_seg, seg)
            
            g_optim.zero_grad()
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}, loss_ce: {3:0.4f},loss_mask: {4:0.4f},loss_dice: {5:0.4f}, lr: {6:0.6f}, prompt: ".format(step, len(dataloader), loss, losses['loss_ce']*cfg.SEG_Decorder.CLASS_WEIGHT,losses['loss_mask']*cfg.SEG_Decorder.MASK_WEIGHT, losses['loss_dice']*cfg.SEG_Decorder.DICE_WEIGHT,float(g_optim.state_dict()['param_groups'][0]['lr'])),prompts)
            loss.backward()
            g_optim.step()
            
            if step%100 ==0 or losses['loss_mask']*cfg.SEG_Decorder.MASK_WEIGHT<0.3:
                
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_pred_results = F.interpolate(
                                    mask_pred_results,
                                    size=(image.shape[-2], image.shape[-1]),
                                    mode="bilinear",
                                    align_corners=False,
                                    )
                
                for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):

                    height = width = 512
#                     if self.sem_seg_postprocess_before_inference:
#                         mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
#                             mask_pred_result, image_size, height, width
#                         )
#                         mask_cls_result = mask_cls_result.to(mask_pred_result)
                    if cfg.SEG_Decorder.task=="semantic":
                        label_pred_prob = retry_if_cuda_oom(semantic_inference)(mask_cls_result, mask_pred_result)
                        label_pred_prob = torch.argmax(label_pred_prob, axis=0)
                        label_pred_prob = label_pred_prob.cpu().numpy()
                        cv2.imwrite(os.path.join(ckpt_dir, 'training/'+'viz_sample_{0:05d}_seg'.format(step)+'.png'),label_pred_prob*100)
                        
                    elif cfg.SEG_Decorder.task=="instance":
                        instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result, cfg.SEG_Decorder.num_classes)
                        pred_masks = instance_r.pred_masks.cpu().numpy().astype(np.uint8)
                        pred_boxes = instance_r.pred_boxes
                        scores = instance_r.scores 
                        pred_classes = instance_r.pred_classes 
                        
                        vis_i  = original_image.cpu().numpy()[0].astype(np.uint8)
                        vis_i = np.array(vis_i,dtype=float)

                        for m in pred_masks:
#                             print(vis_i.shape)
                            vis_i,_ = mask_image(vis_i,m)
#                         print(pred_masks.shape)
                        cv2.imwrite(os.path.join(ckpt_dir, 'training/'+'viz_sample_{0:05d}_seg'.format(step)+'.jpg'),vis_i)
                        try:
                            gt_mask = instances["gt_masks"][0][0]
                            gt_mask = np.array(gt_mask)*255
                            cv2.imwrite(os.path.join(ckpt_dir, 'training/'+'viz_sample_{0:05d}_gt_seg'.format(step)+'.jpg'),gt_mask)
                        except:
                            pass
    
#                 label_pred_prob = torch.sigmoid(total_pred_seg)
#                 label_pred_mask = torch.zeros_like(label_pred_prob, dtype=torch.float32)
#                 label_pred_mask[label_pred_prob>0.5] = 1
#                 annotation_pred = label_pred_mask[0][0]
                
#                 annotation_pred_gt = mask[0].cuda().float()
#                 viz_tensor2 = label_pred_prob
#                 print(annotation_pred_gt.shape, label_pred_prob.shape)
#                 viz_tensor2 = torch.cat([annotation_pred_gt, annotation_pred], axis=1)
                
#                 torchvision.utils.save_image(viz_tensor2.Long(), os.path.join(ckpt_dir, 
#                                                         'training/'+ 'viz_sample_{0:05d}_seg'.format(step)+'.png'), normalize=True, scale_each=True)
                
#             print(total_pred_seg.shape)
        print("Saving latest checkpoint to",ckpt_dir)
        torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        if j%10==0:
            torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
        scheduler.step()


if __name__ == "__main__":
    main()