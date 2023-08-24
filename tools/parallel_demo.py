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
import matplotlib
from random import choice
import os
import math
import argparse
from detectron2.utils.visualizer import ColorMode, Visualizer
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import *
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.depth_module import Depthmodule
from model.segment.transformer_decoder import seg_decorder
from model.segment.transformer_decoder_semantic import seg_decorder_open_word
from model.pose_module_heat import PoseModule_HeatMap
import yaml
from tools.utils import register_all_ade20k_full
from tools.train_instance_coco import dict2obj,instance_inference
from tools.parallel_generate_Semantic_CityScapes_AnyClass import semantic_inference,semantic_inference_huamn
from tools.train_pose_coco_heatmap import get_max_preds,plot_pose
import torch.optim as optim
from train import AttentionStore
import torch.nn.functional as F
import mmcv
from scipy.special import softmax
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import DatasetCatalog, MetadataCatalog
# from torch import autocast
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import random
classes = {
                0: 'road',
                1: 'sidewalk',
                2: 'building',
                3: 'wall',
                4: 'fence',
                5: 'pole',
                6: 'traffic light',
                7: 'traffic sign',
                8: 'vegetation',
                9: 'terrain',
                10: 'sky',
                11: 'person',
                12: 'rider',
                13: 'car',
                14: 'truck',
                15: 'bus',
                16: 'train',
                17: 'motorcycle',
                18: 'bicycle'
            }




DeepFashion_CATEGORIES = [
    {"color": [250, 250, 250], "id": 0, "isthing": 0, "name": "background"},
    {"color": [194, 255, 0], "id": 1, "isthing": 0, "name": "top"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "outer"},
    {"color": [80, 150, 20], "id": 3, "isthing": 0, "name": "skirt"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "dress"},
    {"color": [255, 5, 153], "id": 5, "isthing": 0, "name": "pants"},
    {"color": [0, 255, 245], "id": 6, "isthing": 0, "name": "leggings"},
    {"color": [204, 5, 255], "id": 7, "isthing": 1, "name": "headwear"},
    {"color": [255, 0, 102], "id": 8, "isthing": 1, "name": "eyeglass"},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "neckwear"},
    {"color": [224, 5, 255], "id": 10, "isthing": 1, "name": "belt"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "footwear"},
    {"color": [150, 5, 61], "id": 12, "isthing": 1, "name": "bag"},
    {"color": [0, 255, 112], "id": 13, "isthing": 0, "name": "hair"},
    {"color": [8, 255, 51], "id": 14, "isthing": 1, "name": "face"},
    {"color": [255, 250, 82], "id": 15, "isthing": 1, "name": "skin"},
    {"color": [200, 10, 22], "id": 16, "isthing": 1, "name": "ring"},
    {"color": [255, 200, 52], "id": 17, "isthing": 1, "name": "wrist wearing"},
    {"color": [155, 240, 82], "id": 18, "isthing": 1, "name": "socks"},
    {"color": [125, 6, 102], "id": 19, "isthing": 1, "name": "gloves"},
    {"color": [255, 222, 82], "id": 20, "isthing": 1, "name": "necklace"},
    {"color": [125, 60, 82], "id": 21, "isthing": 1, "name": "rompers"},
    {"color": [215, 29, 22], "id": 22, "isthing": 1, "name": "earrings"},
    {"color": [155, 110, 182], "id": 23, "isthing": 1, "name": "tie"},
]



def _get_deepfashion_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in DeepFashion_CATEGORIES]

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"].split(",")[0] for k in DeepFashion_CATEGORIES]
    
    color = [k["color"] for k in DeepFashion_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "color":color
    }
    return ret


def register_all_deepfashion_full(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    meta = _get_deepfashion_full_meta()
    for name, dirname in [("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"sem_seg_deepfashion"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors = meta["color"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg_deepfashion",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_deepfashion_full(_root)



def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def freeze_params(params):
    for param in params:
        param.requires_grad = False
        
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def plot_mask(img, masks, colors=None, alpha=0.8,indexlist=[0,1]) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    H,W= masks.shape[0],masks.shape[1]
    color_list=[[255,97,0],[128,42,42],[220,220,220],[255,153,18],[56,94,15],[127,255,212],[210,180,140],[221,160,221],[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],[0,255,128],[0,255,255],[0,128,255],[0,0,255],[128,0,255],[255,0,255],[255,0,128]]*6
    final_color_list=[np.array([[i]*512]*512) for i in color_list]
    
    background=np.ones(img.shape)*255
    count=0
    colors=final_color_list[indexlist[count]]
    for mask, color in zip(masks, colors):
        color=final_color_list[indexlist[count]]
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha,background*0.4+img*0.6 )
        count+=1
    return img.astype(np.uint8)

# color the depth, kitti magma_r, nyu jet
def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # TODO: remove hacks

    # for abs
    # vmin=1e-3
    # vmax=80

    # for relative
    # value[value<=vmin]=vmin

    # vmin=None
    # vmax=None

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :, :3] # bgr -> rgb
    rgb_value = value[..., ::-1]

    return rgb_value



def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
#                 print(item.reshape(len(prompts), -1, res, res, item.shape[-1]).shape)
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[0]
                out.append(cross_maps)

    out = torch.cat(out, dim=0)
    return out

def sub_processor(pid , opt):
    
    f = open(opt.config_semantic)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_sem = dict2obj(cfg)
    
    
    f = open(opt.config_depth_Virtual_KITTI_2)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_depth_Virtual_KITTI_2 = dict2obj(cfg)
    
    f = open(opt.config_depth_NYU)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_depth_NYU = dict2obj(cfg)
    
    
    f = open(opt.config_instance)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_instance = dict2obj(cfg)
    
    
    f = open(opt.config_open_word_seg)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_open_seg = dict2obj(cfg)
    
    f = open(opt.config_pose)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_pose = dict2obj(cfg)
    
    f = open(opt.config_human_seg)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_human_seg = dict2obj(cfg)
    

    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)
    
    seed_everything(opt.seed)
    

    MY_TOKEN = 'your huggingface key'
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    tokenizer = CLIPTokenizer.from_pretrained("./dataset/ckpts/imagenet/", subfolder="tokenizer")
    
    #VAE
    vae = AutoencoderKL.from_pretrained("./dataset/ckpts/imagenet/", subfolder="vae")
    freeze_params(vae.parameters())
    vae=vae.to(device)
    vae.eval()
    
    #UNet2DConditionModel UNet2D
    unet = UNet2D.from_pretrained("./dataset/ckpts/imagenet/", subfolder="unet")
    freeze_params(unet.parameters())
    unet=unet.to(device)
    unet.eval()
    
    text_encoder = CLIPTextModel.from_pretrained("./dataset/ckpts/imagenet/text_encoder")
    freeze_params(text_encoder.parameters())
    text_encoder=text_encoder.to(device)
    text_encoder.eval()
    
    scheduler = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device).scheduler
    
    # depth
    depth_module=Depthmodule(max_depth=cfg_depth_Virtual_KITTI_2.Depth_Decorder.max_depth).to(device)
    
    # depth NYU
    depth_NYU_module=Depthmodule(max_depth=cfg_depth_NYU.Depth_Decorder.max_depth).to(device)
    
    
    # semantic
    seg_semantic_model=seg_decorder(num_classes=cfg_sem.SEG_Decorder.num_classes, 
                           num_queries=cfg_sem.SEG_Decorder.num_queries).to(device)
    # instance
    seg_instance_model=seg_decorder(num_classes=cfg_instance.SEG_Decorder.num_classes, 
                           num_queries=cfg_instance.SEG_Decorder.num_queries).to(device)
    
    # open world segmentation
    seg_open_model=seg_decorder_open_word(num_classes=cfg_open_seg.SEG_Decorder.num_classes, 
                           num_queries=cfg_open_seg.SEG_Decorder.num_queries).to(device)
    
    # human pose segmentation
    human_pose_seg_model=seg_decorder(num_classes=cfg_human_seg.SEG_Decorder.num_classes, 
                           num_queries=cfg_human_seg.SEG_Decorder.num_queries).to(device)
    
    # pose estimation
    pose_module=PoseModule_HeatMap(cfg_pose).to(device)
    
    print('load semantic weight:',opt.grounding_ckpt_semantic)
    base_weights = torch.load(opt.grounding_ckpt_semantic, map_location="cpu")
    seg_semantic_model.load_state_dict(base_weights, strict=True)
        
    print('load instance weight:',opt.grounding_ckpt_instance)
    base_weights = torch.load(opt.grounding_ckpt_instance, map_location="cpu")
    seg_instance_model.load_state_dict(base_weights, strict=True)
    
    print('load depth weight:',opt.grounding_ckpt_depth_Virtual_KITTI_2)
    base_weights = torch.load(opt.grounding_ckpt_depth_Virtual_KITTI_2, map_location="cpu")
    depth_module.load_state_dict(base_weights, strict=True)
    
    print('load NYU depth weight:',opt.grounding_ckpt_depth_NYU)
    base_weights = torch.load(opt.grounding_ckpt_depth_NYU, map_location="cpu")
    depth_NYU_module.load_state_dict(base_weights, strict=True)
    
    print('load open segmentation weight:',opt.grounding_ckpt_semantic_open)
    base_weights = torch.load(opt.grounding_ckpt_semantic_open, map_location="cpu")
    seg_open_model.load_state_dict(base_weights, strict=True)
    
    print('load pose estimation weight:',opt.grounding_ckpt_human_pose)
    base_weights = torch.load(opt.grounding_ckpt_human_pose, map_location="cpu")
    pose_module.load_state_dict(base_weights, strict=True)
    
    print('load human segmentation weight:',opt.grounding_ckpt_human_seg)
    base_weights = torch.load(opt.grounding_ckpt_human_seg, map_location="cpu")
    human_pose_seg_model.load_state_dict(base_weights, strict=True)
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
    number_per_thread_num = int(int(opt.number_data)/opt.thread_num)
    seed = pid * (number_per_thread_num*2) + 200000

    with torch.no_grad():
        
        for n in range(number_per_thread_num):
#             
            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()

            seed = random.randint(1,1000000)
#             seed = 107474
            g_cpu = torch.Generator().manual_seed(seed)

            prompts = [opt.prompt]

            print("prompts:",prompts)
            print("open word:",opt.open_word)

            start_code = None
            if opt.fixed_code:
                print('start_code')
                start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


            if isinstance(prompts, tuple):
                prompts = list(prompts)

            images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,scheduler, prompts, controller,  num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, low_resource=LOW_RESOURCE, Train=False)
            ptp_utils.save_images(images_here,out_put = "{}/{}.jpg".format(outpath,seed))

            # depth
            diffusion_features=get_feature_dic()
            pred_depth=depth_module(diffusion_features,controller,prompts,tokenizer)
            pred_depth = (np.array(pred_depth[0].cpu()).astype('float32'))/(cfg_depth_Virtual_KITTI_2.Depth_Decorder.max_depth/8)*255
            mask = colorize(pred_depth,vmin=0,vmax=255,cmap='magma_r')
            mmcv.imwrite(mask.squeeze(), "{}/{}_VKITTI2_depth.png".format(outpath,seed))
            
            # depth_NYU_module
            diffusion_features=get_feature_dic()
            pred_depth=depth_NYU_module(diffusion_features,controller,prompts,tokenizer)
            pred_depth = (np.array(pred_depth[0].cpu()).astype('float32'))/(cfg_depth_NYU.Depth_Decorder.max_depth)*255
            mask = colorize(pred_depth,vmin=0,vmax=255,cmap='jet')
            mmcv.imwrite(mask.squeeze(), "{}/{}_NYU_depth.png".format(outpath,seed))
            
            
            
            #instance
            metadata = MetadataCatalog.get("coco_2017_train")
            outputs=seg_instance_model(diffusion_features,controller,prompts,tokenizer)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                                mask_pred_results,
                                size=(512, 512),
                                mode="bilinear",
                                align_corners=False,
                                )
            instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_results[0], mask_pred_results[0], cfg_instance.SEG_Decorder.num_classes, 2).to(torch.device("cpu"))
            pred_masks_ins = instance_r.pred_masks.numpy().astype(np.uint8)
            pred_boxes = instance_r.pred_boxes
            scores = instance_r.scores 
            pred_classes = instance_r.pred_classes 
            
            image = images_here.copy()[0]
            visualizer = Visualizer(image, metadata, instance_mode=ColorMode.SEGMENTATION)
            vis_output = visualizer.draw_instance_predictions(instance_r)
            vis_output.save("{}/{}_instance.png".format(outpath,seed))  

            # open world segmentation
            query_text = opt.open_word
                    
            text_input = tokenizer(
            query_text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
            text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]
            c_split = tokenizer.tokenize(query_text)

            class_embedding=text_embeddings

            if class_embedding.size()[1] > 1:
                class_embedding = torch.unsqueeze(class_embedding.mean(1),1)
            outputs=seg_open_model(diffusion_features,controller,prompts,tokenizer,class_embedding)  
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                                mask_pred_results,
                                size=(512, 512),
                                mode="bilinear",
                                align_corners=False,
                                )
            instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_results[0], mask_pred_results[0], cfg_open_seg.SEG_Decorder.num_classes, 1).to(torch.device("cpu"))
            pred_masks = instance_r.pred_masks.numpy().astype(np.uint8)
            pred_boxes = instance_r.pred_boxes
            scores = instance_r.scores 
            pred_classes = instance_r.pred_classes 
            
            image = images_here.copy()[0]
            visualizer = Visualizer(image, metadata, instance_mode=ColorMode.SEGMENTATION)
            vis_output = visualizer.draw_instance_predictions(instance_r)
            vis_output.save("{}/{}_open_world.png".format(outpath,seed))       
            
            
            #semantic  cityscapes
            outputs=seg_semantic_model(diffusion_features,controller,prompts,tokenizer)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                                mask_pred_results,
                                size=(512, 512),
                                mode="bilinear",
                                align_corners=False,
                                )
            label_pred_prob = retry_if_cuda_oom(semantic_inference)(mask_cls_results[0], mask_pred_results[0])
            label_pred_prob = torch.argmax(label_pred_prob, axis=0)
            label_pred_prob = label_pred_prob.cpu().numpy()
            
            image = images_here.copy()[0]
            metadata = MetadataCatalog.get("sem_seg_citysscapes_demo")
            visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_sem_seg(label_pred_prob, alpha=0.5)
            vis_output.save("{}/{}_semantic.png".format(outpath,seed))
            
            
            
            
            
            # human pose segmentation
            outputs=human_pose_seg_model(diffusion_features,controller,prompts,tokenizer)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                                mask_pred_results,
                                size=(512, 512),
                                mode="bilinear",
                                align_corners=False,
                                )
            label_pred_prob = retry_if_cuda_oom(semantic_inference_huamn)(mask_cls_results[0], mask_pred_results[0])
            label_pred_prob = torch.argmax(label_pred_prob, axis=0)
            
            label_pred_prob = label_pred_prob.cpu().numpy()* pred_masks[0]

            image = images_here.copy()[0]
            metadata = MetadataCatalog.get("sem_seg_deepfashion")
            visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_sem_seg(label_pred_prob, alpha=0.5)
            vis_output.save("{}/{}_deepfashion.png".format(outpath,seed))     
            
            
            
            
            # pose estimation
            outputs=pose_module(diffusion_features,controller,prompts,tokenizer)
            outputs = F.interpolate(outputs, size=512, mode='bilinear', align_corners=False)
            outputs = outputs.detach().cpu().numpy()
            coords, maxvals = get_max_preds(outputs)

            heatmap_height = outputs.shape[2]
            heatmap_width = outputs.shape[3]
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = outputs[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                        diff = np.array(
                            [
                                hm[py][px+1] - hm[py][px-1],
                                hm[py+1][px]-hm[py-1][px]
                            ]
                        )


                        if maxvals[n][p][0]<0.2:
                            coords[n][p] = [0,0]
                        else:
                            coords[n][p] += np.sign(diff) * .25

            plt = plot_pose( images_here[0],  coords[0], maxvals[0])
            plt.savefig("{}/{}_human_pose.png".format(outpath,seed), format='png', dpi=100)
                
            seed+=1

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a lion on a mountain top at sunset",
        help="the prompt to render"
    )
    parser.add_argument(
        "--open_word",
        type=str,
        nargs="?",
        default="a photo of a lion on a mountain top at sunset",
        help="open_word key"
    )
    parser.add_argument(
        "--category",
        type=str,
        nargs="?",
        default="lion",
        help="the category to ground"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./DataDiffusion/VOC/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=8,
        help="number of threads",
    )
    
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--prompt_root",
        type=str,
        help="uses prompt",
        default="./dataset/Prompts_From_GPT/cityscapes"
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--number_data",
        type=int,
        default=20,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config_semantic",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--config_open_word_seg",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--config_depth_Virtual_KITTI_2",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--config_depth_NYU",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--config_instance",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--config_pose",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--config_human_seg",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which human segmentation",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="stable_diffusion.ckpt",
        help="path to checkpoint of stable diffusion model",
    )
    parser.add_argument(
        "--grounding_ckpt_instance",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--grounding_ckpt_semantic",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--grounding_ckpt_semantic_open",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--grounding_ckpt_human_pose",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--grounding_ckpt_depth_NYU",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--grounding_ckpt_depth_Virtual_KITTI_2",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    
    parser.add_argument(
        "--grounding_ckpt_human_seg",
        type=str,
        default="grounding_module.pth",
        help="path to checkpoint of grounding module",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    
    import multiprocessing as mp
    import threading
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    processes = []
    
    
#     _root = os.getenv("DETECTRON2_DATASETS", "datasets")
#     register_all_ade20k_full(_root)
        
    print('Start Generation')
    for i in range(opt.thread_num):

        p = mp.Process(target=sub_processor, args=(i, opt))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    
 

    


if __name__ == "__main__":
    main()
