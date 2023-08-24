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
import os
import argparse
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import *
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.pose_module import PoseModule
from tools.train_pose_coco import plot_pose
import yaml
from tools.train_instance_coco import dict2obj
import torch.optim as optim
from train import AttentionStore
import torch.nn.functional as F
from scipy.special import softmax
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
# from torch import autocast
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from random import choice
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
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)
    
    seed_everything(opt.seed)
    

    task = "depth"
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
    
    pose_module=PoseModule(cfg).to(device)
    
    
    print('load weight:',opt.grounding_ckpt)
    base_weights = torch.load(opt.grounding_ckpt, map_location="cpu")
    pose_module.load_state_dict(base_weights, strict=True)

        

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    Image_path = os.path.join(outpath, "Image")
    os.makedirs(Image_path, exist_ok=True)
        
    Mask_path = os.path.join(outpath, "Annotation")
    os.makedirs(Mask_path, exist_ok=True)
    
    batch_size = opt.n_samples

    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
    number_per_thread_num = int(int(opt.number_data)/opt.thread_num)
    seed = pid * (number_per_thread_num*2) + 200000
    
    
    body = ["","whole body"]
    
    prompts_list = ["photo of a {} is walking, {}",
                        "photo of a {} is eating, {}",
                        "photo of a {} is play, {}",  
                       
                        #sports
                        "photo of a {} is dancing, {}",
                        "photo of a {} is doing Yoga, {}","photo of a {} is doing Fitness, {}",
                        
                        "photo of a {} is doing High jumping, {}","photo of a {} is Cycling, {}",
                        "photo of a {} is running","photo of a {} is Fishing, {}",
                        "photo of a {} is Climbing, {}",
                        
                        # scenario
                        "photo of a {} is walking in the street, {}","photo of a {} is in the road, {}",
                        "photo of a {} is playing at home, {}",
                        "photo of a {} is in on the mountain, {}","photo of a {} is in on the mountain, {}",
                        "photo of a {} is crossing a road, {}","photo of a {} is sitting, {}"]
    names = ['person',"man","woman","child","boy","girl","old man","teenager"]
    
    #prompt candidate
    sub_classes_list=[]
    for b in body:
        for name in names:
            for prompts_line in prompts_list:
                sub_classes_list.append(prompts_line.format(name,b))
                    
    


    print("prompt candidates:",len(sub_classes_list))

    with torch.no_grad():

        for n in range(number_per_thread_num):

            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()


            g_cpu = torch.Generator().manual_seed(seed)

            prompts = [choice(sub_classes_list)]

            print("prompts:",prompts)


            start_code = None
            if opt.fixed_code:
                print('start_code')
                start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


            if isinstance(prompts, tuple):
                prompts = list(prompts)

            images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,scheduler, prompts, controller,  num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, low_resource=LOW_RESOURCE, Train=False)
            ptp_utils.save_images(images_here,out_put = "{}/{}.jpg".format(Image_path,seed))

            # pose head
            diffusion_features=get_feature_dic()
            output_x, output_y=pose_module(diffusion_features,controller,prompts,tokenizer)
            
            output_x = F.softmax(output_x,dim=2)
            output_y = F.softmax(output_y,dim=2)
            
            
            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            output = torch.ones([1,preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, 1.0))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, 1.0))
            output = output.cpu().numpy()
            
            plt = plot_pose( images_here[0],  output[0])
            plt.savefig("{}/{}.png".format(Mask_path,seed), format='png', dpi=100)
            
#             print(output)
                
#             print(np.unique(pred_depth))
#             cv2.imwrite("{}/{}.png".format(Mask_path,seed), pred_depth)
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
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="stable_diffusion.ckpt",
        help="path to checkpoint of stable diffusion model",
    )
    parser.add_argument(
        "--grounding_ckpt",
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
#     per_thread_video_num = int(len(coco_category_list)/thread_num)
#     thread_num=8
    print('Start Generation')
    for i in range(opt.thread_num):
#         if i == thread_num - 1:
#             sub_video_list = coco_category_list[i * per_thread_video_num:]
#         else:
#             sub_video_list = coco_category_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(i, opt))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    
 

    


if __name__ == "__main__":
    main()
