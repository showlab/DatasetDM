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
import yaml
import argparse
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import *
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.segment.transformer_decoder_semantic import seg_decorder_open_word
import torch.optim as optim
from tools.train_instance_coco import dict2obj,instance_inference
from train import AttentionStore
import torch.nn.functional as F
from scipy.special import softmax
from detectron2.utils.memory import retry_if_cuda_oom
from random import choice
classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

name_to_idx = {
    'background':0,
    'aeroplane':1,
    'bicycle':2,
    'bird':3,
    'boat':4,
    'bottle':5,
    'bus':6,
    'car':7,
    'cat':8,
    'chair':9,
    'cow':10,
    'diningtable':11,
    'dog':12,
    'horse':13,
    'motorbike':14,
    'person':15,
    'pottedplant':16,
    'sheep':17,
    'sofa':18,
    'train':19,
    'tvmonitor':20
}


#,'person'

classes_check = {
    0: [],
    1: ['aeroplane'],
    2: ['bicycle'],
    3: ['bird'],
    4: ['boat'],
    5: ['bottle'],
    6: ['bus'],
    7: ['car'],
    8: ['cat'],
    9: ['chair'],
    10: ['cow'],
    11: ['diningtable'],
    12: ['dog'],
    13: ['horse'],
    14: ['motorbike'],
    15: ['person'],
    16: ['pottedplant'],
    17: ['sheep'],
    18: ['sofa'],
    19: ['train'],
    20: ['tvmonitor']
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
    torch.cuda.set_device(pid)
    text = 'processor %d' % pid
    print(text)
    
    seed_everything(opt.seed)
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
    MY_TOKEN = 'your huggingface key'
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
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
    
    seg_model=seg_decorder_open_word(num_classes=cfg.SEG_Decorder.num_classes, 
                           num_queries=cfg.SEG_Decorder.num_queries).to(device)
    
    
    print('load weight:',opt.grounding_ckpt)
    base_weights = torch.load(opt.grounding_ckpt, map_location="cpu")
    try:
        seg_model.load_state_dict(base_weights, strict=True)
    except:
        new_state_dict = OrderedDict()
        for k, v in base_weights.items():
            name = k[7:]   # remove `vgg.`
            new_state_dict[name] = v 
        seg_module.load_state_dict(new_state_dict, strict=True)
        

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    Image_path = os.path.join(outpath, "Image")
    os.makedirs(Image_path, exist_ok=True)
        
    Mask_path = os.path.join(outpath, "Mask")
    os.makedirs(Mask_path, exist_ok=True)
    
    batch_size = opt.n_samples
#     precision_scope = autocast if opt.precision=="autocast" else nullcontext
    
    prompt_for_aeroplane = [""]
    prompt_for_bicycle = ["photo of a bicycle in the street", "photo of a bicycle", "photo of a bicycle in the road",   "photo of a bicycle in the street at night",      "photo of the back of a bicycle"]
    prompt_for_person = ["photo of a {} is walking",
                        "photo of a {} is eating",
                        "photo of a {} is play",  
                       
                        #sports
                        "photo of a {} is playing baseball","photo of a {} is playing Basketball", 
                        "photo of a {} is playing Badminton","photo of a {} is Swimming",
                        "photo of a {} is playing Bodybuilding","photo of a {} is playing Bowling",
                        "photo of a {} is dancing","photo of a {} is playing Football",
                        "photo of a {} is playing Golf","photo of a {} is playing Frisbee",
                        "photo of a {} is Skiing", "photo of a {} is playing Table Tennis",
                        "photo of a {} is doing Yoga","photo of a {} is doing Fitness",
                        "photo of a {} is doing Rugby","photo of a {} is doing Wrestling",
                        "photo of a {} is doing High jumping","photo of a {} is Cycling",
                        "photo of a {} is running","photo of a {} is Fishing",
                        "photo of a {} is doing Judo","photo of a {} is Climbing",
                        
                        # scenario
                        "photo of a {} is walking in the street","photo of a {} is in the road",
                        "photo of a {} is playing at home","photo of a {} is in the shopping center",
                        "photo of a {} is in on the mountain","photo of a {} is in on the mountain",
                        "photo of a {} is crossing a road","photo of a {} is sitting",
                        "photo of a {} is sitting at home", "photo of a {} is playing at sofa",
                        "photo of a {} is playing at home","photo of back of a {}",
                        
                        
                        #others  
                        "photo of a {} is cooking",
                       "photo of a {}",
                       "photo of arm of a {}",
                        "photo of foot of a {}",
                       "photo of a {} is running"]
    prompt_for_bird = ["Masai Ostrich","Macaw","Eagle","Duck","Hen","Parrot","Peacock","Dove","Stork","Swan","Pigeon","Goose",
                        "Pelican","Macaw","Parakeet","Finches","Crow","Raven","Vulture","Hawk","Crane","Penguin", "Hummingbird",
                        "Sparrow","Woodpecker","Hornbill","Owl","Myna","Cuckoo","Turkey","Quail","Ostrich","Emu","Cockatiel"
                        ,"Kingfisher","Kite","Cockatoo","Nightingale","Blue jay","Magpie","Goldfinch","Robin","Swallow",
                        "Starling","Pheasant","Toucan","Canary","Seagull","Heron","Potoo","Bush warbler","Barn swallow",
                        "Cassowary","Mallard","Common swift","Falcon","Megapode","Spoonbill","Ospreys","Coot","Rail",
                        "Budgerigar","Wren","Lark","Sandpiper","Arctic tern","Lovebird","Conure","Rallidae","Bee-eater",
                        "Grebe","Guinea fowl","Passerine","Albatross","Moa","Kiwi","Nightjar","Oilbird","Gannet","Thrush",
                        "Avocet","Catbird","Bluebird","Roadrunner","Dunnock","Northern cardinal","Teal",
                        "Northern shoveler","Gadwall","Northern pintail",
                        "Hoatzin","Kestrel","Oriole","Partridge","Tailorbird","Wagtail","Weaverbird","Skylark"]
    prompt_for_boat = ["Fishing","Dinghy","Deck","Bowrider","Catamaran","Cuddy Cabins","Centre Console","House","Trawler","Cabin Cruiser","Game","Motor Yacht","Runabout","Jet","Pontoon","Sedan Bridge","","",""]
    prompt_for_bottle = [""]
    prompt_for_bus = ["Coach","Motor","School","Shuttle","Mini",
                       "Minicoach","Double-decker","Single-decker","Low-floor","Step-entrance","Trolley",
                        "Articulated","Guided","Neighbourhood","Gyrobus","Hybrid","Police",
                       "Open top","Electric","Transit","Tour","Commuter","Party","","",""]
    prompt_for_car = ["SEDAN","COUPE","SPORTS","STATION WAGON","HATCHBACK",
                       "CONVERTIBLE","SPORT-UTILITY VEHICLE","MINIVAN","PICKUP TRUCK","IT DOESN'T STOP THERE","","","",""]
    prompt_for_cat = ["Abyssinian","American Bobtail","American Curl","American Shorthair","American Wirehair",
                       "Balinese-Javanese","Bengal","Birman","Bombay","British Shorthair","Burmese","Chartreux Cat",
                        "Cornish Rex","Devon Rex","Egyptian Mau","European Burmese","Exotic Shorthair","Havana Brown",
                       "Himalayan","Japanese Bobtail","Korat","LaPerm", "Maine Coon","Manx","Munchkin",
                       "Norwegian Forest","Ocicat","Oriental","Persian","Peterbald","Pixiebob","Ragamuffin",
                       "Ragdoll","Russian Blue","Savannah","Scottish Fold","Selkirk Rex","Siamese","Siberian",
                       "Singapura","Somali","Sphynx","Tonkinese","Toyger","Turkish Angora","Turkish Van","","","","","","","",""]
    prompt_for_chair = [""]
    prompt_for_cow = ["Ayrshire","Brown Swiss","Guernsey","Holstein","Jersey",
                       "Milking Shorthorn","Red & White", "small Ayrshire","small Brown Swiss","small Guernsey",
                       "small Holstein","small Jersey",
                       "small Milking Shorthorn","small Red & White", "small ",
                       "small","small","small","small","small","small","","","","","","","","","","","","","","","",""]
    prompt_for_diningtable = ["photo of a table"]
    prompt_for_dog = ["Affenpinscher","Afghan Hound","Airedale Terrier","Akita","Alaskan Malamute",
                       "American English Coonhound","American Eskimo Dog","American Foxhound","American Staffordshire Terrier","American Water Spaniel","Anatolian Shepherd","Australian Cattle","Australian Shepherd","Australian Terrier",
                      "Basenji","Basset Hound","Beagle","Bearded Collie","Beauceron","Bedlington Terrier","Belgian Malinois",
                      "Belgian","Belgian Tervuren","Berger Picard","Bernese Mountain","Bichon Frise","Black and Tan Coonhound"
                      ,"Black Russian Terrier","Bloodhound","Bluetick Coonhound","Boerboel","Border Collie","Border Terrier"
                      ,"Borzoi","Boston Terrier","Bouvier des Flandres","Boxer","Boykin Spaniel","Briard","Brittany","Brussels Griffon"
                      ,"Bull Terrier","Bull","Bullmastiff","Cairn Terrier","Canaan","Cane Corso","Cardigan Welsh Corgi","Cavalier King Charles Spaniel"
                      ,"Cesky Terrier","Chesapeake Bay Retriever","Chihuahua","Chinese Crested","Chinese Shar-Pei","Chinook",
                      "Chow Chow","Cirneco dell’Etna","Clumber Spaniel","Cocker Spaniel","Collie","Corgi","Coton de Tulear",
                       "Curly-Coated Retriever","Dachshund","Dalmatian","Dandie Dinmont Terrier","Doberman Pinscher",
                       "Dogue de Bordeaux","English Cocker Spaniel","English Foxhound","English Setter","Whippet","Wire Fox Terrier"
                      ,"Wirehaired Pointing Griffon","Wirehaired Vizsla","Xoloitzcuintli","Finnish Spitz","Glen of Imaal Terrier",
                      "Great Pyrenees","Irish Setter","Irish Water Spaniel","Keeshond","Labrador Retriever","Lagotto Romagnolo",
                      "Leonberger","Manchester Terrier","Mastiff","Miniature Bull Terrier","Miniature Pinscher","Neapolitan Mastiff",
                      "Norwegian Elkhound","Nova Scotia Duck Tolling Retriever","Otterhound","Papillon",
                       "","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
    prompt_for_horse = ["American Quarter","Arabian","Thoroughbred","Appaloosa","Morgan",
                       "Warmbloods","Ponies","Grade","Gaited Breeds","Draft Breeds","","","","","","","","",""]
    
    prompt_for_motorbike = ["Cruisers","Sportbikes","Standard & Naked ","Adventure","Dual Sports & Enduros",
                       "Dirtbikes","Electric","Choppers","Touring","Sport Touring","Vintage & Customs",
                        "Modern Classics","Commuters & Minis","Scooters","","","","","","","","",""]
    prompt_for_pottedplant = ["Spider","Aloe Vera","Peace Lily","Jade","African Violet",
                       "Weeping Fig","Baby Rubber","Bromeliads","Calathea","Dracaena","Ficus","Orchid","Eternal Flame",
                               "Rattlesnake","Pin Stripe Calathea","Barberton Daisy","Areca Palm",
                             "Corn","","","","","","","","",""]
    prompt_for_sheep = ["Merino Wool","Rambouillet","Suffolk","Hampshire","Katahdin",
                       "Dorper","Dorset","Southdown","Karakul","Lincoln","Icelandic","Navajo Churro","","","","","","",""]
    prompt_for_sofa = [""]
    prompt_for_train = [""]
    prompt_for_tvmonitor = [""]
    
#     sampler,seg_module,model = accelerator.prepare(sampler,seg_module,model)
    number_per_thread_num = int(int(opt.n_each_class)/opt.thread_num)
    seed = pid * (number_per_thread_num*2) + 200000
#     seed = 0
    
    map_dict = {"aeroplane":prompt_for_aeroplane,"bird":prompt_for_bird,"boat":prompt_for_boat,"bottle":prompt_for_bottle,
               "bus":prompt_for_bus,"car":prompt_for_car,"cat":prompt_for_cat,"chair":prompt_for_chair,"cow":prompt_for_cow,
               "dog":prompt_for_dog,"horse":prompt_for_horse,
                "motorbike":prompt_for_motorbike,"pottedplant":prompt_for_pottedplant,"sheep":prompt_for_sheep,
               "sofa":prompt_for_sofa,"train":prompt_for_train,"tvmonitor":prompt_for_tvmonitor}
    
    sub_classes = ["aeroplane","bird","boat","bottle","bus","car","cat","chair","cow","dog","horse","motorbike",
                  "pottedplant","sheep","sofa","train","tvmonitor"]
    
    debug_class = {

    11: 'diningtable',

}
    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    
#     classes
    for idx in classes:
        
        if idx==0:
            continue
        
                
        class_target = classes[idx]
    
        if class_target in sub_classes:
            prompts_list = ["a photo of a {} {}"]
            sub_cls = map_dict[class_target]
            sub_classes_list = [prompts_list[0].format(i,class_target) for i in sub_cls]
        elif class_target == "bicycle":
            sub_classes_list = prompt_for_bicycle
        elif class_target == "person":
            names = ['person',"man","woman","child","boy","girl","old man","teenager"]
            sub_classes_list = []
            for name in names:
                for prompts_line in prompt_for_person:
                    sub_classes_list.append(prompts_line.format(name))
        elif class_target == "diningtable":
            sub_classes_list = prompt_for_diningtable
        
        
        prompt_path = os.path.join(opt.prompt_root,class_target+".txt")
        
#         #read prompt txt for each class 
        if os.path.exists(prompt_path):
            f2 = open(prompt_path,"r")
            lines = f2.readlines()
            for line3 in lines:
                sub_classes_list.append(line3.replace("\n",""))
                
        with torch.no_grad():
  

            for n in range(number_per_thread_num):
        
                # clear all features and attention maps
                clear_feature_dic()
                controller.reset()
            
                g_cpu = torch.Generator().manual_seed(seed)

                if len(sub_classes_list)!=0:
                    if random.random()>0.3:
                        prompts = [choice(sub_classes_list)]
                    else:
                        prompts = ["a photo of {}".format(class_target)]
                else:
                    prompts = ["a photo of {}".format(class_target)]
            
#                 prompts = ["a photo of a {}".format(class_target)]
                print("prompts:",prompts)
                trainclass = class_target


                start_code = None
                if opt.fixed_code:
                    print('start_code')
                    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                
                images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,scheduler, prompts, controller,  num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=g_cpu, low_resource=LOW_RESOURCE, Train=False)
                ptp_utils.save_images(images_here,out_put = "{}/{}_{}.jpg".format(Image_path,trainclass,seed))
                

                
                full_arr = np.zeros((21, 512,512), np.float32)
                full_arr[0]=0.5
                
                for idxx in classes:
                    if idxx==0:
                        continue

                    class_name = classes[idxx]
                    if class_name not in classes_check[idx]:
                        continue   
                        
                    # train segmentation
                    query_text = class_name
                    
                    text_input = tokenizer(
                    query_text,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                    text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]
                    c_split = tokenizer.tokenize(query_text)

#                     class_embedding=text_embeddings[:,5:len(c_split)+1,:]
                    class_embedding=text_embeddings
                    
                    if class_embedding.size()[1] > 1:
                        class_embedding = torch.unsqueeze(class_embedding.mean(1),1)

                    diffusion_features=get_feature_dic()
#                     class_target  class_name
                    outputs=seg_model(diffusion_features,controller,prompts,tokenizer,class_embedding)
                    
                    mask_cls_results = outputs["pred_logits"]
                    mask_pred_results = outputs["pred_masks"]
                    mask_pred_results = F.interpolate(
                                        mask_pred_results,
                                        size=(512, 512),
                                        mode="bilinear",
                                        align_corners=False,
                                        )
                    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                        
                        instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result,class_n = cfg.SEG_Decorder.num_classes,test_topk_per_image=3,query_n = cfg.SEG_Decorder.num_queries)
                            
                        pred_masks = instance_r.pred_masks.cpu().numpy().astype(np.uint8)
                        pred_boxes = instance_r.pred_boxes
                        scores = instance_r.scores 
                        pred_classes = instance_r.pred_classes 


                        import heapq
                        topk_idx = heapq.nlargest(1, range(len(scores)), scores.__getitem__)
                        mask_instance = (pred_masks[topk_idx[0]]>0.5 * 1).astype(np.uint8) 
                        full_arr[idxx] = np.array(mask_instance)

                full_arr = softmax(full_arr, axis=0)
                mask = np.argmax(full_arr, axis=0)
                cv2.imwrite("{}/{}_{}.png".format(Mask_path,trainclass,seed), mask)
#                 cv2.imwrite("{}/{}_{}_att.png".format(Mask_path,trainclass,seed), attention_maps_3211*255)
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
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--prompt_root",
        action='store_true',
        help="uses prompt",
        default="./dataset/Prompts_From_GPT/voc2012"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_each_class",
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

        p = mp.Process(target=sub_processor, args=(i, opt))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    
 

    


if __name__ == "__main__":
    main()
