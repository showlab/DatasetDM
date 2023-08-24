import os
import random
import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from packaging import version
import PIL
import matplotlib.pyplot as plt
import pickle
import json
import tqdm
from random import choice
import cv2
from dataset.augment import DataAugment
data_aug = DataAugment()
from detectron2.structures import BitMasks, Instances

# from detectron2.utils.file_io import PathManager
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

# ,"a photograph of a {}"

prompt_templates = [
    "a photo of a {} the urban street"
]



def file2id(folder_path, file_path):
    # extract relative path starting from `folder_path`
    image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    # remove file extension
    image_id = os.path.splitext(image_id)[0]
    return image_id

class Semantic_Cityscapes(Dataset):
    def __init__(
            self,
            text_encoder=None,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            image_limitation = 5,
            center_crop=False,
            keep_class=None,
            initialization=True,
            
    ):
#         self.root = cfg.data.root
#         self.tokenizer = tokenizer
#         self.text_encoder = text_encoder
#         self.learnable_property = cfg.data.learnable_property
        self.size = 512
#         self.center_crop = center_crop
#         self.flip_p = flip_p
        self.set = set
        self.scale = np.array([0.5, 0.8, 1.0, 1.3, 1.8])
        root = os.path.join("./data/", "cityscapes")
    #     meta = _get_voc_full_meta()

        image_root = os.path.join(root, 'leftImg8bit/train')
        gt_root = os.path.join(root, 'gtFine/train')
        
        input_files = []
        gt_files = []
        for cls in os.listdir(image_root):
            image_root_cls = os.path.join(image_root,cls)
            for image in os.listdir(image_root_cls):
                if "png" in image:
                    input_files.append(os.path.join(image_root_cls,image))
    

        gt_files = [i.replace("_leftImg8bit","_gtFine_labelTrainIds").replace("leftImg8bit","gtFine") for i in input_files] 
    
        
        
        self.input_files=[]
        self.gt_files = []
        
#         # 10 images for each class
#         image_limitation=5
        print("image_limitation:",image_limitation)
        classes_fileter = {}
        for i in range(20):
            i = i
            classes_fileter[i]=0
            
        for idx, (img_p,gt_p) in enumerate(zip(input_files,gt_files)):
            
            if len(self.input_files)>=image_limitation*(19):
                break
            mask = Image.open(gt_p)
            mask = np.array(mask).astype(np.uint8)
            if len(mask.shape) == 3:
                mask = mask[:,:,0]
            class_list = np.unique(mask)
            for class_id in class_list:
                if class_id!=255 and classes_fileter[class_id]<image_limitation and np.sum(mask==class_id)>10000:
                    self.input_files.append(img_p)
                    self.gt_files.append(gt_p)
                    classes_fileter[class_id]+=1
                    break
                    
        self._length = len(self.input_files)
        self.ignore_label=255
        self.classes = {
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
        print("selected training sample:",len(self.input_files)) 
        print(self.input_files)
        

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        image_path = self.input_files[i]
        gt_path = self.gt_files[i]
        
        image = Image.open(image_path).convert('RGB')
        
        mask = Image.open(gt_path).convert('RGB')
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        
        class_list = np.unique(mask)

#         class_list = [c for c in class_list if c!=0 and c!=255 and c<21]
        
#         select_class = int(choice(class_list))
#         if select_class!=1:
#             mask[mask==1] = 0
#         mask[mask==select_class] = 1
#         mask[mask!=1] = 0
        
#         prompt = prompt_templates[0].format(classes[select_class])
    
    
        #data augmentation
        image, mask = data_aug.random_scale(image, mask, self.scale)
        short_edge = min(image.shape[0], image.shape[1])
        if short_edge < self.size:
            # 保证短边 >= inputsize
            scale = self.size / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            mask = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
        image, mask = data_aug.random_crop_author([image, mask], (self.size, self.size))
        
        
        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        sem_seg_gt = mask
        instances = Instances(image_shape)
        classes = np.unique(sem_seg_gt)
        # remove ignored region
        classes = classes[classes != self.ignore_label]
        classes = classes[classes < 19]
        
        mapper_classes = classes.tolist()
        
        instances = {}
        instances["gt_classes"] = torch.tensor(mapper_classes, dtype=torch.int64)
#         instances.gt_classes = torch.tensor(mapper_classes, dtype=torch.int64)
        masks = []
        prompt_class = ""
        for class_id in classes:
            prompt_class = prompt_class + self.classes[class_id] + ","
            masks.append(sem_seg_gt == class_id)
        print(masks[0].shape) 
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances["gt_masks"] = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances["gt_masks"] = masks.tensor
        dataset_dict = {}
        
        dataset_dict["instances"] = instances

        dataset_dict["classes_str"] = [self.classes[el] for el in classes]
    
        
#         image = image.resize((self.size, self.size), resample=PIL.Image.LINEAR)
        image = np.array(image).astype(np.uint8)
        original_image = image
        image = (image / 127.5 - 1.0).astype(np.float32)
        
#         mask = cv2.resize(mask, (self.size, self.size))
        
#         example = {}
#         example["image"] = torch.from_numpy(image).permute(2, 0, 1)
#         example["mask"] = torch.from_numpy(mask).long()
        dataset_dict["prompt"] = prompt_templates[0].format(prompt_class)
#         example["class"] = classes[select_class]
        dataset_dict["original_image"] = original_image
        
        dataset_dict["image"] = torch.from_numpy(image).permute(2, 0, 1)
        return dataset_dict