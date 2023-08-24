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
    "a photo of a {}"
]




def file2id(folder_path, file_path):
    # extract relative path starting from `folder_path`
    image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    # remove file extension
    image_id = os.path.splitext(image_id)[0]
    return image_id

class Semantic_VOC(Dataset):
    def __init__(
            self,
            text_encoder=None,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            image_limitation = 5,
            is_zero = False,
            is_long_tail = False,
            center_crop=False,
            keep_class=None,
            initialization=True,
            
    ):
        self.is_zero = is_zero
        self.is_long_tail = is_long_tail
        self.size = 512

        self.set = set
        self.scale = np.array([0.5, 0.8, 1.0, 1.3, 1.8, 2.0, 3.0])
        root = os.path.join("./data/", "PascalVOC12")
    #     meta = _get_voc_full_meta()

        image_root = os.path.join(root, 'JPEGImages')
        gt_root = os.path.join(root, 'SegmentationClassAug')
        
        input_files = sorted(
        (os.path.join(image_root, f) for f in os.listdir(image_root) if "jpg" in f),
        key=lambda file_path: file2id(image_root, file_path),
         )
        gt_files = sorted(
            (os.path.join(gt_root, f) for f in os.listdir(gt_root) if "png" in f),
            key=lambda file_path: file2id(gt_root, file_path),
        )
    
        
        txt = 'data/PascalVOC12/splits/train_aug.txt'
        with open(txt,'r') as fr:
            training_lst = fr.readlines()
        training_images_lst = [os.path.basename(el.split(' ')[0]) for el in training_lst]
        training_gts_lst = [os.path.basename(el.split(' ')[1].strip()) for el in training_lst]
        input_files = [el for el in input_files if os.path.basename(el) in training_images_lst]
        gt_files = [el for el in gt_files if os.path.basename(el) in training_gts_lst]
        
        self.input_files=[]
        self.gt_files = []

#         # 10 images for each class
#         image_limitation=image_limitation = 5,
        
        classes_fileter = {}
        for i in range(20):
            i = i +1
            classes_fileter[i]=0
            
        for img_p,gt_p in zip(input_files,gt_files):
            mask = Image.open(gt_p)
            mask = np.array(mask).astype(np.uint8)
            if len(mask.shape) == 3:
                mask = mask[:,:,0]
            class_list = np.unique(mask)
            for class_id in class_list:
                if self.is_zero:
                    if class_id!=0 and class_id!=255 and classes_fileter[class_id]<image_limitation and np.sum(mask==class_id)>18000 and class_id<=15:
                        self.input_files.append(img_p)
                        self.gt_files.append(gt_p)
                        classes_fileter[class_id]+=1
                        break
                elif self.is_long_tail:
                    if class_id<11:
                        image_limitation = 20
                    else:
                        image_limitation = 2
                    if class_id!=0 and class_id!=255 and classes_fileter[class_id]<image_limitation and np.sum(mask==class_id)>18000:
                        self.input_files.append(img_p)
                        self.gt_files.append(gt_p)
                        classes_fileter[class_id]+=1
                        break    
                else:
                    if class_id!=0 and class_id!=255 and classes_fileter[class_id]<image_limitation and np.sum(mask==class_id)>18000 and len(class_list)<3:
                        self.input_files.append(img_p)
                        self.gt_files.append(gt_p)
                        classes_fileter[class_id]+=1
                        break
        if self.is_zero:
            print("current is zero shot setting")
        print("selected training sample:",len(self.input_files)) 
        print(self.input_files)
        self._length = len(self.input_files)
        self.classes_zero_shot = {
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
        15: 'person'
    }
        self.classes = {
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
        
        if self.is_zero:
            class_list = [c for c in class_list if c!=0 and c!=255 and c<=15]
        else:
            class_list = [c for c in class_list if c!=0 and c!=255 and c<21]
            
        select_class = int(choice(class_list))
        if select_class!=1:
            mask[mask==1] = 0
        mask[mask==select_class] = 1
        mask[mask!=1] = 0
        
        if self.is_zero:
            prompt = prompt_templates[0].format(self.classes_zero_shot[select_class])
        else:
            prompt = prompt_templates[0].format(self.classes[select_class])
            
        #data augmentation
        image, mask = data_aug.random_scale(image, mask, self.scale)
        short_edge = min(image.shape[0], image.shape[1])
        if short_edge < self.size:
            scale = self.size / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            mask = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
        image, mask = data_aug.random_crop_author([image, mask], (self.size, self.size))
        mask = np.array(mask)
        
        instances = {}
        mapper_classes = [1]
        instances["gt_classes"] = torch.tensor(mapper_classes, dtype=torch.int64)
        masks = []
        masks.append(mask == 1)
            
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
        dataset_dict["classes_str"] = [self.classes[el] for el in mapper_classes]
        
        image = np.array(image).astype(np.uint8)
        original_image = image
        image = (image / 127.5 - 1.0).astype(np.float32)
        
        dataset_dict["prompt"] = prompt
        dataset_dict["original_image"] = original_image
        dataset_dict["image"] = torch.from_numpy(image).permute(2, 0, 1)

        return dataset_dict