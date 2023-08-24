# -*- coding: utf-8 -*-
import cv2
import os
import copy
import numpy as np
import math
# import Levenshtein
from cv2 import VideoWriter, VideoWriter_fourcc
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import shutil
import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from detectron2.data import MetadataCatalog

coco_category_to_id_v1 = {
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

labels2dict = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18}

ADE20K_150_CATEGORIES = [
    {"color": [128, 64, 128], "id": 0, "isthing": 0, "name": "road"},
    {"color": [244, 35, 232], "id": 1, "isthing": 0, "name": "sidewalk"},
    {"color": [70, 70, 70], "id": 2, "isthing": 0, "name": "building"},
    {"color": [102, 102, 156], "id": 3, "isthing": 0, "name": "wall"},
    {"color": [190, 153, 153], "id": 4, "isthing": 0, "name": "fence"},
    {"color": [153, 153, 153], "id": 5, "isthing": 0, "name": "pole"},
    {"color": [250, 170, 30], "id": 6, "isthing": 0, "name": "traffic light"},
    {"color": [220, 220, 0], "id": 7, "isthing": 1, "name": "traffic sign"},
    {"color": [107, 142, 35], "id": 8, "isthing": 1, "name": "vegetation "},
    {"color": [152, 251, 152], "id": 9, "isthing": 0, "name": "terrain"},
    {"color": [70, 130, 180], "id": 10, "isthing": 1, "name": "sky"},
    {"color": [220, 20, 60], "id": 11, "isthing": 0, "name": "person"},
    {"color": [255, 0, 0], "id": 12, "isthing": 1, "name": "rider"},
    {"color": [0, 0, 142], "id": 13, "isthing": 0, "name": "car"},
    {"color": [0, 0, 70], "id": 14, "isthing": 1, "name": "truck"},
    {"color": [0, 60, 100], "id": 15, "isthing": 1, "name": "bus"},
    {"color": [0, 80, 100], "id": 16, "isthing": 0, "name": "train"},
    {"color": [0, 0, 230], "id": 17, "isthing": 0, "name": "motorcycle"},
    {"color": [119, 11, 32], "id": 18, "isthing": 1, "name": "bicycle"}
  
]


def get_findContours(mask):
    mask_instance = (mask>0.5 * 1).astype(np.uint8) 
    ontours, hierarchy = cv2.findContours(mask_instance.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    
    min_area = 0
    polygon_ins = []
    x,y,w,h = 0,0,0,0
    
    image_h, image_w = mask.shape[0:2]
    gt_kernel = np.zeros((image_h,image_w), dtype='uint8')
    max_area = 0
    cont = []
    for cnt in ontours:
        # 外接矩形框，没有方向角
        x_ins_t, y_ins_t, w_ins_t, h_ins_t = cv2.boundingRect(cnt)

        if w_ins_t*h_ins_t>max_area:
            max_area = w_ins_t*h_ins_t
            cont = cnt
#             continue
#         cv2.fillPoly(gt_kernel, [cnt], 1)
    
    return cont



def mask_image(image, mask_2d, rgb=None, valid = False):
    h, w = mask_2d.shape

    # mask_3d = np.ones((h, w), dtype="uint8") * 255
    mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
    # mask_3d[mask_2d[:, :] == 1] = 0
    
#     print(image.shape)
#     image.astype("uint8")
    mask = (mask_2d!=0).astype(bool)
    if rgb is None:
        rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        
    mask_3d_color[mask_2d[:, :] == 1] = rgb
    image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5
    
    if valid:
        mask_3d_color[mask_2d[:, :] == 1] = [[0,0,0]]
        kernel = np.ones((5,5),np.uint8)  
        mask_2d = cv2.dilate(mask_2d,kernel,iterations = 4)
        mask = (mask_2d!=0).astype(bool)
        image[mask] = image[mask] * 0 + mask_3d_color[mask] * 1
        return image,rgb
        
    return image,rgb






def _get_ade20k_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in ADE20K_150_CATEGORIES]
#     assert len(stuff_ids) == 847, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"].split(",")[0] for k in ADE20K_150_CATEGORIES]
    
    color = [k["color"] for k in ADE20K_150_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "color":color
    }
    return ret


def register_all_ade20k_full(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    meta = _get_ade20k_full_meta()
    for name, dirname in [("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"sem_seg_citysscapes_demo"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors = meta["color"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg_citysscapes",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_full(_root)


