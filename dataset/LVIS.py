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
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import io
import contextlib
import pycocotools.mask as mask_util
import tqdm
import copy
from random import choice
from fvcore.common.timer import Timer
import cv2
from .augment import DataAugment
data_aug = DataAugment()
from detectron2.structures import BitMasks, Instances
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes,polygons_to_bitmask
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from tqdm import tqdm





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
    "a photo of {}"
]

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


def load_lvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        meta = get_lvis_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    print("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file))

    if extra_annotation_keys:
        print(
            "The following extra annotation keys will be loaded: {} ".format(extra_annotation_keys)
        )
    else:
        extra_annotation_keys = []

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root +"/"+ split_folder, file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
            if dataset_name is not None and "thing_dataset_id_to_contiguous_id" in meta:
                obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][anno["category_id"]]
            else:
                obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            segm = anno["segmentation"]  # list[list[float]]
            # filter out invalid polygons (< 3 points)
            valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            assert len(segm) == len(
                valid_segm
            ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            obj["segmentation"] = segm
            for extra_ann_key in extra_annotation_keys:
                obj[extra_ann_key] = anno[extra_ann_key]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def file2id(folder_path, file_path):
    # extract relative path starting from `folder_path`
    image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    # remove file extension
    image_id = os.path.splitext(image_id)[0]
    return image_id

class Instance_LVIS(Dataset):
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
        self.size = 512
#         self.center_crop = center_crop
#         self.flip_p = flip_p
        self.set = set
        self.scale = np.array([0.5, 0.8, 1.0, 1.3, 1.8])
        root = os.path.join("./data/", "lvis")
    #     meta = _get_voc_full_meta()
        
        #         # 10 images for each class
#         image_limitation=5
        
        classes_fileter = {}
        for i in range(1203):
            i = i
            classes_fileter[i]=0
            
        image_root = os.path.join(root, 'train2017')
        gt_root = os.path.join(root, 'lvis_v1_train.json')
        dataset_dict = load_lvis_json(gt_root,root)
        print("train sample before filtering: ",len(dataset_dict))
        dataset_dict = dataset_dict
        
        train_file_name = []
        self.dataset_dict = []
        for idx, dataset_one in tqdm(enumerate(dataset_dict)):
            dataset_one = copy.deepcopy(dataset_dict[idx])
            
            image = utils.read_image(dataset_one["file_name"])

            tfm_gens = []
            aug_input = T.AugInput(image)
            aug_input, transforms = T.apply_transform_gens(tfm_gens, aug_input)
            image = aug_input.image


            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_one["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            classes = [int(obj["category_id"]) for obj in annos]
            
            segms = [obj["segmentation"] for obj in annos]
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
                
            for class_id,mask in zip(classes,masks):
                if classes_fileter[class_id]<image_limitation and np.array(mask).sum()>50:
#                     class_list = []
#                     for class_id_1,mask_1 in zip(classes,masks): 
#                         if np.array(mask_1).sum()>5000:
#                             class_list.append(class_id_1)
                    self.dataset_dict.append(dataset_dict[idx])
                    classes_fileter[class_id]+=1
                    train_file_name.append(dataset_one["file_name"])
                    break
            if len(self.dataset_dict)>=image_limitation*(1201):
                break
                
        print("train sample after filtering: ",len(self.dataset_dict))           
        print("train sample list: ",train_file_name)
        
        
        # 打开文本文件以写入模式
        with open('./dataset/Selected_File/LVIS/1202.txt', 'w') as txt_file:
            for item in train_file_name:
                txt_file.write(item + '\n')
        
        
#         assert False
        self.max_length = 15
        self.aug = self.augmentation()
        self._length = len(self.dataset_dict)
        self.ignore_label=255
        
        self.classes = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',69: 'microwave',70: 'oven',71: 'toaster',72: 'sink',73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush'

    }


    def __len__(self):
        return self._length
    
    def augmentation(self):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                [532,832],
                2048,
                "range",
            )
        ]

        augs.append(
            T.RandomCrop(
                'absolute',
                (512,512),
            )
        )
#         augs.append(ColorAugSSDTransform(img_format="RGB"))
        augs.append(T.RandomFlip())

#         ret = {
#             "is_train": is_train,
#             "augmentations": augs,
#             "image_format": cfg.INPUT.FORMAT,
#             "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
#         }
        return augs
    
    def __getitem__(self, i):
#         class_ids
        dataset_one = copy.deepcopy(self.dataset_dict[i])
        
        image = utils.read_image(dataset_one["file_name"])

        tfm_gens = self.aug
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(tfm_gens, aug_input)
        image = aug_input.image


        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_one["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        classes = [int(obj["category_id"]) for obj in annos]
        segms = [obj["segmentation"] for obj in annos]
        
#         for c in range(5):
#             class_id = choice(class_ids)
        f_classes = []
        masks = []
        for idx,(classe, segm) in enumerate(zip(classes,segms)):
            poly_mask = polygons_to_bitmask(segm, *image.shape[:2])
            if poly_mask.sum()<500:
                continue
#                 if classe!=class_id:
#                     continue
            f_classes.append(1)
            masks.append(poly_mask)
#             if len(f_classes)>0:
#                 break
        
#         classes = f_classes[:self.max_length]
#         masks =   masks[:self.max_length]    
        classes = f_classes
        masks =   masks
        print("instance number:",len(classes))
        
        
        filter_classes = []
        for i in classes:
            if i not in filter_classes:
                filter_classes.append(i)
        
        prompt_class = ""
        for class_id in filter_classes:
            prompt_class = prompt_class + self.classes[class_id+1] + ","
              
        image_shape = (self.size, self.size)  # h, w
#         instances = Instances(image_shape)
        instances = {}
#         mapper_classes = [1 for i in classes]
        
        classes = [1 for i in classes]
        print(classes)
        print([self.classes[i+1] for i in classes])
        
        instances["gt_classes"] = torch.tensor(classes, dtype=torch.int64)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances["gt_masks"] = torch.zeros((0, self.size,self.size))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances["gt_masks"] = masks.tensor
        dataset_dict = {}
        
        dataset_dict["instances"] = instances
        dataset_dict["file_name"] = dataset_one["file_name"]
        dataset_dict["classes_str"] = [self.classes[el+1] for el in classes]
    
        image = np.array(image).astype(np.uint8)
        original_image = image.copy()
        image = (image / 127.5 - 1.0).astype(np.float32)
        

        dataset_dict["prompt"] = prompt_templates[0].format(prompt_class)
#         dataset_dict["class_name"] = self.classes[class_id+1]
#         example["class"] = classes[select_class]
        dataset_dict["original_image"] = original_image
        try:
            dataset_dict["image"] = torch.from_numpy(image).permute(2, 0, 1)
        except:
            print(dataset_one["file_name"])
            return self.__getitem__(i+1)
        return dataset_dict
    
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

if __name__ == "__main__":
    dataset = Instance_COCO(
            set="train",image_limitation = 5
        )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    for step, batch in enumerate(dataloader):

        image = batch["image"]
        instances = batch["instances"][0]["gt_masks"]
        prompts = batch["prompt"]
        class_name = batch["class_name"]
        original_image = batch["original_image"]
        
        
        original_image = original_image[0]
        for idx,(c) in enumerate(instances):

           
            original_image,_ = mask_image(original_image,c)
        cv2.imwrite("./dataset/debug/{}.jpg".format(step),original_image)