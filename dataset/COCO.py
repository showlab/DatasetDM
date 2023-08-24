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


def load_coco_json(json_file, image_root, dataset_name="coco_2017_train", extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        print("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                print(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)

    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        print(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    print("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    print("training image",len(dataset_dicts))
    if num_instances_without_valid_segmentation > 0:
        print(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def file2id(folder_path, file_path):
    # extract relative path starting from `folder_path`
    image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    # remove file extension
    image_id = os.path.splitext(image_id)[0]
    return image_id

class Instance_COCO(Dataset):
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
        root = os.path.join("./data/", "COCO2017")
    #     meta = _get_voc_full_meta()
        
        #         # 10 images for each class
#         image_limitation=5
        
        classes_fileter = {}
        for i in range(81):
            i = i
            classes_fileter[i]=0
            
        image_root = os.path.join(root, 'train2017')
        gt_root = os.path.join(root, 'annotations/instances_train2017.json')
        dataset_dict = load_coco_json(gt_root,image_root)
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
                if class_id!=255 and classes_fileter[class_id]<image_limitation and np.array(mask).sum()>10000:
#                     class_list = []
#                     for class_id_1,mask_1 in zip(classes,masks): 
#                         if np.array(mask_1).sum()>5000:
#                             class_list.append(class_id_1)
                    self.dataset_dict.append(dataset_dict[idx])
                    classes_fileter[class_id]+=1
                    train_file_name.append(dataset_one["file_name"])
                    break
            if len(self.dataset_dict)>=image_limitation*(80):
                break
                
        print("train sample after filtering: ",len(self.dataset_dict))           
        print("train sample list: ",train_file_name)
        
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
        
#         mask = cv2.resize(mask, (self.size, self.size))
#         example = {}
#         example["image"] = torch.from_numpy(image).permute(2, 0, 1)
#         example["mask"] = torch.from_numpy(mask).long()
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