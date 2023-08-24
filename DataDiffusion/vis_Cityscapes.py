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
        name = f"sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors = meta["color"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_full(_root)


metadata = MetadataCatalog.get("sem_seg")
# small text
# image_car_93258


root_image = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/cityscapes/DataDiffusion/Image"


# ground_truth
root_mask = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/cityscapes/DataDiffusion/Mask"

vis_list = os.listdir(root_image)
mask_lsit = os.listdir(root_mask)
# root_mask_check_list = os.listdir(root_mask_check)


vis_list = [i for i in vis_list if i.replace("_leftImg8bit.png","_gtFine_labelIds.png").replace("jpg","png") in mask_lsit]

    
    
for image_pa in vis_list[:100]:

    mask_path = os.path.join(root_mask,image_pa.replace("_leftImg8bit.png","_gtFine_labelIds.png").replace("jpg","png"))
    image_path = os.path.join(root_image,image_pa)
    
    
    mask_path = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/demo/cityscapes_pred/1.png"
    image_path = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/demo/demo_datadiffusion/1.png"
#     print(image_path)
    if not os.path.isfile(mask_path):
        continue
        

    print(mask_path)
    mask = cv2.imread(mask_path)[:,:,0]
#     mask[mask==2] = 8
#     mask[mask==1] = 8
    
    print(np.unique(mask))
    
    
#     background_ = mask.copy()
#     background_ = background_*0
#     id_list = np.unique(mask)
#     for i in id_list:
#         if i in labels2dict:
#             background_[mask==i]=labels2dict[i]
#         else:
#             background_[mask==i]=0
        
#     mask = background_
    
    
#     mask = mask*(coco_category_to_id_v1[image_pa.split("_")[0]]+1)
    image = cv2.imread(image_path)

        
    #BGR转RGB，方法2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    
    # mask = mask
    print(mask.shape)
#     if 20 not in np.unique(mask):
#         continue
    
#     mask[mask==19]=2
    h,w = image.shape[:2]

#     try:
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(
                        mask, alpha=0.5
                    )

    vis_output.save("./DataDiffusion/Vis/{}".format(image_pa))
    break

