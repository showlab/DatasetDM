import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

coco_category_to_id_v1 = { 'aeroplane':0,
    'bicycle':1,
    'bird':2,
    'boat':3,
    'bottle':4,
    'bus':5,
    'car':6,
    'cat':7,
    'chair':8,
    'cow':9,
    'diningtable':10,
    'dog':11,
    'horse':12,
    'motorbike':13,
    'person':14,
    'pottedplant':15,
    'sheep':16,
    'sofa':17,
    'train':18,
    'tvmonitor':19}

ADE20K_150_CATEGORIES = [
    {"color": [250, 250, 250], "id": 0, "isthing": 0, "name": "background"},
    {"color": [194, 255, 0], "id": 1, "isthing": 0, "name": "aeroplane"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "bicycle"},
    {"color": [80, 150, 20], "id": 3, "isthing": 0, "name": "bird"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "boat"},
    {"color": [255, 5, 153], "id": 5, "isthing": 0, "name": "bottle"},
    {"color": [0, 255, 245], "id": 6, "isthing": 0, "name": "bus"},
    {"color": [204, 5, 255], "id": 7, "isthing": 1, "name": "car"},
    {"color": [255, 0, 102], "id": 8, "isthing": 1, "name": "cat "},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "chair"},
    {"color": [224, 5, 255], "id": 10, "isthing": 1, "name": "cow"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "diningtable"},
    {"color": [150, 5, 61], "id": 12, "isthing": 1, "name": "dog"},
    {"color": [0, 255, 112], "id": 13, "isthing": 0, "name": "horse"},
    {"color": [8, 255, 51], "id": 14, "isthing": 1, "name": "motorbike"},
    {"color": [255, 6, 82], "id": 15, "isthing": 1, "name": "person"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "pottedplant"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "sheep"},
    {"color": [255, 51, 7], "id": 18, "isthing": 1, "name": "sofa"},
    {"color": [204, 70, 3], "id": 19, "isthing": 1, "name": "train"},
    {"color": [0, 102, 200], "id": 20, "isthing": 1, "name": "tvmonitor"},
  
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


metadata = MetadataCatalog.get("sem_seg"
        )
# small text
# image_car_93258


# /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former    ./VOC_Multi_Attention_car_GPT3/train_image/
root_image = "./DataDiffusion/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/Image"


# ground_truth
# root_mask = "./DiffMask_VOC/VOC_Multi_Attention_bird_sub_30000_NoClipRetrieval/refine_gt_crf_0.6/"
root_mask = "./DataDiffusion/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/Mask"

vis_list = os.listdir(root_image)
mask_lsit = os.listdir(root_mask)


vis_list = [i for i in vis_list if i.replace("jpg","png") in mask_lsit]
# vis_list = [i for i in vis_list if i.replace("jpg","png") in root_mask_check_list]

    
for image_pa in vis_list[:300]:
    

    print(image_pa)

    mask_path = os.path.join(root_mask,image_pa.replace("jpg","png"))
    
    # mask_path = os.path.join(root_mask,image_pa.replace("jpg","png"))
    image_path = os.path.join(root_image,image_pa)
    
    if not os.path.isfile(mask_path):
        continue
        
    print(mask_path)
    mask = cv2.imread(mask_path)[:,:,0]
    image = cv2.imread(image_path)

    #BGR转RGB，方法2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    
    # mask = mask
    print(mask.shape)

    print(np.unique(mask))
#     mask[mask==19]=2
    h,w = image.shape[:2]

    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(
                        mask, alpha=0.5
                    )

    vis_output.save("./DataDiffusion/Vis/{}".format(image_pa))

