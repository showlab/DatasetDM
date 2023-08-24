import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from PIL import Image
from tqdm import tqdm
# 0: 'background'	1: 'top'	2: 'outer'	3: 'skirt'
# 4: 'dress'	5: 'pants'	6: 'leggings'	7: 'headwear'
# 8: 'eyeglass'	9: 'neckwear'	10: 'belt'	11: 'footwear'
# 12: 'bag'	13: 'hair'	14: 'face'	15: 'skin'
# 16: 'ring'	17: 'wrist wearing'	18: 'socks'	19: 'gloves'
# 20: 'necklace'	21: 'rompers'	22: 'earrings'	23: 'tie'

class_list = ['background','top','outer','skirt','dress','pants','leggings','headwear','eyeglass','neckwear','belt','footwear','bag','hair','face','skin','ring','wrist wearing','socks','gloves','necklace','rompers','earrings','tie']                         
                         
# coco_category_to_id_v1 = {0:0, 139:1, 144:2, 174:3, 179:4, 205:5, 250:6, 180:7, 211:8, 220:9, 88:10, 212:11, 90:12, 215:13, 50:14, 151:15}
# hhh = [0, 139, 144, 174, 179, 205, 250, 180, 211, 220, 88, 212, 90, 215, 50, 151]

ADE20K_150_CATEGORIES = [
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
    {"color": [255, 6, 82], "id": 15, "isthing": 1, "name": "skin"},
    {"color": [200, 10, 22], "id": 16, "isthing": 1, "name": "ring"},
    {"color": [255, 200, 52], "id": 17, "isthing": 1, "name": "wrist wearing"},
    {"color": [155, 240, 82], "id": 18, "isthing": 1, "name": "socks"},
    {"color": [125, 6, 102], "id": 19, "isthing": 1, "name": "gloves"},
    {"color": [255, 222, 82], "id": 20, "isthing": 1, "name": "necklace"},
    {"color": [125, 60, 82], "id": 21, "isthing": 1, "name": "rompers"},
    {"color": [215, 29, 22], "id": 22, "isthing": 1, "name": "earrings"},
    {"color": [155, 110, 182], "id": 23, "isthing": 1, "name": "tie"},
    {"color": [255, 210, 13], "id": 255, "isthing": 1, "name": "test"},
    

  
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


# /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former    ./VOC_Multi_Attention_car_GPT3/train_image/
root_image = "./DataDiffusion/DeepFashion_MM_Train_105_images_t1/Image"
# root_image = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Diffusion_Sem/DiffMask/DiffMask_VOC/VOC_Multi_Attention_bird_sub_8_NoClipRetrieval/train_image"

# ground_truth
root_mask = "./DataDiffusion/DeepFashion_MM_Train_105_images_t1/Mask"
# root_mask_check = "./All_Class/gt_car/"
# root_mask = "./ground_truth/"  ./VOC_Multi_Attention_car_GPT3/ground_truth_final/
vis_list = os.listdir(root_image)
mask_lsit = os.listdir(root_mask)
# root_mask_check_list = os.listdir(root_mask_check)


# vis_list = [i for i in vis_list if i.replace(".jpg","_segm.png") in mask_lsit]
vis_list = [i for i in vis_list if i.replace("jpg","png") in mask_lsit]

    
ids = []
for image_pa in tqdm(vis_list[:100]):

    print(image_pa)

    mask_path = os.path.join(root_mask,image_pa.replace(".jpg",".png"))
    
    image_path = os.path.join(root_image,image_pa)
    
    if not os.path.isfile(mask_path):
        continue
        
    print(mask_path)
    mask = cv2.imread(mask_path)
    segm = Image.open(mask_path)
    mask = np.array(segm) # shape: [750, 1101]

        
#     BGR转RGB，方法2
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in np.unique(mask):
# #         mask[mask == i ] = coco_category_to_id_v1[i]
        if i not in ids:
            ids.append(i)

    print(np.unique(mask))


    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_sem_seg(
                        mask, alpha=0.6
                    )

    vis_output.save("./DataDiffusion/Vis/{}".format(image_pa))
print(ids)