import matplotlib
import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from detectron2.structures import BitMasks, Instances
import mmcv
from PIL import Image
# color the depth, kitti magma_r, nyu jet
def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # TODO: remove hacks

    # for abs
    # vmin=1e-3
    # vmax=80

    # for relative
    # value[value<=vmin]=vmin

    # vmin=None
    # vmax=None

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :, :3] # bgr -> rgb
    rgb_value = value[..., ::-1]

    return rgb_value


# /mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former    ./VOC_Multi_Attention_car_GPT3/train_image/
root_image = "./DataDiffusion/KITTI_Train_50_images_t1/Image"
# root_image = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/datasets/coco/val2017"

# ground_truth
# root_mask = "./DiffMask_VOC/VOC_Multi_Attention_bird_sub_30000_NoClipRetrieval/refine_gt_crf_0.6/"
root_mask = "./DataDiffusion/KITTI_Train_50_images_t1/Annotation"
# root_mask = "/mmu-ocr/weijiawu/Code/Diffusion_Model/Mask2Former/demo/COCO2017_pred_ins"

vis_list = os.listdir(root_image)
mask_lsit = os.listdir(root_mask)
# root_mask_check_list = os.listdir(root_mask_check)

print(vis_list[0])
print(mask_lsit[0])
vis_list = [i for i in vis_list if i.replace("jpg","png") in mask_lsit]
print(len(vis_list))

for image_pa in vis_list[:20]:
    
    print(image_pa)

#     mask_path = os.path.join(root_mask,image_pa.replace("jpg","txt"))
    
    mask_path = os.path.join(root_mask,image_pa.replace("jpg","png"))
    image_path = os.path.join(root_image,image_pa)

    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('float32')
#     mask = np.asarray(Image.open(mask_path),
#                               dtype=np.float32) / 256.0
    
    image = cv2.imread(image_path)
    
    print(np.unique(mask))
#     assert False
    mask = np.expand_dims(mask[:,:], axis=0)

    print(mask.max())
    print(mask.shape)
    mask = colorize(mask,vmin=0,vmax=128)
#     mask = mask[0,:,:,:,0,0]
    print(mask.shape)

#     vmin = mask.min()
#     vmax = mask.max()

#     mask = mask/vmax*255
        
#     image = np.concatenate([image, mask], axis=1)
    mmcv.imwrite(mask.squeeze(), "./DataDiffusion/Vis/{}".format(image_pa))
    cv2.imwrite("./DataDiffusion/Vis/image_{}".format(image_pa),image)


