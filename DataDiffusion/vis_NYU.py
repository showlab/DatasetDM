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
def colorize(value, cmap='jet', vmin=None, vmax=None):

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


# ./VOC_Multi_Attention_car_GPT3/train_image/
root_image = "./DataDiffusion/NYU_Train_250_images_t1/Image"

# ground_truth
root_mask = "./DataDiffusion/NYU_Train_250_images_t1/Annotation"

vis_list = os.listdir(root_image)
mask_lsit = os.listdir(root_mask)
# root_mask_check_list = os.listdir(root_mask_check)

print(vis_list[0])
print(mask_lsit[0])
vis_list = [i for i in vis_list if i.replace("jpg","png") in mask_lsit]
print(len(vis_list))

for image_pa in vis_list[:100]:
    
    print(image_pa)

#     mask_path = os.path.join(root_mask,image_pa.replace("jpg","txt"))
    
    mask_path = os.path.join(root_mask,image_pa.replace("jpg","png"))
    image_path = os.path.join(root_image,image_pa)

#     image_path = "./data/nyudepthv2/sync/study_room_0005b/rgb_00007.jpg"
#     mask_path = "./data/nyudepthv2/sync/study_room_0005b/sync_depth_00007.png"
    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('float32') / 256 * 10
    print(np.unique(mask))
    
    image = cv2.imread(image_path)
    
    mask = np.expand_dims(mask[:,:], axis=0)
#     mask = np.expand_dims(mask, axis=0)
#     mask = np.repeat(mask, 3, 3)
    print(mask.max())
    print(mask.shape)
    mask = colorize(mask,vmin=0,vmax=10)
#     mask = mask[0,:,:,:,0,0]
    print(mask.shape)

#     vmin = mask.min()
#     vmax = mask.max()

#     mask = mask/vmax*255
        
#     image = np.concatenate([image, mask], axis=1)
    
    mmcv.imwrite(mask.squeeze(), "./DataDiffusion/Vis/{}".format(image_pa))
    cv2.imwrite("./DataDiffusion/Vis/image_{}".format(image_pa),image)


