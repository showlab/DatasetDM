# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation
CUDA_VISIBLE_DEVICES=4 python tools/train_depth_NYU.py --save_name Train_250_images_t1_attention_transformer_NYU_10layers_NOAug --config ./config/NYU/NYU_Depth.yaml --image_limitation 250