# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation
CUDA_VISIBLE_DEVICES=2 python tools/train_depth_KITTI.py --save_name Train_200_images_t1_attention_transformer_KITTI_10layers --config ./config/KITTI/KITTI_Depth_50images.yaml --image_limitation 200 --dataset KITTI