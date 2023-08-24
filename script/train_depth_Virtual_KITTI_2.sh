# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation
CUDA_VISIBLE_DEVICES=5 python tools/train_depth_KITTI.py --save_name Train_t1_attention_transformer_Virtual_KITTI_2_10layers --config ./config/KITTI/KITTI_Depth_50images.yaml --image_limitation 200 --dataset VirtualKITTI