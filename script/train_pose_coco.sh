# CUDA_LAUNCH_BLOCKING=4 
# COCO/ pose estimation

CUDA_VISIBLE_DEVICES=7 python tools/train_pose_coco_heatmap.py --save_name Train_800_images_t1_attention_transformer_Pose_COCO_HeatMap_10layers --config './config/coco_pose/coco_pose.yaml' --image_limitation 800 --dataset coco_pose
