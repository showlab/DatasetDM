
# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python tools/train_instance_LVIS.py --save_name Train_1_images_t1_attention_transformer_LVIS_10layers_NoClass --config ./config/LVIS/LVIS_Instance_Seg_10Images.yaml --image_limitation 1