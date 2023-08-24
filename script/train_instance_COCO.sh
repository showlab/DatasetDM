# CUDA_VISIBLE_DEVICES=1 python train.py --save_name Train_5_Text_images_t1_attention
# CUDA_VISIBLE_DEVICES=3 python train_maskformer.py --save_name Train_100_images_t1_attention_transformer_Cityscapes

# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation
CUDA_LAUNCH_BLOCKING=4 CUDA_VISIBLE_DEVICES=0 python tools/train_instance_coco.py --save_name Train_50_images_t1_attention_transformer_COCO_10layers_NoClass --config ./config/coco/COCO_Instance_Seg_10Images.yaml --image_limitation 50