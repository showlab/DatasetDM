# CUDA_VISIBLE_DEVICES=1 python train.py --save_name Train_5_Text_images_t1_attention
# CUDA_VISIBLE_DEVICES=3 python train_maskformer.py --save_name Train_100_images_t1_attention_transformer_Cityscapes


# CUDA_LAUNCH_BLOCKING=4 
# COCO/ Instance Segmentation
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 python tools/train_semantic_voc.py --save_name Train_30_images_t1_attention_transformer_VOC_10layers_ZeroShot --config ./config/VOC_Zero/VOC_Sematic_Seg_30Images.yaml --image_limitation 30