# Cityscapes Semantic Segmentation
CUDA_VISIBLE_DEVICES=3 python tools/train_semantic_cityscapes.py --save_name Train_10_images_t1_attention_transformer_Cityscapes_10layers --config ./config/cityscapes/Cityscapes_Semantic_Seg_50images.yaml --image_limitation 10