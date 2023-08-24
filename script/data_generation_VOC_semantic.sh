
# CUDA_VISIBLE_DEVICES=3 
# Semantic Segmentation VOC
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tools/parallel_generate_Semantic_VOC_AnyClass.py --sd_ckpt './models/ldm/stable-diffusion-v1/stable_diffusion.ckpt' --grounding_ckpt './checkpoint/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/latest_checkpoint.pth' --n_each_class 1000 --outdir './DataDiffusion/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/' --thread_num 7 --H 512 --W 512 --config ./config/VOC/VOC_Sematic_Seg_5Images.yaml

