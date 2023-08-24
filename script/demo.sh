# CUDA_VISIBLE_DEVICES=3 
# demo  "Two cars are parked on the urban street"  A full-body shot of 
#A full-body shot of
CUDA_VISIBLE_DEVICES=2 python tools/parallel_demo.py --sd_ckpt './models/ldm/stable-diffusion-v1/stable_diffusion.ckpt' --grounding_ckpt_semantic './checkpoint/Train_1_images_t1_attention_transformer_Cityscapes_10layers/latest_checkpoint.pth' --grounding_ckpt_depth_Virtual_KITTI_2 './checkpoint/Train_t1_attention_transformer_Virtual_KITTI_2_10layers/latest_checkpoint.pth' --grounding_ckpt_depth_NYU './checkpoint/Train_Depth_NYU_20000_Demo/latest_checkpoint.pth' --grounding_ckpt_instance './checkpoint/Train_50_images_t1_attention_transformer_COCO_10layers_NoClass/latest_checkpoint.pth' --grounding_ckpt_semantic_open './checkpoint/Train_5_images_t1_attention_transformer_VOC_10layers_ablation_multifeature/latest_checkpoint.pth' --grounding_ckpt_human_pose './checkpoint/Train_800_images_t1_attention_transformer_Pose_COCO_HeatMap_10layers/latest_checkpoint.pth' --grounding_ckpt_human_seg './checkpoint/Train_50_images_t1_attention_transformer_DeepFashionMM_10layers/latest_checkpoint.pth' --number_data 30 --outdir './DataDiffusion/Demo/' --thread_num 1 --H 512 --W 512 --config_semantic './config/cityscapes/Cityscapes_Semantic_Seg_50images.yaml' --config_depth_Virtual_KITTI_2 './config/KITTI/KITTI_Depth_50images.yaml' --config_depth_NYU './config/NYU/NYU_Depth.yaml' --config_instance './config/coco/COCO_Instance_Seg_10Images.yaml' --config_open_word_seg './config/VOC_Zero/VOC_Sematic_Seg_30Images.yaml' --config_instance './config/coco/COCO_Instance_Seg_10Images.yaml' --config_pose './config/coco_pose/coco_pose.yaml' --config_human_seg './config/deepfashion/Deepfashion_Semantic_Seg_5mages.yaml' --prompt "A man is riding a motorcycle on a deserted road." --open_word "motorcycle"

# A man in a sharp grey suit, crisp white shirt, and black polished shoes strides confidently down a city street
# A man in a sharp grey suit, crisp white shirt, and black polished shoes strides confidently down a city street   