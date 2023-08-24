import os
import json
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import argparse
import json
from random import choice
import cv2
import numpy as np
import random
import os
from tqdm import tqdm

def splicing_1x2(root,out_root, image_list,n_image=100,start_id=0):
    print("-------------- start augmentation: 1x2 -------------")
    start_id = 0
    ha = 0

    # 20000
    for idx in tqdm(range(n_image)):
        image_1 = choice(image_list)
        mask_1 = image_1.replace("jpg","png")

        image_2 = choice(image_list)
        mask_2 = image_2.replace("jpg","png")
        
        if "jpg" not in image_1 or "jpg" not in image_2:
            continue
        
        if not os.path.exists("./DataDiffusion/{}/Mask/{}".format(root,mask_1)) or not os.path.exists("./DataDiffusion/{}/Mask/{}".format(root,mask_2)):
            continue
            
        img1 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_1))
        img2 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_2)) 

        mas1 = cv2.imread("./DataDiffusion/{}/Mask/{}".format(root,mask_1))
        mas2 = cv2.imread("./DataDiffusion/{}/Mask/{}".format(root,mask_2))
    
        if random.random()>0.5:
            heng = True
            image = np.concatenate([img1, img2], axis=1)
            mask = np.concatenate([mas1, mas2], axis=1)             

        else:
            heng = False
            image = np.concatenate((img1, img2)) 
            mask = np.concatenate((mas1, mas2))            
                
        
        cv2.imwrite("./DataDiffusion/{}/Image/splicing_{}.jpg".format(out_root,start_id),image)
        cv2.imwrite("./DataDiffusion/{}/Mask/splicing_{}.png".format(out_root,start_id),mask)
            
        start_id+=1

def splicing_NxN(root,out_root, image_list,n_image=100,start_id=0,size=2):
    print("-------------- start augmentation: {}x{} -------------".format(size,size))
#     start_id = 0
    ha = 0

    # 20000
    
    for idx in tqdm(range(n_image)):
        list_image = []
        list_mask = []
        for x in range(size):
            image_1 = choice(image_list)
            mask_1 = image_1.replace("jpg","png")

            img1 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_1))
            mas1 = cv2.imread("./DataDiffusion/{}/Mask/{}".format(root,mask_1))
            
            # add Horizontal information
            for y in range(size-1):
                image_2 = choice(image_list)
                mask_2 = image_2.replace("jpg","png")

                img2 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_2))
                mas2 = cv2.imread("./DataDiffusion/{}/Mask/{}".format(root,mask_2))
            
                
                # concate annotation
                img1 = np.concatenate([img1, img2], axis=1)
                mas1 = np.concatenate([mas1, mas2], axis=1)
            list_image.append(img1)
            list_mask.append(mas1)
            
        list_image_ha = list_image[0]
        list_mask_ha = list_mask[0]
        for i in range(1,size):

            list_image_ha = np.concatenate((list_image_ha, list_image[i])) 
            list_mask_ha = np.concatenate((list_mask_ha, list_mask[i])) 
            
        cv2.imwrite("./DataDiffusion/{}/Image/splicing_{}.jpg".format(out_root,start_id),list_image_ha)
        cv2.imwrite("./DataDiffusion/{}/Mask/splicing_{}.png".format(out_root,start_id),list_mask_ha)
            
        start_id+=1
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        nargs="?",
        default="./config/",
        help="config for training"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()
    
    os.makedirs("./DataDiffusion/{}/Image/".format(opt.out_root), exist_ok=True)
    os.makedirs("./DataDiffusion/{}/Mask/".format(opt.out_root), exist_ok=True)
    
    image_list = os.listdir("./DataDiffusion/{}/Image/".format(opt.input_root))
    
    import multiprocessing as mp
    import threading
    result_dict = mp.Manager().dict()
    mp = mp.get_context("spawn")
    processes = []
#     per_thread_video_num = int(len(coco_category_list)/thread_num)
#     thread_num=8
    print('Start Generation')
    for i in range(opt.thread_num):
        
        if i == 0:
            p = mp.Process(target=splicing_1x2, args=(opt.input_root,opt.out_root,image_list, 8000, 0))
            p.start()
            processes.append(p)
        elif i == 1:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,10000,15001,2))
            p.start()
            processes.append(p)
        elif i == 2:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,10000,35001,3))
            p.start()
            processes.append(p)
        elif i == 3:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,10000,55001,5))
            p.start()
            processes.append(p)
        elif i == 4:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,6000,65001,6))
            p.start()
            processes.append(p)
        elif i == 5:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,3000,75001,7))
            p.start()
            processes.append(p)
        elif i == 6:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,2000,85001,8))
            p.start()
            processes.append(p)
            
            
    for p in processes:
        p.join()

    result_dict = dict(result_dict)

if __name__ == "__main__":
    main()