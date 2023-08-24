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
        txt_1 = image_1.replace("jpg","txt")

        image_2 = choice(image_list)
        txt_2 = image_2.replace("jpg","txt")
        
        if "jpg" not in image_1 or "jpg" not in image_2:
            continue
        
        if not os.path.exists("./DataDiffusion/{}/Mask/{}".format(root,txt_1)) or not os.path.exists("./DataDiffusion/{}/Mask/{}".format(root,txt_2)):
            continue
            
        img1 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_1))
        img2 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_2)) 
        
        if random.random()>0.5:
            heng = True
            image = np.concatenate([img1, img2], axis=1)
            
            data = []
            for line in open("./DataDiffusion/{}/Mask/{}".format(root,txt_1),"r"): #设置文件对象并读取每一行文件
                data.append(line)               #将每一行文件加入到list中 
            
            for line in open("./DataDiffusion/{}/Mask/{}".format(root,txt_2),"r"): #设置文件对象并读取每一行文件
                int_line = [int(i) for i in line.split(",")[:-2]]
                new_int_line = []
                for c,i in enumerate(int_line): 
                    if c%2==0:
                        new_int_line.append(str(i+512)) 
                    else:
                        new_int_line.append(str(i))
                
                
                str_cont = ""
                for i in new_int_line:
                    str_cont=str_cont+ str(i) + ","
                str_cont=str_cont + line.split(",")[-2] + "," + line.split(",")[-1]
                            
                data.append(str_cont)               #将每一行文件加入到list中 

        else:
            heng = False
            image = np.concatenate((img1, img2)) 
            
            data = []
            for line in open("./DataDiffusion/{}/Mask/{}".format(root,txt_1),"r"): #设置文件对象并读取每一行文件
                data.append(line)               #将每一行文件加入到list中 
            
            for line in open("./DataDiffusion/{}/Mask/{}".format(root,txt_2),"r"): #设置文件对象并读取每一行文件
                int_line = [int(i) for i in line.split(",")[:-2]]
                new_int_line = []
                for c,i in enumerate(int_line): 
                    if c%2!=0:
                        new_int_line.append(str(i+512)) 
                    else:
                        new_int_line.append(str(i))
                
                
                str_cont = ""
                for i in new_int_line:
                    str_cont=str_cont+ str(i) + ","
                str_cont=str_cont + line.split(",")[-2] + "," + line.split(",")[-1]
                            
                data.append(str_cont)               #将每一行文件加入到list中 
                
        
        cv2.imwrite("./DataDiffusion/{}/Image/splicing_{}.jpg".format(out_root,start_id),image)
        with open("./DataDiffusion/{}/Mask/splicing_{}.txt".format(out_root,start_id),'w') as f:    #设置文件对象
            f.writelines(data)
            
        start_id+=1

def splicing_NxN(root,out_root, image_list,n_image=100,start_id=0,size=2):
    print("-------------- start augmentation: {}x{} -------------".format(size,size))
#     start_id = 0
    ha = 0

    # 20000
    
    for idx in tqdm(range(n_image)):
        list_image = []
        ann_data = []
        for x in range(size):
            image_1 = choice(image_list)
            txt_1 = image_1.replace("jpg","txt")

            img1 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_1))
            # concate annotation
            for line in open("./DataDiffusion/{}/Mask/{}".format(root,txt_1),"r"): 
                int_line = [int(i) for i in line.split(",")[:-2]]
                new_int_line = []
                for c,i in enumerate(int_line): 
                    if c%2!=0:
                        new_int_line.append(str(i+512*x)) 
                    else:
                        new_int_line.append(str(i))
                
                
                str_cont = ""
                for i in new_int_line:
                    str_cont=str_cont+ str(i) + ","
                str_cont=str_cont + line.split(",")[-2] + "," + line.split(",")[-1]
                ann_data.append(str_cont) 
            
            # add Horizontal information
            for y in range(size-1):
                image_2 = choice(image_list)
                txt_2 = image_2.replace("jpg","txt")

                img2 = cv2.imread("./DataDiffusion/{}/Image/{}".format(root,image_2))
                img1 = np.concatenate([img1, img2], axis=1)
                
                # concate annotation
                for line in open("./DataDiffusion/{}/Mask/{}".format(root,txt_2),"r"): 
                    int_line = [int(i) for i in line.split(",")[:-2]]
                    new_int_line = []
                    for c,i in enumerate(int_line): 
                        if c%2!=0:
                            new_int_line.append(str(i+512*x)) 
                        else:
                            new_int_line.append(str(i+512*(y+1)))


                    str_cont = ""
                    for i in new_int_line:
                        str_cont=str_cont+ str(i) + ","
                    str_cont=str_cont + line.split(",")[-2] + "," + line.split(",")[-1]
                    ann_data.append(str_cont) 
            list_image.append(img1)
        list_image_ha = list_image[0]
        for i in range(1,size):
            list_image_ha = np.concatenate((list_image_ha, list_image[i])) 
            

        
        cv2.imwrite("./DataDiffusion/{}/Image/splicing_{}.jpg".format(out_root,start_id),list_image_ha)
        with open("./DataDiffusion/{}/Mask/splicing_{}.txt".format(out_root,start_id),'w') as f:    #设置文件对象
            f.writelines(ann_data)
            
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
            p = mp.Process(target=splicing_1x2, args=(opt.input_root,opt.out_root,image_list, 15000, 0))
            p.start()
            processes.append(p)
        elif i == 1:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,20000,15001,2))
            p.start()
            processes.append(p)
        elif i == 2:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,20000,35001,3))
            p.start()
            processes.append(p)
        elif i == 3:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,20000,55001,5))
            p.start()
            processes.append(p)
        elif i == 4:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,10000,65001,6))
            p.start()
            processes.append(p)
        elif i == 5:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,5000,75001,7))
            p.start()
            processes.append(p)
        elif i == 6:
            p = mp.Process(target=splicing_NxN, args=(opt.input_root,opt.out_root,image_list,3000,85001,8))
            p.start()
            processes.append(p)
            
            
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    
#     splicing_1x2(opt.input_root,opt.out_root,image_list,n_image=15000,start_id=0)
#     splicing_NxN(opt.input_root,opt.out_root,image_list,n_image=20000,start_id=15001,size=2)
#     splicing_NxN(opt.input_root,opt.out_root,image_list,n_image=20000,start_id=35001,size=3)
#     splicing_NxN(opt.input_root,opt.out_root,image_list,n_image=20000,start_id=55001,size=5)
#     splicing_NxN(opt.input_root,opt.out_root,image_list,n_image=10000,start_id=65001,size=6)
#     splicing_NxN(opt.input_root,opt.out_root,image_list,n_image=5000,start_id=75001,size=7)
#     splicing_NxN(opt.input_root,opt.out_root,image_list,n_image=3000,start_id=85001,size=8)
    
if __name__ == "__main__":
    main()