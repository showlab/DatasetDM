# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Zhejiang University.
# author: weijia wu
# ------------------------------------------------------------------------------


import os
import cv2
import random
from dataset.base_dataset_depth import BaseDataset
import json

class KITTI(BaseDataset):
    def __init__(self, data_path="./data/",
                 is_train=True, image_limitation =50, crop_size=(512, 512), scale_size=None,depth_scale = 80):
        super().__init__(crop_size)
        
        self.is_train = is_train
        self.size=512
        self.image_limitation = image_limitation
        self.data_root = os.path.join(data_path, 'kitti')
        self.depth_scale = depth_scale
        
        # join paths if data_root is specified  data/kitti/kitti_eigen_train.txt
        self.img_dir = os.path.join(self.data_root, "input")
        self.ann_dir = os.path.join(self.data_root, "gt_depth")
        self.split = os.path.join(self.data_root, "kitti_eigen_train.txt")
        
        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.ann_dir, self.split)
        random.shuffle(self.img_infos)
        self.img_infos = self.img_infos[:self.image_limitation]

        print("Dataset: KITTI")
        print("Training Sample:",len(self.img_infos))
        print([i['filename'] for i in self.img_infos])
    
    def load_annotations(self, img_dir, ann_dir, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """

        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    if ann_dir is not None: # benchmark test or unsupervised future
                        depth_map = line.strip().split(" ")[1]
                        if depth_map == 'None':
                            self.invalid_depth_num += 1
                            continue
                        img_info['ann'] = dict(depth_map=depth_map)
                    img_name = line.strip().split(" ")[0]
                    img_info['filename'] = img_name
                    img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered')
        
        return img_infos
    
    def __len__(self):
        return len(self.img_infos)
    
    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['img_prefix'] = self.img_dir
        results['depth_prefix'] = self.ann_dir
        results['depth_scale'] = self.depth_scale

        results['cam_intrinsic_dict'] = {
            '2011_09_26' : [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
                            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]],
            '2011_09_28' : [[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01], 
                            [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]],
            '2011_09_29' : [[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
                            [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]],
            '2011_09_30' : [[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01], 
                            [0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03]],
            '2011_10_03' : [[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01], 
                            [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01], 
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]],
        }
        
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return results
    
    def __getitem__(self, idx):
        
        results = self.prepare_train_img(idx)
#         print(results["ann_info"])
#         print(results["img_info"])
        
        img_path = results["img_info"]['filename']
        gt_path = results["img_info"]['ann']['depth_map']
        
        img_path = os.path.join(self.img_dir, img_path)
        gt_path = os.path.join(self.ann_dir, gt_path)
        print(img_path)
        print(gt_path)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        short_edge = min(image.shape[0], image.shape[1])
        if short_edge < self.size:
            # 保证短边 >= inputsize
            scale = self.size / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            depth = cv2.resize(depth, dsize=None, fx=scale, fy=scale)
        original_depth = depth/depth.max()*255
        original_image = image.copy()
        
        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)
        depth = depth / 256.0  # convert in meters
#         depth = depth / 1000.0  # convert in meters
#         print(depth.max())
        return {'image': image, 'depth': depth, 'filename': img_path, 'original_image':original_image, 'original_depth': original_depth, "prompt":"a photo of "}
