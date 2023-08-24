import os
import cv2
import random
from dataset.base_dataset_depth import BaseDataset
import json
import numpy as np

class nyudepthv2(BaseDataset):
    def __init__(self, data_path="./data/",
                 is_train=True, image_limitation =50, crop_size=(512, 512), scale_size=None):
        super().__init__(crop_size)
        
        self.scale = np.array([0.5, 0.8, 1.0, 1.3, 1.8, 2.0, 2.5])
        
        self.size=512
        self.image_limitation = image_limitation
        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'nyudepthv2')

        self.image_path_list = []
        self.depth_path_list = []
        
        json_path = os.path.join(self.data_path,'nyu_class_list.json')
        with open(json_path, 'r') as f:
            self.class_list = json.load(f)

        txt_path = self.data_path
        if is_train:
            txt_path += '/train_list.txt'
            self.data_path = self.data_path + '/sync'
        else:
            txt_path += '/test_list.txt'
            self.data_path = self.data_path + '/official_splits/test/'
 
        self.filenames_list = self.readTXT(txt_path) # debug
        random.shuffle(self.filenames_list)
        self.filenames_list = self.filenames_list[:self.image_limitation]
        
        phase = 'train' if is_train else 'test'
        print("Dataset: NYU Depth V2")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))
        print([i.split(' ')[0] for i in self.filenames_list])
        
    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0]
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[1]
        filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]

        class_id = -1
        for i, name in enumerate(self.class_list):
            if name in filename:
                class_id = i
                break
        
        assert class_id >= 0

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        
        # random_scale
        rd_scale = float(np.random.choice(self.scale))
        image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale)
        depth = cv2.resize(depth, dsize=None, fx=rd_scale, fy=rd_scale)
        
        short_edge = min(image.shape[0], image.shape[1])
        if short_edge < self.size:
            # short side >= inputsize
            scale = self.size / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            depth = cv2.resize(depth, dsize=None, fx=scale, fy=scale)
        original_image = image.copy()
        original_depth = depth.copy()

        # print(image.shape, depth.shape, self.scale_size)

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 1000.0  # convert in meters

        return {'image': image, 'depth': depth, 'filename': filename, 'class_id': class_id, 'original_image':original_image,'original_depth':original_depth,
                "prompt":"a photo of "}
