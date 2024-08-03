import cv2
import glob
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from patch import init_usap_patch
import os

class CarlaDatasetSAC(Dataset):
    def __init__(self, attack_method, dataset_path='dataset/train') -> None:
        self.attack_method = attack_method
        self.dataset_path = dataset_path
        self.dataset = glob.glob(f'{dataset_path}/*/*.png')
        

    def __getitem__(self, index):
        image_path = self.dataset[index]
        image = cv2.imread(image_path)[:,:,::-1].copy()
        image = torch.tensor(image, dtype=torch.float32)
        if self.attack_method == 'EOT' or self.attack_method == 'SIB':
            patch_path, _ = os.path.split(image_path)
            patch_path = os.path.join(patch_path, self.attack_method+'.png').replace('train', 'patch_train')
            patch = cv2.imread(patch_path)[:,:,::-1].copy()
            patch = torch.tensor(patch, dtype=torch.float32)
        elif self.attack_method == 'usap':
            patch = init_usap_patch()
            patch = patch*255
        with open(image_path.replace('.png','.json'), 'r') as file:
            annotations = json.load(file)
            rpoints = np.array(annotations["rpoints"], dtype=float)
            rpoints = torch.tensor(rpoints, dtype=float)

        return image, patch, rpoints

    def __len__(self):
        return len(self.dataset)
