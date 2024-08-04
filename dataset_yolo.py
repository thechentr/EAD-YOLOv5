import os
import cv2
import numpy as np
import torch
import json
import random
from patch import init_usap_patch

def yolo_collate_fn(batch_list):
    return (batch_list[0][i] for i in range(len(batch_list[0])))

class YOLODataset():
    def __init__(self, split, batch_size) -> None:
        if split == 'train':
            self.dataset = [os.path.join(f'dataset/{split}', str(car_idx).zfill(2), str(view_idx).zfill(3)+'.png') for car_idx in range(70000, 83600, 17*10) for view_idx in range(150)]
        elif split == 'test':
            self.dataset = [os.path.join(f'dataset/{split}', str(car_idx).zfill(2), str(view_idx).zfill(3)+'.png') for car_idx in range(10000, 13400, 17) for view_idx in range(150)]
        else:
            raise NotImplementedError
        self.batch_size = batch_size
    
    def shuffle(self):
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        images = torch.zeros((self.batch_size, 256, 256, 3))
        labels = torch.zeros((self.batch_size, 6))
        rpoints = torch.zeros((self.batch_size, 4, 2))
        for i in range(self.batch_size):
            image_path = self.dataset[index*self.batch_size+i]
            image = cv2.imread(image_path)[:,:,::-1].copy()
            images[i] = torch.tensor(image, dtype=float)
            with open(image_path.replace('.png','.json'), 'r') as file:
                annotations = json.load(file)
                label = np.array(([i, 0]+annotations["bbox"]), dtype=float)
                rpoint = np.array(annotations["rpoints"], dtype=float)
                labels[i] = torch.tensor(label, dtype=float)
                rpoints[i] = torch.tensor(rpoint, dtype=float)
        return images, labels, rpoints
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

class AdvYOLODataset():
    def __init__(self, batch_size, attack_method) -> None:

        self.dataset = [os.path.join(f'dataset/test', str(car_idx).zfill(2), '050.png') for car_idx in range(40)]
        self.batch_size = batch_size
        self.attack_method = attack_method
    
    def shuffle(self):
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        images = torch.zeros((self.batch_size, 256, 256, 3))
        patches = torch.zeros((self.batch_size, 64, 111, 3))
        labels = torch.zeros((self.batch_size, 6))
        rpoints = torch.zeros((self.batch_size, 4, 2))
        for i in range(self.batch_size):
            image_path = self.dataset[index*self.batch_size+i]
            patch_path, _ = os.path.split(image_path)
            patch_path = os.path.join(patch_path, self.attack_method+'.png').replace('test','patch_train')
            image = cv2.imread(image_path)[:,:,::-1].copy()
            images[i] = torch.tensor(image, dtype=float)
            if self.attack_method == 'DIM' or self.attack_method == 'EOT' or self.attack_method == 'SIB' or self.attack_method == 'CAMOU':
                patch = cv2.imread(patch_path)[:,:,::-1].copy()
                patches[i] = torch.tensor(patch, dtype=float)
            elif self.attack_method == 'usap':
                patch = init_usap_patch()
                patch = patch*255
            elif self.attack_method == 'clean':
                pass
            else:
                raise NotImplementedError
            with open(image_path.replace('.png','.json'), 'r') as file:
                annotations = json.load(file)
                label = np.array(([i, 0]+annotations["bbox"]), dtype=float)
                rpoint = np.array(annotations["rpoints"], dtype=float)
                labels[i] = torch.tensor(label, dtype=float)
                rpoints[i] = torch.tensor(rpoint, dtype=float)
        return images, patches, labels, rpoints
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
class EADYOLODataset():
    def __init__(self, split, batch_size, max_steps, attack_method) -> None:
        if split == 'train':
            self.dataset = [os.path.join(f'dataset/{split}', str(car_idx).zfill(2), str(view_idx).zfill(3)) for car_idx in range(70000, 83600, 17*10) for view_idx in range(150)]
        elif split == 'test':
            self.dataset = [os.path.join(f'dataset/{split}', str(car_idx).zfill(2), str(view_idx).zfill(3)) for car_idx in range(10000, 13400, 17) for view_idx in range(150)]
        else:
            raise NotImplementedError
    
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.attack_method = attack_method
    
    def shuffle(self):
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        images = torch.zeros((self.batch_size, self.max_steps, 256, 256, 3))
        patches = torch.ones((self.batch_size, self.max_steps, 256, 256, 3))*(-255)
        labels = torch.zeros((self.batch_size, self.max_steps, 6))
        rpoints = torch.zeros((self.batch_size, self.max_steps, 4, 2))
        for batch_idx in range(self.batch_size):
            car_path, instance_idx = os.path.split(self.dataset[index*self.batch_size+batch_idx])
            instance_idx = int(instance_idx)
            for step_idx in range(self.max_steps):
                if step_idx == 0:
                    image_path = os.path.join(car_path, str(instance_idx).zfill(3)+'.png')
                else:
                    image_path = os.path.join(car_path, str(np.random.randint(0,150)).zfill(3)+'.png')
                image = cv2.imread(image_path)[:,:,::-1].copy()
                # image = cv2.resize(image, (256, 256))
                images[batch_idx, step_idx] = torch.tensor(image, dtype=float)
                with open(image_path.replace('.png','.json'), 'r') as file:
                    annotations = json.load(file)
                    label = np.array(([batch_idx, 0]+annotations["bbox"]), dtype=float)
                    rpoint = np.array(annotations["rpoints"], dtype=float)
                    labels[batch_idx, step_idx] = torch.tensor(label, dtype=float)
                    rpoints[batch_idx, step_idx] = torch.tensor(rpoint, dtype=float)

            patch_path = os.path.join(car_path, self.attack_method+'.png').replace('test','patch_train')
            if self.attack_method == 'EOT' or self.attack_method == 'SIB' or self.attack_method == 'CAMOU':
                patch = cv2.imread(patch_path)[:,:,::-1].copy()
                patches[batch_idx] = torch.tensor(patch, dtype=float)
            elif self.attack_method == 'usap':
                patch = init_usap_patch()
                patches[batch_idx] = patch*255
            elif self.attack_method == 'clean':
                pass
            else:
                raise NotImplementedError

            
        return images, patches, labels, rpoints
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    