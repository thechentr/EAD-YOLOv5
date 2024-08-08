import cv2
import json
import torch
import os
from torch.utils.data import Dataset
import glob

def patch_collate_fn(batch_list):
    return batch_list[0][0], batch_list[0][1], batch_list[0][2]

class TrainCarlaPatchDataset(Dataset):
    def __init__(self, car_idx, split='test', front_only=False) -> None:
        super(TrainCarlaPatchDataset).__init__()
        self.dataset_path = os.path.join(f'dataset/{split}', str(car_idx))
        if split == 'train':
            self.dataset = [f'{str(i).zfill(3)}.png' for i in range(0, 150)]
            if front_only:
                self.dataset = [f'{str(i*30+15).zfill(3)}.png' for i in range(0, 5)]
        elif split == 'test':
            self.dataset = [f'{str(i).zfill(3)}.png' for i in range(0, 150)]
            if front_only:
                self.dataset = [f'{str(i*30+15).zfill(3)}.png' for i in range(0, 5)]
        else:
            raise NotImplementedError

    
    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_path, self.dataset[index])
        image = cv2.imread(image_path)[:,:,::-1].copy()
        image = torch.tensor(image, dtype=torch.float)
        with open(image_path.replace('.png','.json'), 'r') as file:
            label = json.load(file)
            
        rpoints = torch.tensor(label['rpoints'], dtype=int)

        return image, rpoints

    def __len__(self):
        return len(self.dataset)
    
class UniversalCarlaPatchDataset(Dataset):
    def __init__(self, dataset_path='dataset/train') -> None:
        self.dataset_path = dataset_path
        self.dataset = glob.glob(f'{dataset_path}/*/*.png')

    def __getitem__(self, index):
        image_path = self.dataset[index]
        image = cv2.imread(image_path)[:,:,::-1].copy()
        image = torch.tensor(image, dtype=torch.float)
        with open(image_path.replace('.png','.json'), 'r') as file:
            label = json.load(file)

        rpoints = torch.tensor(label['rpoints'], dtype=int)

        return image, rpoints

    def __len__(self):
        return len(self.dataset)