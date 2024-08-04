import torch
import cv2
import math
from torchvision.transforms import functional as F
import torchvision
import json
import os
import numpy as np


car_idx = 0
view_idx = 0
car_path = f'dataset/test/{str(car_idx).zfill(2)}'
img = cv2.imread(os.path.join(car_path, f'{str(view_idx).zfill(3)}.png'))[:,:,::-1].copy()/255
img = torch.tensor(img, dtype=torch.float).cuda().permute(2,0,1).unsqueeze(0)
with open(os.path.join(car_path, f'{str(view_idx).zfill(3)}.json'), 'r') as file:
    label = json.load(file)

patch = torch.ones_like(img).uniform_(0,1)
points_orig = [[0, 0], [0, img.shape[2]], [img.shape[3], img.shape[2]], [img.shape[3], 0]]
rotated_points = label['rpoints']
patch = F.perspective(patch, points_orig, rotated_points, fill=-1)
img = torch.where(patch==-1, img, patch)
img = img[0].permute(1,2,0).cpu().numpy()[:,:,::-1]*255
print(img.shape)

bbox = label['bbox']
x = bbox[0]*img.shape[1]
y = bbox[1]*img.shape[0]
width = bbox[2]*img.shape[1]
height = bbox[3]*img.shape[0]

x_min = int(x - width / 2)
y_min = int(y - height / 2)
x_max = int(x + width / 2)
y_max = int(y + height / 2)
img = cv2.rectangle(img.copy(), (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
cv2.imwrite('test.png',img)