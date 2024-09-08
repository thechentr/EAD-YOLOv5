import torch
import random
from torchvision.transforms import functional as F
import cv2
import os

def init_patch():
    patch = torch.ones((32, 64, 3)).normal_(0.5,0.1).cuda()
    return patch

def upsample_patch(patch):
    is_time_seq = False
    if len(patch.shape)==5:
        is_time_seq = True
        B, S, H, W, C = patch.shape
        patch = patch.reshape(B*S, H, W, C)
    patch = patch.permute(0,3,1,2)/255
    patch = F.resize(patch, size=[256,256])
    patch = patch.permute(0,2,3,1)*255
    if is_time_seq:
        patch = patch.reshape(B, S, 256, 256, C)
    return patch

def init_usap_patch(size=(16, 32, 3)):
    patch = torch.ones(size).uniform_(0,1)
    return patch

def apply_patch(image, patch, rotated_points, patch_ratio=1.0):
    is_time_seq = False
    if len(image.shape)==5:
        is_time_seq = True
        B, S, H, W, C = image.shape
        image = image.reshape(B*S, H, W, C)
        patch = patch.reshape(B*S, H, W, C)
        rotated_points = rotated_points.reshape(B*S, 4, 2)
    rotated_points[:,:,0] = rotated_points[:,:,0]
    rotated_points[:,:,1] = rotated_points[:,:,1]
    image = image.permute(0,3,1,2)/255
    patch = patch.permute(0,3,1,2)/255
    points_orig = [[0, 0], [0, image.shape[2]], [image.shape[3], image.shape[2]], [image.shape[3], 0]]
    for i in range(image.shape[0]):
        if random.randint(0,100) <= patch_ratio * 100:
            patch[i] = F.perspective(patch[i], points_orig, rotated_points[i], fill=-1)
        else:
            patch[i] = -torch.ones_like(patch[i])
    image = torch.where(patch<0, image, patch)
    image = image.permute(0,2,3,1)*255
    if is_time_seq:
        image = image.reshape(B, S, H, W, C)
    return image

class PatchManager(object):

    def __init__(self, method, root_dir):
        assert method in ['CAMOU', 'EOT', 'SIB', 'UAP', 'noise']
        self.root_dir = root_dir
        self.method = method


    def load_patch(self, car_seeds):
        patches = torch.rand((len(car_seeds), 32, 64, 3))
        if self.method == 'noise':
            return patches
        for i, seed in enumerate(car_seeds):
            patch_path = os.path.join(self.root_dir, str(seed), f'{self.method}.png')
            patch = cv2.imread(patch_path)[:,:,::-1].copy()
            patches[i] = torch.tensor(patch, dtype=torch.float32) / 255
        return patches
