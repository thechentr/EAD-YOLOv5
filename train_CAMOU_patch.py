import torch
import cv2
import utils.modelTool as modelTool
from dataset import TrainCarlaPatchDataset
from patch import apply_patch, upsample_patch
from utils.post_process import post_process_pred
from torch.utils.data import DataLoader
from logger import Logger
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as F
from generator import prob_fix_color, gumbel_color_fix_seed
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

def seed2texture(seeds_train,seeds_fixed,control_point,coordinates,h,w):
    seeds_ratio = 0.7
    temp = 0.7
    seeds = seeds_ratio * seeds_train + (1 - seeds_ratio) * seeds_fixed
    blur=1 # 计算像素颜色概率值的平滑系数
    prob_map = prob_fix_color(control_point, coordinates, colors, h, w, blur=blur).unsqueeze(0)

    # 概率值平滑处理
    prob_map = camouflage_kernel(prob_map)
    prob_map = prob_map.squeeze(0).permute(1, 2, 0)
    gb = -(-(seeds + 1e-20).log() + 1e-20).log()
    adv_texture = gumbel_color_fix_seed(prob_map, gb, colors, tau=temp, type='gumbel')
    adv_texture = expand_kernel(adv_texture.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    return adv_texture

num_colors=4
n_init = 10
batch_size = 100
iteration_number = 200
device = torch.device('cuda:0')

modelTool.seed_everything()
model = modelTool.get_det_model(pretrain_weights='checkpoints/freeze17_7000_4step_usap_1500.pt', freeze = 17, device=device)
modelTool.transfer_paramaters(pretrain_weights='checkpoints/freeze17_7000_4step_usap_1500.pt', detModel=model)
model.eval()


num_points = 40
resolution = 1
h, w = int(32 / resolution), int(64 / resolution)
lr_points = 5e-4
lr_seeds = 5e-3

k = 3
k2 = k * k
camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, int(k / 2)).to(device)
camouflage_kernel.weight.data.fill_(0)
camouflage_kernel.bias.data.fill_(0)
for i in range(num_colors):
    camouflage_kernel.weight[i, i, :, :].data.fill_(1 / k2)
expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
expand_kernel.weight.data.fill_(0)
expand_kernel.bias.data.fill_(0)
for i in range(3):
    expand_kernel.weight[i, i, :, :].data.fill_(1)

coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)


for car_idx in tqdm(range(10000, 13400, 17)):
    colors_set = np.zeros((20,256,256,3))
    for i in range(0,20):
        colors_set[i]=cv2.imread(f'dataset/test/{str(car_idx)}/{str(i+40).zfill(3)}.png')[:,:,::-1]
    colors_set=colors_set.reshape(colors_set.shape[0]*colors_set.shape[1]*colors_set.shape[2],-1)
    estimator = KMeans(n_clusters=num_colors,n_init=n_init)
    estimator.fit(colors_set)
    colors = estimator.cluster_centers_
    colors = torch.tensor(colors/255, dtype=torch.float32).to(device)
    control_point = torch.rand((num_colors, num_points, 2), requires_grad=True, device=device)
    seeds_train = torch.zeros(size=[h, w, num_colors], device=device).uniform_(0.01,1 - 0.01).requires_grad_(True)
    seeds_fixed = torch.zeros(size=[h, w, num_colors], device=device).uniform_()
    optimizer = optim.Adam([control_point], lr=lr_points)
    optimizer_seed = optim.Adam([seeds_train], lr=lr_seeds)


    print(f'train CAMOU patch for car {car_idx}')
    logger = Logger('patch loss')
    dataset = TrainCarlaPatchDataset(car_idx, split='test')
    datalodaer = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    iteration = 0
    while iteration < iteration_number:
        for _, (image, rotated_points) in enumerate(datalodaer):
            image = image.cuda()
            rotated_points = rotated_points.cuda()
            patch = seed2texture(seeds_train,seeds_fixed,control_point,coordinates,h,w)
            patch = patch.permute(0,3,1,2)/255
            patch = F.resize(patch, size=[64,111])
            
            patch = patch.permute(0,2,3,1)*255
            
            eot_patch = patch*255
            eot_patch = eot_patch.repeat(image.shape[0],1,1,1)
            eot_patch = upsample_patch(eot_patch)

            adv_image = apply_patch(image, eot_patch, rotated_points)

            adv_image = adv_image.permute(0,3,1,2)/255
            preds = model(adv_image) # forward

            output_objectness = preds[0][0][:, :, 4]  # [batch, anchors, objectness]
            max_conf, max_conf_idx = torch.max(output_objectness, dim=1)

            loss = max_conf.mean()
            
            optimizer.zero_grad()
            optimizer_seed.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_seed.step()
            logger.add_value(max_conf.mean().item())
            logger.plot()

            seeds_train.data.clamp_(0, 1 - 0) #控制采样概率优化的上下界
            control_point.data = control_point.data.clamp(0, 1)

            with torch.no_grad():
                post_process_pred(preds, adv_image[0:4], conf_thres=0.5)
            iteration += 1
            patch_path = f'dataset/patch_train/{str(car_idx).zfill(2)}'
            if not os.path.exists(patch_path):
                os.mkdir(patch_path)
            cv2.imwrite(os.path.join(patch_path, 'CAMOU.png'), patch[0].detach().cpu().numpy()[:,:,::-1]*255)