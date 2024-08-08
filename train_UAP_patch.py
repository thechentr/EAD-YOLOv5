import torch
import cv2
import utils.modelTool as modelTool
from dataset import UniversalCarlaPatchDataset
from patch import init_patch, apply_patch, upsample_patch
from utils.post_process import post_process_pred
from attack import PGD_attack_step
from torch.utils.data import DataLoader
from logger import Logger
import os
import numpy as np



batch_size = 100
learning_rate = 8/255
# iteration_number = 200
iteration_number = 200
device = torch.device('cuda:0')

modelTool.seed_everything()
model = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5_2000.pt', detModel=model)
model.eval()


print(f'train UAP patch for all cars')
logger = Logger('patch loss', path='logs')
dataset = UniversalCarlaPatchDataset()
datalodaer = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

patch = init_patch().requires_grad_(True)
iteration = 0
while iteration < iteration_number:
    for _, (image, rotated_points) in enumerate(datalodaer):
        image = image.cuda()
        rotated_points = rotated_points.cuda()
        patch = patch.detach().clone().requires_grad_(True)
        eot_patch = patch.unsqueeze(0)*255
        eot_patch = eot_patch.repeat(image.shape[0],1,1,1)
        eot_patch = upsample_patch(eot_patch)
        adv_image = apply_patch(image, eot_patch, rotated_points)
        adv_image = adv_image.permute(0,3,1,2)/255
        preds = model(adv_image) # forward

        output_objectness = preds[0][0][:, :, 4]  # [batch, anchors, objectness]
        max_conf, max_conf_idx = torch.max(output_objectness, dim=1)
        loss = max_conf.mean()
        logger.add_value(loss.item())
        logger.plot()

        patch = PGD_attack_step(patch, loss, learning_rate)

        with torch.no_grad():
            post_process_pred(preds, adv_image[0:1], conf_thres=0.5)
        iteration += 1
        if iteration == iteration_number:
            break

for car_idx in range(10000, 13400, 17):
    patch_path = f'dataset/patch_train/{str(car_idx).zfill(2)}'
    if not os.path.exists(patch_path):
        os.mkdir(patch_path)
    cv2.imwrite(os.path.join(patch_path, 'UAP.png'), patch.detach().cpu().numpy()[:,:,::-1]*255)