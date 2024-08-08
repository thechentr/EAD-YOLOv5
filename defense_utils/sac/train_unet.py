import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from defense_utils.sac.dataset import CarlaDatasetSAC
from torch.utils.data import DataLoader
from defense_utils.sac.unet import UNet
from patch import upsample_patch, apply_patch
from logger import Logger

def main(attack_method):
    epoch_number = 1
    unet = UNet(3, 1, bilinear=True, base_filter=16)
    unet.cuda()
    unet.train()
    dataset = CarlaDatasetSAC(attack_method)
    dataloader = DataLoader(dataset,batch_size=128,shuffle=True,drop_last=True, num_workers=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(unet.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)

    logger = Logger(name='unet loss', path='logs')
    for epoch in range(0,epoch_number):
        for iteration,(images, patches, rpoints) in enumerate(dataloader):
            images = images.cuda()
            patches = patches.cuda()
            rpoints = rpoints.cuda()

            patches = upsample_patch(patches)
            
            if np.random.randint(0,100)<100:
                images = apply_patch(images, patches, rpoints)
                
            else:
                patches = torch.where(patches>0, torch.rand((128,1,1,3)).cuda()*255, patches)
                images = apply_patch(images, patches, rpoints)
            patches = torch.where(patches<0, patches, torch.ones_like(patches)*255)
            masks = apply_patch(torch.zeros_like(images), patches, rpoints)
            
            pred_mask = unet(images.permute(0,3,1,2)/255)

            loss = criterion(pred_mask, torch.mean(masks.permute(0,3,1,2)/255,dim=1,keepdim=True))
            print('epoch: {}/{}, iteration: {}/{}, loss = {:.5f}'.format(epoch, epoch_number, iteration, len(dataloader), loss.item()))
            logger.add_value(loss.item())
            logger.plot()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(unet.parameters(), 0.1)
            optimizer.step()
            with torch.no_grad():
                cv2.imwrite('obj.png',images[0].detach().cpu().numpy()[:,:,::-1])
                cv2.imwrite('mask.png',masks[0].detach().cpu().numpy()[:,:,::-1])
                cv2.imwrite('pred_mask.png',torch.sigmoid(pred_mask)[0].detach().permute(1,2,0).cpu().numpy()[:,:,::-1]*255)
        torch.save(unet.state_dict(),f'defense_utils/sac/unet_eot.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-am', '--attack_method', type=str, help='attack method')
    args = parser.parse_args()
    main(args.attack_method)