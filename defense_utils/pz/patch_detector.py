import torch
from defense_utils.sac.unet import UNet
from defense_utils.sac.patch_detector import ThresholdSTEFunction
import numpy as np
import math
from sklearn.cluster import KMeans

class PatchDetector(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base_filter=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), unet=None,):
        super(PatchDetector, self).__init__()
        if unet:
            self.unet = unet
        else:
            self.unet = UNet(n_channels, n_classes, bilinear=bilinear, base_filter=base_filter)
        self.device = device

    def filter(self, xs, bpda=False):
        ys = torch.zeros_like(xs)
        mask_list = []  # mask: 1 patch; 0 background
        raw_mask_list = []
        idx=0
        for x in xs:
            x = x.unsqueeze(0).to(self.device)
            mask = self.unet(x)
            mask = torch.sigmoid(mask)
            if bpda:
                raw_mask = ThresholdSTEFunction.apply(mask)
            else:
                raw_mask = (mask > 0.5).float()
            raw_mask_list.append(raw_mask)
            mask = raw_mask
            mask_list.append(mask)
            mask = torch.cat((mask, mask, mask), 1)
            mask = 1.0 - mask
            means = torch.ones_like(mask[0])*torch.FloatTensor([0.485, 0.456, 0.406]).cuda().expand(mask[0].shape[1],mask[0].shape[2],3).permute(2,0,1)
            ys[idx]= torch.where(mask[0]==0,means,x[0])
            idx+=1
        return ys, mask_list, raw_mask_list
    
    def forward(self,input):
        input = input.permute(0,3,1,2)/255
        output, _, _ = self.filter(input, bpda=False)
        return output.permute(0,2,3,1)*255
