import torch.nn as nn
import torch.nn.functional as F
from defense_utils.jpeg.compression import compress_jpeg
from defense_utils.jpeg.decompression import decompress_jpeg
from defense_utils.jpeg.utils import quality_to_factor
from defense_utils.jpeg.utils import diff_round

class DiffJPEG(nn.Module):
    '''
       reference: https://github.com/mlomnitz/DiffJPEG
    '''
    def __init__(self, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        factor = quality_to_factor(quality)
        rounding = diff_round
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(factor=factor)

    def forward(self, x):
        '''
        '''
        x = x.permute(0,3,1,2)
        # x = F.interpolate(x, [256, 448], mode='bilinear')
        b, c, h, w = x.shape
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, h, w, cb, cr)
        # x = F.interpolate(x, [256, 444], mode='bilinear')
        recovered = recovered.permute(0,2,3,1)
        return recovered