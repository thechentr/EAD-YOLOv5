import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import GPT

def check_tensor(tensor, tensor_name='Tensor'):
    if torch.isnan(tensor).any():
        print(f'{tensor_name} has NaN')
        return False
    if torch.isinf(tensor).any():
        print(f'{tensor_name} has Inf')
        return False
    return True

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        num_layers
    ):
        super(MLP, self).__init__()
        assert num_layers > 1, \
            "Please use `torch.nn.Linear` to construct shallow MLP"
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(n, k) 
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (F.relu(layer(x)) if i < self.num_layers - 1 
                else layer(self.bn(x)))
        return x

class ConvDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=[3,3], stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=[3,3], stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=[3,3], stride=2, padding=1)
        self.mlp = MLP(64*4*7, 64*4*7, output_dim, num_layers=4)
        self.bn = nn.BatchNorm1d(64*4*7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.mlp(self.bn(x.reshape(x.shape[0],-1)))
        return x

class Transformer(nn.Module):
    """
    Variant of Decision Transformer, which wrap a GPT model
    """
    def __init__(
        self,
        num_layers,
        num_heads,
        num_blocks,
        residual_pdrop,
        attention_pdrop,
        embedding_pdrop,
        embedding_dim,
        action_dim,
        horizontal_scale,
        vertical_scale
    ):
        super(Transformer, self).__init__()
        # self.img_feature_embedding = nn.Linear(config.IMG_FEATURE_SIZE, config.EMBEDDING_DIM)
        # self.prediction_embedding = nn.Linear(config.BOX_EMB_SIZE + config.NUM_CLASSES + 5, config.EMBEDDING_DIM)
        self.model = GPT(
            num_layers, num_heads, num_blocks, embedding_pdrop,  attention_pdrop, residual_pdrop, 
            embedding_dim=embedding_dim, output_dim=embedding_dim,
        )

        self.action_decoder = ConvDecoder(output_dim=2)
        self.value_decoder = ConvDecoder(output_dim=1)
        

            
        scaling = torch.tensor([vertical_scale, horizontal_scale]).float()
        self.register_buffer('action_scaling', scaling)

    def forward(self, feats):
        B, S, C, H, W = feats.shape
        feats = feats.reshape(B,S*C,H*W)

        refined_feats = self.model(feats)
        refined_feats = refined_feats.reshape(B,S,C,H,W)

        refined_feats = refined_feats[:,-1,:,:,:]


        return refined_feats
    
    def get_action(self, feats):
        return self.action_scaling * torch.tanh(self.action_decoder(feats))
    
    def get_value(self, feats):
        return self.value_decoder(feats)