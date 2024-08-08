import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from env import EADEnv
from EG3DEnv import EG3DEnv
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.data import DataLoader
from logger import Logger

import utils.modelTool as modelTool
from utils.loss import ComputeLoss
from embodiment.transformer import MLP


device = torch.device('cuda:0')
ead = modelTool.get_ead_model(max_steps=4)
ead.load_state_dict(torch.load('checkpoints/ead_offline.pt'))

mlp2 = MLP(64*4*4, 64*4*4, 2, num_layers=4)
bn2 = nn.BatchNorm1d(64*4*4)

mlp1 = MLP(64*4*4, 64*4*4, 1, num_layers=4)
bn1 = nn.BatchNorm1d(64*4*4)

ead.action_decoder.mlp = mlp2
ead.action_decoder.bn = bn2
ead.value_decoder.mlp = mlp1
ead.value_decoder.bn = bn1
torch.save(ead.state_dict(), 'checkpoints/ead_offlinecp.pt')
