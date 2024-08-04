from models.yolo import Model
from utils.general import check_img_size, intersect_dicts
from utils.torch_utils import de_parallel
import random
from embodiment.transformer import Transformer

import numpy as np
import torch
import torch.nn as nn

def get_ead_model(max_steps):
    C,H,W = 64, 32, 56
    ead = Transformer(num_layers=2,
                        num_heads=8,
                        num_blocks=(max_steps*C),
                        residual_pdrop=0.1,
                        attention_pdrop=0.1,
                        embedding_pdrop=0.1,
                        embedding_dim=H*W,
                        vertical_scale=15,
                        horizontal_scale=60,
                        action_dim=2).cuda()
    return ead

def get_det_model(pretrain_weights, device, freeze=0):
    ckpt = torch.load(pretrain_weights, map_location='cpu')
    model = Model(ckpt["model"].yaml, ch=3, nc=1, anchors='').to(device)
    

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(128, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    
    accumulate = 1
    weight_decay = 0.0005

    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)

    hyp = {
        'anchor_t' : 4.0,
        'box' : 0.05 * 3 / nl,
        'cls' : 0.5 * 3 / nl,
        'obj' : (imgsz / 128) ** 2 * 3 / nl,
        'cls_pw' : 1.0,
        'obj_pw' : 1.0,
        'label_smoothing' : 0.0,
        'fl_gamma' : 0.0,
        'weight_decay' : weight_decay,
        'accumulate': accumulate
    }

    model.nc = 1  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.names = {0 : 'car'}

    # Freeze backbone ------------------------------------------------------------------
    if freeze > 0:
        freeze = list(range(freeze))
        print(freeze)
        freeze = [f"model.{x}." for x in freeze]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False

    return model


def get_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer

def transfer_paramaters(pretrain_weights, detModel=None, policyModel=None, optimizer=None, scheduler=None):
    ckpt = torch.load(pretrain_weights, map_location='cpu')
    if detModel is not None:
        exclude = ['anchor']
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, detModel.state_dict(), exclude=exclude)
        detModel.load_state_dict(csd, strict=False)
        print(f"🚀 Transferred {len(csd)}/{len(detModel.state_dict())} items from {pretrain_weights} to yolo 🚀")
    if policyModel is not None:
        csd = ckpt['policy'].float().state_dict()
        csd = intersect_dicts(csd, policyModel.state_dict())
        policyModel.load_state_dict(csd, strict=False)
        print(f"🛰️ Transferred {len(csd)}/{len(policyModel.state_dict())} items from {pretrain_weights} to EAD 🛰️")
    if optimizer is not None and 'optimizer_state_dict' in ckpt.keys():
        csd = ckpt['optimizer_state_dict']
        optimizer.load_state_dict(csd)
        print(f"📡 Transferred from {pretrain_weights} to optimizer 📡")
    if scheduler is not None and 'scheduler_state_dict' in ckpt.keys():
        csd = ckpt['scheduler_state_dict']
        scheduler.load_state_dict(csd)
        print(f"🔭 Transferred from {pretrain_weights} to scheduler 🔭")
    
    return ckpt['ni'] if 'ni' in ckpt.keys() else 0

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

