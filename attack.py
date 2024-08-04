import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

def PGD_attack_step(patch,adv_loss,learning_rate):
    grad = torch.autograd.grad(adv_loss, [patch])[0].detach()
    grad = grad.sign()
    patch = patch - grad*learning_rate
    patch = torch.clamp(patch,0,1)
    return patch

def MIM_attack_step(patch,adv_loss,momentum,learning_rate,decay_factor):
    grad = torch.autograd.grad(adv_loss, [patch])[0].detach()
    grad_norm = torch.norm(grad, p=1, dim=(1,2), keepdim=True)
    grad = grad/(grad_norm + 1e-6)
    momentum =  grad + decay_factor*momentum
    update = momentum.sign()
    patch = patch - update*learning_rate
    patch = torch.clamp(patch,0,1)
    return patch, momentum


transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=(-5,5),translate=(0.1,0.1),scale=(0.9,1.1),fill=0.5)
        ])

def DIM_attack_transform(image):

    image = image.permute(0,3,1,2)

    image = transform(image/255)
    image = image.permute(0,2,3,1)*255
    return image
