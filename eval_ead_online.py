import torch
import utils.modelTool as modelTool
from dataset_yolo import EADYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from utils.logger import Logger
from utils.loss import ComputeLoss
from eval_ead import evaluation
from patch import apply_patch, upsample_patch, PatchManager
from utils.post_process import post_process_pred
from utils.visualize import draw_boxes_on_grid_image
import cv2
from EG3Drender.render import EG3DRender
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
from tqdm import tqdm 

def seed_loader(seed_list, batch_size=4):  
    seed_list = list(seed_list)  
    random.shuffle(seed_list)  
    for i in range(0, len(seed_list), batch_size):  
        batch = seed_list[i:i + batch_size]  
        if len(batch) == batch_size:  
            yield batch  

def opoints(img):
    return [[0, 0], 
            [0, img.shape[1]], 
            [img.shape[2], img.shape[1]], 
            [img.shape[1], 0]]

@torch.no_grad
def _annotate(imgs_tensor):
    """
    imgs_tensor:[bs, w, h, c] (0, 255)
    """
    label = []
    for i in range(imgs_tensor.shape[0]):
        box = [0, *_calculate_box(imgs_tensor[i])]
        if box[-1] + box[-2] < 1.8:  # 过滤掉没有物体的box
            label.append([i, *box])
    label = torch.tensor(label, dtype=torch.float32)

    return label

@torch.no_grad
def _calculate_box(img_tensor):
    gray = torch.sum(img_tensor, dim=2)/3
    def get_dim_len(dim):
        minn = 0
        maxx = 255
        gray_dim = torch.sum(gray, dim=dim)/256
        for i in range(0, 256):
            if gray_dim[i] <= 0.995 and minn == 0:
                minn = i
            if gray_dim[255-i] <= 0.995 and maxx == 255:
                maxx = 255-i
        return (maxx+minn)/2, maxx-minn
    x, width = get_dim_len(0)
    y, height = get_dim_len(1)
    return x/256, y/256, width/256, height/256
   

from eval_ead import calculate_merit

@torch.no_grad()
def eval_online(batch_size=20,  # batch size
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.6,  # NMS IoU threshold
                device=torch.device('cuda'),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                model=None,
                policy=None,
                max_steps=None,
                attack_method='clean',):
    
    green = lambda x: f'\033[0;32m{x}\033[0m' 
    print(green(f'\nstart test:\t attack method-{attack_method}'))
    
    pm = PatchManager(attack_method, 'dataset/patch_train') if attack_method != 'clean' else None
    render = EG3DRender(device=device)

    model.eval()
    policy.eval()

    preds_list= []
    targets_list =[]
    seeds_list = list(range(10000, 13400, 17))
    for seeds in seed_loader(seeds_list, batch_size):

        patch_tensor = upsample_patch(pm.load_patch(seeds)).to(device).permute(0, 3, 1, 2) if attack_method != 'clean' else None
        imgs_seq_tensor = torch.empty(batch_size, max_steps, 3, 256, 256, device=device)
        features_seq_tensor = torch.empty(batch_size, max_steps, 64, 32, 32, device=device)

    
        for step in range(max_steps):

            if step == 0:
                init_state = torch.zeros((batch_size, 2), dtype=torch.float32, requires_grad=True, device=device)
                imgs_tensor, rpoints = render.reset(seeds, init_state)
            else:
                imgs_tensor, rpoints = render.step(action)

            targets = _annotate(imgs_tensor.permute(0, 2, 3, 1)).to(device)
            
            if attack_method != 'clean':
                for i in range(imgs_tensor.shape[0]):
                    patch = TF.perspective(patch_tensor[i], opoints(patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
                    imgs_tensor[i] = torch.where(patch.mean(0) == -1, imgs_tensor[i], patch)

            imgs_tensor = imgs_tensor.permute(0, 2, 3, 1) * 255


            feats = model.ead_stage_1(imgs_tensor.unsqueeze(1))
            features_seq_tensor[:, step] = feats.squeeze(1)
            
            refined_feats = policy(features_seq_tensor[:, :step*2+1])
            preds, train_out = model.ead_stage_2(refined_feats)
            action = policy.get_action(refined_feats)
            # action = torch.tensor([15., 60.], device=device) * torch.randn_like(action)

            # print(action.shape)
            # input(action)
            

            with torch.no_grad():
                input('k')
                post_process_pred(preds[:4], imgs_tensor[:4].permute(0, 3, 1, 2)/255, conf_thres=0.5)
                

        preds_list.append(preds)
        targets_list.append(targets)


    return calculate_merit(targets_list, preds_list, [batch_size, 3, 256, 256], conf_thres,iou_thres, device)

    
    

if __name__ == '__main__':
    modelTool.seed_everything()
    device = torch.device('cuda:0')
    
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
    model.eval()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5_2000.pt', detModel=model)

    max_steps = 4
    ead = modelTool.get_ead_model(max_steps=max_steps).to(device)
    ead.load_state_dict(torch.load('checkpoints/ead_offline_s4.pt'), strict=False)
    # ead.load_state_dict(torch.load('checkpoints/ead_online_SOTA.pt'), strict=False)
    # ead.load_state_dict(torch.load('checkpoints/ead_online_RL.pt'), strict=False)
    

    # eval_online(batch_size=1, model=model, policy=ead, max_steps=4, device=device)
    # eval_online(batch_size=1, model=model, policy=ead, max_steps=4, device=device, attack_method='EOT')
    # eval_online(batch_size=1, model=model, policy=ead, max_steps=4, device=device, attack_method='SIB')
    eval_online(batch_size=1, model=model, policy=ead, max_steps=4, device=device, attack_method='UAP')
    eval_online(batch_size=1, model=model, policy=ead, max_steps=4, device=device, attack_method='CAMOU')