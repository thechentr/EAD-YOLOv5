import torch
import utils.modelTool as modelTool
from dataset_yolo import EADYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from logger import Logger
from utils.loss import ComputeLoss
from eval_ead_online import eval_online
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

def main(epoch_number, batch_size):
    device = torch.device('cuda:0')
    
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
    model.eval()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5_2000.pt', detModel=model)

    max_steps = 4
    ead = modelTool.get_ead_model(max_steps=max_steps).to(device)
    ead.load_state_dict(torch.load('checkpoints/ead_offline_paper.pt'), strict=False)
    ead.load_state_dict(torch.load('checkpoints/ead_online_paper.pt'), strict=False)
    

    
    render = EG3DRender(device=device)

    optimizer = modelTool.get_optimizer(ead, lr=0.0001)
    compute_loss = ComputeLoss(model)  # init loss class

    loss_logger = Logger(name='EAD YOLO Loss', path='logs')
    mAP_logger = Logger(name='EAD YOLO mAP', path='logs')

    eval_online(batch_size=40, model=model, policy=ead, max_steps=4, device=device)
    # eval_online(batch_size=40, model=model, policy=ead, max_steps=4, device=device, attack_method='noise')
    # eval_online(batch_size=40, model=model, policy=ead, max_steps=4, device=device, attack_method='SIB')
    # eval_online(batch_size=40, model=model, policy=ead, max_steps=4, device=device, attack_method='UAP')
    # eval_online(batch_size=40, model=model, policy=ead, max_steps=4, device=device, attack_method='CAMOU')


    pm = PatchManager('noise', 'dataset/patch_train')
    for epoch in range(epoch_number):
        seeds_list = list(range(70000, 83600, 17))
        add_patch = (torch.rand(batch_size) < 0.4).tolist()
        for seeds in tqdm(seed_loader(seeds_list, batch_size), total=len(seeds_list)//4):

            patch_tensor = upsample_patch(pm.load_patch(seeds)).to(device).permute(0, 3, 1, 2)
            imgs_seq_tensor = torch.empty(batch_size, max_steps, 3, 256, 256, device=device)
            features_seq_tensor = torch.empty(batch_size, max_steps, 256, 8, 8, device=device)

        
            for step in range(max_steps//2):

                if step == 0:
                    init_state = torch.zeros((batch_size, 2), dtype=torch.float32, requires_grad=True, device=device)
                    imgs_tensor, rpoints = render.reset(seeds, init_state)
                else:
                    imgs_tensor, rpoints = render.step(action)
                
                for i in range(imgs_tensor.shape[0]):
                    if add_patch[i]:
                        patch = TF.perspective(patch_tensor[i], opoints(patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
                        imgs_tensor[i] = torch.where(patch.mean(0) == -1, imgs_tensor[i], patch)

                imgs_tensor = imgs_tensor.permute(0, 2, 3, 1) * 255


                feats = model.ead_stage_1(imgs_tensor.unsqueeze(1))
                features_seq_tensor[:, step*2] = feats.squeeze(1)
                
                refined_feats = ead(features_seq_tensor[:, :step*2+1])
                preds, train_out = model.ead_stage_2(imgs_tensor, refined_feats)
                action = ead.get_action(refined_feats)

                # with torch.no_grad():
                #     post_process_pred(preds, imgs_tensor.permute(0, 3, 1, 2)/255, conf_thres=0.5)

                imgs_tensor, rpoints = render.step(action)
                targets = _annotate(imgs_tensor.permute(0, 2, 3, 1)).to(device)


                for i in range(imgs_tensor.shape[0]):
                    if add_patch[i]:
                        patch = TF.perspective(patch_tensor[i], opoints(patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
                        imgs_tensor[i] = torch.where(patch.mean(0) == -1, imgs_tensor[i], patch)

                imgs_tensor = imgs_tensor.permute(0, 2, 3, 1) * 255

                feats = model.ead_stage_1(imgs_tensor.unsqueeze(1))
                features_seq_tensor[:, step*2+1] = feats.squeeze(1)

                refined_feats = ead(features_seq_tensor[:, :step*2+2])
                preds, train_out = model.ead_stage_2(imgs_tensor, refined_feats)
                action = ead.get_action(refined_feats)

                # with torch.no_grad():
                #     post_process_pred(preds, imgs_tensor.permute(0, 3, 1, 2)/255, conf_thres=0.5)
                

                loss, loss_items = compute_loss(train_out, targets, loss_items=['box', 'obj'])  # loss scaled by batch_size
                loss_logger.add_value(loss.item())
                loss_logger.plot()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                features_seq_tensor = features_seq_tensor.detach().clone()

                
        with torch.no_grad():
            _, _, _, _, mAP = eval_online(batch_size=40, model=model, policy=ead, max_steps=4, device=device)
            mAP_logger.add_value(mAP)
            mAP_logger.plot()
            torch.save(ead.state_dict(), 'checkpoints/ead_online_paper.pt')
    

if __name__ == '__main__':
    modelTool.seed_everything()
    main(200, 4)
