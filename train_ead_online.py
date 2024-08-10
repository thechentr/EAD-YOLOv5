import torch
import utils.modelTool as modelTool
from dataset_yolo import EADYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from logger import Logger
from utils.loss import ComputeLoss
from eval_ead import evaluation
from patch import apply_patch, upsample_patch, PatchManager
from utils.post_process import post_process_pred
import cv2
from EG3Drender.render import EG3DRender
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def main(epoch_number, batch_size):
    device = torch.device('cuda:0')
    
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
    model.eval()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5_2000.pt', detModel=model)

    max_steps = 4
    ead = modelTool.get_ead_model(max_steps=max_steps)
    ead.load_state_dict(torch.load('checkpoints/ead_offline.pt'), strict=False)

    
    render = EG3DRender(device=device)

    optimizer = modelTool.get_optimizer(ead, lr=0.0001)
    compute_loss = ComputeLoss(model)  # init loss class

    loss_logger = Logger(name='EAD YOLO Loss', path='logs')
    mAP_logger = Logger(name='EAD YOLO mAP', path='logs')

    import random
    def seed_loader(batch_size=4):
        seed_list = list(range(70000, 83600, 17))
        random.shuffle(seed_list)
        for i in range(0, len(seed_list), batch_size):
            yield seed_list[i:i + batch_size]
    pm = PatchManager('noise', 'dataset/patch_train')
    from tqdm import tqdm 
    for epoch in range(epoch_number):
        for seeds in seed_loader(batch_size):

            patch_tensor = upsample_patch(pm.load_patch(seeds))
            imgs_seq_tensor = torch.empty(batch_size, max_steps, 3, 256, 256)
            features_seq_tensor = torch.empty(batch_size, max_steps, 64, 32, 32)

        
            for step in range(max_steps//2):

                if step == 0:
                    init_state = torch.zeros((batch_size, 2), dtype=torch.float32, requires_grad=True, device=device)
                    imgs_tensor, rpoints = render.reset(seeds, init_state)
                else:
                    imgs_tensor, rpoints = render.step(seeds, action)

                imgs_tensor = imgs_tensor.permute(0, 2, 3, 1) * 255
                for i in range(imgs_tensor.shape[0]):
                    patch = TF.perspective(patch_tensor[i], rpoints(patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
                    imgs_tensor[i] = torch.where(patch.mean(0) == -1, imgs_tensor[i], patch)

                
                feats = model.ead_stage_1(imgs_tensor)
                features_seq_tensor[:, step*2] = feats
                
                refined_feats = ead(features_seq_tensor[:, :step*2+1])
                preds, train_out = model.ead_stage_2(refined_feats)  # HACK
                action = ead.get_action(refined_feats)

                with torch.no_grad():
                    iimg = imgs_tensor[0:4, -1, :].squeeze(1).permute(0, 3, 1, 2) * 255
                    post_process_pred(preds, iimg, conf_thres=0.5)

                imgs_tensor, rpoints = render.step(seeds, action)
                imgs_tensor = imgs_tensor.permute(0, 2, 3, 1) * 255
                targets = _annotate(imgs_tensor).to(device)


                for i in range(imgs_tensor.shape[0]):
                    patch = TF.perspective(patch_tensor[i], rpoints(patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
                    imgs_tensor[i] = torch.where(patch.mean(0) == -1, imgs_tensor[i], patch)

                feats = model.ead_stage_1(imgs_tensor)
                features_seq_tensor[:, step*2+1] = feats                

                refined_feats = ead(features_seq_tensor[:, :step*2+2])
                preds, train_out = model.ead_stage_2(refined_feats)
                action = ead.get_action(refined_feats)


                loss, loss_items = compute_loss(train_out, targets[:,-1,:], loss_items=['box', 'obj'])  # loss scaled by batch_size
                loss_logger.add_value(loss.item())
                loss_logger.plot()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    iimg = imgs_tensor[0:4, -1, :].squeeze(1).permute(0, 3, 1, 2) * 255
                    post_process_pred(preds, iimg, conf_thres=0.5)

                
                with torch.no_grad():
                    _, _, _, _, mAP = evaluation(batch_size=40, model=model, policy=ead, max_steps=4, attack_method='clean')
                    mAP_logger.add_value(mAP)
                    mAP_logger.plot()
                    torch.save(ead.state_dict(), 'checkpoints/ead_offline.pt')

    @torch.no_grad
    def _annotate(imgs_tensor):
        label = []
        for i in range(imgs_tensor.shape[0]):
            box = [0, *_calculate_box(imgs_tensor[i].permute(1,2,0))]
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


    
    

if __name__ == '__main__':
    modelTool.seed_everything()
    main(200, 4)
