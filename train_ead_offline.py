import torch
import utils.modelTool as modelTool
from dataset_yolo import EADYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from utils.logger import Logger
from utils.loss import ComputeLoss
from eval_ead import evaluation
from patch import apply_patch, upsample_patch
from utils.post_process import post_process_pred
import cv2
from tqdm import tqdm 

def main(epoch_number, batch_size):
    device = torch.device('cuda:0')
    
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
    model.eval()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5_2000.pt', detModel=model)

    max_steps = 4
    ead = modelTool.get_ead_model(max_steps=max_steps)
    # ead.load_state_dict(torch.load('checkpoints/ead_offline.pt'), strict=False)

    dataset = EADYOLODataset(split='train', batch_size=batch_size, max_steps=max_steps, attack_method='usap')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=yolo_collate_fn, drop_last=True)

    optimizer = modelTool.get_optimizer(ead, lr=0.00001)
    compute_loss = ComputeLoss(model)  # init loss class

    loss_logger = Logger(name='EAD YOLO Loss', path='logs')
    mAP_logger = Logger(name='EAD YOLO mAP', path='logs')


    evaluation(batch_size=40, model=model, policy=ead, max_steps=4, attack_method='clean')
    for epoch in range(epoch_number):
        for iter, (images, patches, targets, rotated_points) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = images.cuda()
            patches = patches.cuda()
            targets = targets.cuda()
            rotated_points = rotated_points.cuda()

            patches = upsample_patch(patches)
            images = apply_patch(images, patches, rotated_points, patch_ratio=0.2)

            # for step in range(max_steps):
            #     cv2.imwrite(f'logs/test{step}.png', images[0,step].detach().cpu().numpy()[:,:,::-1])
            
            for step in range(1, 5):
                with torch.no_grad():
                    feats = model.ead_stage_1(images[:, :step])
                refined_feats = ead(feats)
                preds, train_out = model.ead_stage_2(refined_feats)
                loss, loss_items = compute_loss(train_out, targets[:,step-1,:], loss_items=['box', 'obj'])  # loss scaled by batch_size
                loss_logger.add_value(loss.item())
                loss_logger.plot()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                with torch.no_grad():
                    iimg = images[0:4, step-1, :].squeeze(1).permute(0, 3, 1, 2) * 255
                    post_process_pred(preds, iimg, conf_thres=0.5)

        with torch.no_grad():
            print(epoch, '--------------------------')
            _, _, _, _, mAP = evaluation(batch_size=40, model=model, policy=ead, max_steps=4, attack_method='clean')
            mAP_logger.add_value(mAP)
            mAP_logger.plot()
            torch.save(ead.state_dict(), 'checkpoints/ead_offline_s4.pt')

    
    

if __name__ == '__main__':
    modelTool.seed_everything()
    main(200, 256)
