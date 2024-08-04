
import torch
import utils.modelTool as modelTool
from dataset_yolo import EADYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from logger import Logger
from utils.loss import ComputeLoss
from eval_ead import evaluation
from patch import apply_patch, upsample_patch
import cv2

def main(epoch_number, batch_size):
    device = torch.device('cuda:0')
    
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolo_carla.pt', freeze = 17, device=device)
    model.eval()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolo_carla.pt', detModel=model)

    max_steps = 4
    ead = modelTool.get_ead_model(max_steps=max_steps)

    dataset = EADYOLODataset(split='train', batch_size=batch_size, max_steps=max_steps, attack_method='usap')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=yolo_collate_fn, drop_last=True)

    optimizer = modelTool.get_optimizer(ead, lr=0.0001)
    compute_loss = ComputeLoss(model)  # init loss class

    loss_logger = Logger(name='EAD RL YOLO Loss', path='checkpoints')
    mAP_logger = Logger(name='EAD RL YOLO mAP', path='checkpoints')
    ead.load_state_dict(torch.load('checkpoints/ead_offline.pt'))

    for epoch in range(epoch_number):
        dataset.shuffle()
        for iter, (images, patches, targets, rotated_points) in enumerate(dataloader):
            images = images.cuda()
            patches = patches.cuda()
            targets = targets.cuda()
            rotated_points = rotated_points.cuda()

            patches = upsample_patch(patches)
            images = apply_patch(images, patches, rotated_points, patch_ratio=0.2)

            for step in range(max_steps):
                cv2.imwrite(f'test{step}.png', images[0,step].detach().cpu().numpy()[:,:,::-1])
            
            with torch.no_grad():
                feats = model.ead_stage_1(images)
            refined_feats = ead(feats)
            
            print(ead.get_value(refined_feats))

            preds, train_out = model.ead_stage_2(refined_feats)
            loss, loss_items = compute_loss(train_out, targets[:,-1,:], loss_items=['box', 'obj'])  # loss scaled by batch_size
            loss_logger.add_value(loss.item())
            loss_logger.plot()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # with torch.no_grad():
        #     _, _, _, _, mAP = evaluation(batch_size=40, model=model, policy=ead, max_steps=4, attack_method='clean')
        #     mAP_logger.add_value(mAP)
        #     mAP_logger.plot()
        #     torch.save(ead.state_dict(), 'checkpoints/ead_offline.pt')

    
    

if __name__ == '__main__':
    modelTool.seed_everything()
    main(200, 8)


