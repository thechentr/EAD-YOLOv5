import torch
import utils.modelTool as modelTool
from dataset_yolo import YOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from logger import Logger
from utils.loss import ComputeLoss
from eval import evaluation
from patch import init_usap_patch, apply_patch


def main(epoch_number, batch_size):
    device = torch.device('cuda:0')
    
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
    model.train()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5n.pt', detModel=model)

    dataset = YOLODataset(split='train', batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=yolo_collate_fn)

    optimizer = modelTool.get_optimizer(model, lr=0.001)
    compute_loss = ComputeLoss(model)  # init loss class

    loss_logger = Logger(name='YOLO Loss', path='checkpoints')
    mAP_logger = Logger(name='YOLO mAP', path='checkpoints')

    for epoch in range(epoch_number):
        dataset.shuffle()
        for iter, (imgs_tensor, targets, rotated_points) in enumerate(dataloader):
            model.train()
            imgs_tensor = imgs_tensor.cuda()
            targets = targets.cuda()
            rotated_points = rotated_points.cuda()


            patch = init_usap_patch()
            patch = patch*255
            patch = patch.unsqueeze(0)
            patch = patch.repeat(imgs_tensor.shape[0],1,1,1)
            imgs_tensor = apply_patch(imgs_tensor, patch, rotated_points, patch_ratio=0.2)

            imgs_tensor = imgs_tensor.permute(0,3,1,2)/255
            pred, _ = model(imgs_tensor)  # forward

            loss, loss_items = compute_loss(pred, targets.to(device), loss_items=['box', 'obj'])  # loss scaled by batch_size
            loss_logger.add_value(loss.item())
            loss_logger.plot()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, _, _, _, mAP = evaluation(batch_size=40, model=model, attack_method='clean')
            mAP_logger.add_value(mAP)
            mAP_logger.plot()
            torch.save({'model': model}, 'checkpoints/yolo_carla.pt')

    
    

if __name__ == '__main__':
    modelTool.seed_everything()
    main(20, 128)
