
import argparse
import numpy as np
import torch
import utils.modelTool as modelTool
from utils.general import (non_max_suppression,scale_boxes, xywh2xyxy)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.post_process import post_process_pred
from dataset_yolo import EADYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from patch import init_usap_patch, apply_patch, upsample_patch
from pathlib import Path
from embodiment.transformer import Transformer

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    correct_class = torch.ones_like(correct_class)  # HACK
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)



@torch.no_grad()
def evaluation(
    batch_size=16,  # batch size
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    single_cls=False,  # treat as single-class dataset
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    model=None,
    policy=None,
    max_steps=None,
    save_dir=Path(""),
    plots=False,
    attack_method=False,
):
    dataset = EADYOLODataset(split='test', batch_size=batch_size, max_steps=max_steps, attack_method=attack_method)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=yolo_collate_fn, drop_last=True)


    green = lambda x: f'\033[0;32m{x}\033[0m' 
    print(green(f'\nstart test:\t attack method-{attack_method}'))

    model.float()

    device = next(model.parameters()).device  # get model device, PyTorch model

    # Configure
    model.eval()
    nc = 1  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    p, r, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap = [], []


    for _, (imgs_tensor, patches, targets, rotated_points) in enumerate(dataloader):
        imgs_tensor = imgs_tensor.cuda()
        patches = patches.cuda()
        targets = targets[:,-1].cuda()
        rotated_points = rotated_points.cuda()

        patches = upsample_patch(patches)
        imgs_tensor = apply_patch(imgs_tensor, patches, rotated_points, patch_ratio=1.0)
        
        feats = model.ead_stage_1(imgs_tensor)
        refined_feats = policy(feats)
        preds = model.ead_stage_2(refined_feats)
        imgs_tensor = imgs_tensor[:,-1].permute(0,3,1,2)/255
        # preds = model(imgs_tensor)
        post_process_pred(preds, imgs_tensor[0:1])

        shapes = [[[imgs_tensor[i].shape[1], imgs_tensor[i].shape[2]], [[1.0, 1.0], [0.0, 0.0]]] for i in range(imgs_tensor.shape[0])]

        nb, _, height, width = imgs_tensor.shape  # batch size, channels, height, width


        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

        preds = non_max_suppression(
            preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
        )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            shape = shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            # model 没有给出预测
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            predn = pred.clone().detach()
            scale_boxes(imgs_tensor[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(imgs_tensor[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)


    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    print("Instances: {}, P: {:.4f}, R: {:.4f}, mAP50: {:.4f}, mAP50-95: {:.4f}".format(nt.sum(), mp, mr, map50, map))
    
    return nt.sum(), mp, mr, map50, map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-am', '--attack_method', type=str, help='attack method')
    args = parser.parse_args()
    batch_size = 40
    device = torch.device('cuda:0')
    # modelTool.seed_everything()
    model = modelTool.get_det_model(pretrain_weights='checkpoints/yolo_carla.pt', freeze = 17, device=device)
    model.train()
    modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolo_carla.pt', detModel=model)

    max_steps = 4
    C,H,W = 64, 32, 56
    policy = Transformer(num_layers=2,
                        num_heads=8,
                        num_blocks=(max_steps*C),
                        residual_pdrop=0.1,
                        attention_pdrop=0.1,
                        embedding_pdrop=0.1,
                        embedding_dim=H*W,
                        vertical_scale=30,
                        horizontal_scale=60,
                        action_dim=2).cuda()
    policy.load_state_dict(torch.load('checkpoints/ead_offline.pt'))

    _, _, _, _, mAP = evaluation(batch_size=batch_size, model=model, policy=policy, max_steps=4, attack_method=args.attack_method)
    

    
    
    
