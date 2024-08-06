from utils.general import non_max_suppression
from utils.visualize import *  # 用于可视化的工具

def post_process_xlw(im, pred):
    """ post process for model predictions """
    label=[]

    for i, det in enumerate(pred):  # per image

        if len(det):

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                x_mid = x1 + (x2 - x1) / 2
                y_mid = y1 + (y2 - y1) / 2
                w = x2 - x1
                h = y2 - y1
                c = int(cls)  # integer class
                label.append(torch.tensor([i, c, conf, x_mid / im.shape[-1], y_mid / im.shape[-2], w / im.shape[-1], h / im.shape[-2]]).view(1, -1))

    if len(label) == 0:
        # 如果为空，可以返回一个空Tensor，或者根据需要进行其他处理
        return torch.tensor([[0, 0, 0, 0, 0, 0, 0]])  # 返回一个空Tensor作为示例
    else:
        # 如果不为空，则正常合并Tensor
        res = torch.cat(label, dim=0).float()
        return res

@torch.no_grad()
def pred_image(model, im0, device='cuda:0', drawAs=None, policy_features=None):
    """
    use model to predict im0, model will be use and change to eval

    parameter:
        model (torch.nn.Module): 预训练的模型
        im0 (torch.Tensor): 待预测的图像[0, 1]
        device (str)
        draw (bool): 是否在图像上绘制检测到的边界框

    return:
        boxes (list): 检测到的边界框列表，每个边界框的格式为[imgIndex, class, x, y, w, h]。

    """
    modelIsTraining = model.training
    model.eval()

    im0_copy = im0.detach().clone().to(device).float()

    if policy_features is None:
        pred, _ = model(im0_copy, augment=False)
    else:
        pred, _ = model(im0_copy, augment=False, policy_features=policy_features)

    if modelIsTraining:
        model.train()

    return post_process_pred(pred, im0_copy, drawAs)


def post_process_pred(pred, img, conf_thres=0.25):
    """
    process raw pred to boxes
    """
    # Checks
    if isinstance(pred, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        pred = pred[0]  # select only inference output
        if isinstance(pred, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            pred = pred[0]  # select only inference output
    device = pred.device
    im0_copy = img.clone().detach()

    pred = non_max_suppression(pred, 
        conf_thres=conf_thres, iou_thres=0.6, 
        classes=None, agnostic=False, 
        max_det=1000)
    boxes = post_process_xlw(im0_copy, pred)  #（imgIndex, cls, conf， x， , y, w, h）
    draw_boxes_on_grid_image(im0_copy*255, boxes)

    if boxes.shape[0] > 0:
        ground_truth_label = torch.cat((boxes[:, :2], boxes[:, 3:]), dim=1)  #（imgIndex, cls, x， , y, w, h
    else:
        ground_truth_label = boxes
    
    # Free up memory
    del pred, im0_copy  # Free up memory from predictions
    return ground_truth_label.to(device)