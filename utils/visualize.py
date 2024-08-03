from PIL import Image, ImageDraw, ImageFont
import torch

import math

def save_tensor_image_all(image_tensor, file_path):
    """
    支持image_tensor的值范围包括[0, 1]、[-1, 1]和[0, 255]。

    参数:
        image_tensor (torch.Tensor): 待保存的图像Tensor，形状为(C, H, W)或(N, C, H, W)。
        file_path (str): 图像保存的路径。
    """
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        # 如果是四维Tensor且批量大小为1，去掉批量维度
        image_tensor = image_tensor.squeeze(0)
        
    if image_tensor.ndim != 3 or image_tensor.shape[0] not in [1, 3]:
        raise ValueError("图像Tensor的形状应为(C, H, W)，其中C为1或3")

    # 判断Tensor的值范围并进行相应的调整
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    if min_val >= 0 and max_val <= 1:
        # 值范围[0, 1]
        re = '[0, 1]'
        image_tensor = (image_tensor * 255).clamp(0, 255).byte()
    elif min_val >= -1 and max_val <= 1:
        # 值范围[-1, 1]
        re = '[-1, 1]'
        image_tensor = ((image_tensor + 1) / 2 * 255).clamp(0, 255).byte()
    elif min_val >= 0 and max_val > 1:
        # 值范围[0, 255]，假设图像数据已经是uint8类型
        re = '[0, 255]'
        image_tensor = image_tensor.clamp(0, 255).byte()
    else:
        raise ValueError("不支持的图像Tensor值范围")
    # 转换为PIL Image并保存
    if image_tensor.shape[0] == 3:
        # 对于3通道图像，调整通道顺序为(H, W, C)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image_np, 'RGB')
    elif image_tensor.shape[0] == 1:
        # 对于单通道图像，去掉通道维度并转换为灰度图像
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image = Image.fromarray(image_np, 'L')
        
    # 保存图像
    image.save(file_path)
    return re

def save_tensor_image(image_tensor, file_path):
    """
    支持image_tensor的值范围包括[0, 1]

    参数:
        image_tensor (torch.Tensor): 待保存的图像Tensor，形状为(C, H, W)或(N, C, H, W)。
        file_path (str): 图像保存的路径。
    """
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        # 如果是四维Tensor且批量大小为1，去掉批量维度
        image_tensor = image_tensor.squeeze(0)
        
    if image_tensor.ndim != 3 or image_tensor.shape[0] not in [1, 3]:
        raise ValueError("图像Tensor的形状应为(C, H, W)，其中C为1或3")

    # 判断Tensor的值范围并进行相应的调整
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    if min_val >= 0 and max_val <= 1:
        # 值范围[0, 1]
        image_tensor = (image_tensor * 255).clamp(0, 255).byte()
    else:
        err = f"不支持的图像Tensor值范围 {min_val}-{max_val}"
        raise ValueError(err)
    # 转换为PIL Image并保存
    if image_tensor.shape[0] == 3:
        # 对于3通道图像，调整通道顺序为(H, W, C)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image_np, 'RGB')
    elif image_tensor.shape[0] == 1:
        # 对于单通道图像，去掉通道维度并转换为灰度图像
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image = Image.fromarray(image_np, 'L')
        
    # 保存图像
    image.save(file_path)




from PIL import Image, ImageDraw

def draw_boxes_on_image(tensor, boxes, file_path):
    """
    在图像上绘制YOLO格式的锚框并保存。
    
    参数:
    - tensor: 图像的Tensor，形状为[3, 640, 640]，值在[0, 255]范围内。
    - boxes: 锚框列表，每个锚框格式为[class, x_center, y_center, width, height]。
    - file_path: 保存图片的路径。
    """
    # 调整通道顺序为H×W×C，并转换为PIL Image
    tensor = tensor.permute(1, 2, 0).cpu().detach()
    image = Image.fromarray(tensor.numpy().astype('uint8'))

    # 获取图像尺寸
    img_width, img_height = image.size
    
    # 创建一个draw对象
    draw = ImageDraw.Draw(image)
    

    # 反归一化坐标
    # print(boxes)
    x_center, y_center, width, height = boxes[1:]  # 忽略类别信息
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
        
    # 计算锚框的左上角和右下角坐标
    left = x_center - width / 2
    top = y_center - height / 2
    right = x_center + width / 2
    bottom = y_center + height / 2
        
    # 绘制矩形
    draw.rectangle([left, top, right, bottom], outline="red", width=2)
    
    # 保存图像
    image.save(file_path)


def draw_boxes_on_batch_images(tensors, boxes, file_path_prefix):
    """
    在批量图像上绘制YOLO格式的预测框并保存。

    参数:
    - tensors: 图像的Tensor批量，形状为[N, 3, H, W]，值在[0, 255]范围内。
    - boxes: 模型输出的预测框，每个预测框格式为[batch_index, class, conf, x_center, y_center, width, height]。
    - file_path_prefix: 保存图片的路径前缀。
    """
    tensors = tensors.detach().clone()
    boxes = boxes.detach().clone()
    # 遍历批量中的每张图像
    for i, tensor in enumerate(tensors):
        # 调整通道顺序为H×W×C，并转换为PIL Image
        tensor = tensor.permute(1, 2, 0).cpu().detach()
        image = Image.fromarray(tensor.numpy().astype('uint8'))
        draw = ImageDraw.Draw(image)

        # 获取当前图像的预测框
        img_boxes = [box for box in boxes if box[0] == i]  # 筛选属于当前图像的框

        # 绘制每个预测框
        for box in img_boxes:
            _, _, conf, x_center, y_center, width, height = box
            img_width, img_height = image.size

            # 反归一化坐标
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # 计算锚框的左上角和右下角坐标
            # 计算锚框的左上角和右下角坐标
            left = x_center - width / 2
            right = x_center + width / 2
            top = y_center - height / 2
            bottom = y_center + height / 2

            # 确保top小于等于bottom
            if top > bottom:
                top, bottom = bottom, top
            # 确保top小于等于bottom
            if left > right:
                left, right = right, left
            # 绘制矩形
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

            # 在矩形上方绘制置信度
            text = f"Conf: {conf:.2f}"
            draw.text((left, top - 15), text, fill="red")

        # 保存图像
        image.save(f"./debug/{file_path_prefix}_img_{i}.png")
        print(f'img save to ./debug/{file_path_prefix}_img_{i}.png')


def draw_boxes_on_grid_image(tensors, boxes):
    """
    在网格形式的单张图片上绘制整个批量的YOLO格式的预测框并保存。

    参数:
    - tensors: 图像的Tensor批量，形状为[N, 3, H, W]，值在[0, 255]范围内。
    - boxes: 模型输出的预测框，每个预测框格式为[batch_index, class, conf, x_center, y_center, width, height]。
    """
    N, C, H, W = tensors.shape
    # 计算网格的尺寸
    grid_size = math.ceil(math.sqrt(N))  # 网格的行列数
    total_width = W * grid_size
    total_height = H * grid_size
    total_image = Image.new('RGB', (total_width, total_height))

    # 遍历批量中的每张图像
    for i, tensor in enumerate(tensors):
        # 计算当前图像在网格中的位置
        row = i // grid_size
        col = i % grid_size

        # 调整通道顺序为H×W×C，并转换为PIL Image
        tensor = tensor.permute(1, 2, 0).cpu().detach()
        image = Image.fromarray(tensor.numpy().astype('uint8'))
        draw = ImageDraw.Draw(image)

        # 获取当前图像的预测框
        img_boxes = [box for box in boxes if box[0] == i]  # 筛选属于当前图像的框

        # 绘制每个预测框
        for box in img_boxes:
            if len(box) == 7:
                _, _, conf, x_center, y_center, width, height = box
            else:  # groundtruth 没有置信度
                _, _, x_center, y_center, width, height = box
                conf = 1.0
            img_width, img_height = image.size

            # 反归一化坐标
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # 计算锚框的左上角和右下角坐标
            left = x_center - width / 2
            right = x_center + width / 2
            top = y_center - height / 2
            bottom = y_center + height / 2

            # 绘制矩形
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

            # 在矩形上方绘制置信度
            text = f"置信度: {conf:.2f}"
            font = ImageFont.truetype('NotoSansCJK-Regular.ttc')  # Specify the path to your font file here

            draw.text((left, top - 15), text, "red", font)

        # 将每张处理后的图像拼接到总图片上
        total_image.paste(image, (col * W, row * H))

    # 保存图像
    total_image.save('test.png')
