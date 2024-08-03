import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torchvision import transforms
import torch
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from utils.visualize import save_tensor_image
import math
import time


def affine(x, vgrid, device='cuda'):
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output

def warp(x, flo, device='cuda'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = flo.size()
    H_in, W_in = x.size()[-2:]
    vgrid = torch.rand((B, 2, H, W)).to(device)
    # mesh grid

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * flo[:, 0, :, :].clone() / max(W_in - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * flo[:, 1, :, :].clone() / max(H_in - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    return affine(x, vgrid)

def WarpPerspective(x, tmatrix, out_H=None, out_W=None, dstsize=None, device='cuda', inverse=False):
    '''
    formulation: http://www.cnblogs.com/zipeilu/p/6138423.html
    input:
        x(torch.tensor): NxCxHxW
        tmatrix(numpy.array or list): 3x3
    output:
        warp_res(torch.tensor): NxCxHxW
    '''

    assert (len(x.size()) == 4)

    if inverse:
        tmatrix = np.linalg.inv(tmatrix)
    H, W = x.size()[2:]
    if out_H is None and out_W is None:
        out_H, out_W = H, W
    if dstsize is not None:
        out_H, out_W = dstsize

    flow = torch.zeros(2, out_H, out_W).to(device)
    identity = torch.ones(out_H, out_W).to(device)
    xx = torch.arange(0, out_W).view(1, -1).repeat(out_H, 1).type_as(identity).to(device)
    yy = torch.arange(0, out_H).view(-1, 1).repeat(1, out_W).type_as(identity).to(device)
    _A = (tmatrix[1][1] - tmatrix[2][1] * yy)
    _B = (tmatrix[2][2] * xx - tmatrix[0][2])
    _C = (tmatrix[0][1] - tmatrix[2][1] * xx)
    _D = (tmatrix[2][2] * yy - tmatrix[1][2])
    _E = (tmatrix[0][0] - tmatrix[2][0] * xx)
    _F = (tmatrix[1][0] - tmatrix[2][0] * yy)
    xa = _A * _B - _C * _D
    xb = _A * _E - _C * _F
    ya = _F * _B - _E * _D
    yb = _F * _C - _E * _A
    flow[0] = xa / xb
    flow[1] = ya / yb
    flow = flow.view(1, 2, out_H, out_W).repeat(x.size(0), 1, 1, 1)
    return warp(x, flow, device=device)


class WarpFunction(Function):

    @staticmethod
    def forward(ctx, input, matrix, dstsize=None):
        ctx.save_for_backward(input, torch.from_numpy(matrix))
        return WarpPerspective(input, matrix, dstsize=dstsize)

    @staticmethod
    def backward(ctx, grad_output):
        input, matrix = ctx.saved_variables
        dstsize = input.size()[-2:]
        return WarpPerspective(grad_output, matrix.cpu().numpy(), dstsize=dstsize, inverse=True), None, None


def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1),
              shearing=(0, 0, 0)):
    """
    input:
        image(torch.Tensor): NxCxHxW
    output:
        transformed image(torch.Tensor): NxCxHxW
    """
    
    # get the values on each axis
    t_x, t_y, t_z = translation
    r_x, r_y, r_z = rotation
    sc_x, sc_y, sc_z = scaling
    sh_x, sh_y, sh_z = shearing
    
    # convert degree angles to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)
    theta_shx = np.deg2rad(sh_x)
    theta_shy = np.deg2rad(sh_y)
    theta_shz = np.deg2rad(sh_z)

    # convert rad to rad
    # theta_rx = (r_x)
    # theta_ry = (r_y)
    # theta_rz = (r_z)
    # theta_shx = (sh_x)
    # theta_shy = (sh_y)
    # theta_shz = (sh_z)
    
    # get the height and the width of the image
    h, w = image.shape[-2: ]
    # compute its diagonal
    # diag = (h ** 2 + w ** 2) ** 0.5
    
    # compute the focal length
    # f = diag * 5.625    # NOTE: magic number estimated by binary search
    f = h / (2 * np.tan(18.837 * np.pi / 360.0))
    
    # compute its diagonal
    # diag = (h ** 2 + w ** 2) ** 0.5
    # # compute the focal length
    # f = diag
    # if np.sin(theta_rz) != 0:
    #     f /= 2 * np.sin(theta_rz)


    # set the image from cartesian to projective dimension
    H_M = np.array([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0,      1],
                    [0, 0,      1]])
    # set the image projective to carrtesian dimension
    Hp_M = np.array([[f, 0, w / 2, 0],
                     [0, f, h / 2, 0],
                     [0, 0,     1, 0]])
    
    Identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    
    # adjust the translation on z
    t_z = (f - t_z) / sc_z ** 2
    # translation matrix to translate the image
    T_M = np.array([[1, 0, 0, t_x],
                    [0, 1, 0, t_y],
                    [0, 0, 1, t_z],
                    [0, 0, 0,  1]])

    # calculate cos and sin of angles
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
    # get the rotation matrix on x axis
    R_Mx = np.array([[1,      0,       0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx,  cos_rx, 0],
                     [0,      0,       0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [     0, 1,       0, 0],
                     [sin_ry, 0,  cos_ry, 0],
                     [     0, 0,       0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz,  cos_rz, 0, 0],
                     [     0,       0, 1, 0],
                     [     0,       0, 0, 1]])
    # compute the full rotation matrix
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

    # get the scaling matrix
    Sc_M = np.array([[sc_x,     0,    0, 0],
                     [   0,  sc_y,    0, 0],
                     [   0,     0, sc_z, 0],
                     [   0,     0,    0, 1]])

        # get the tan of angles
    tan_shx = np.tan(theta_shx)
    tan_shy = np.tan(theta_shy)
    tan_shz = np.tan(theta_shz)
    # get the shearing matrix on x axis
    Sh_Mx = np.array([[      1, 0, 0, 0],
                      [tan_shy, 1, 0, 0],
                      [tan_shz, 0, 1, 0],
                      [      0, 0, 0, 1]])
    # get the shearing matrix on y axis
    Sh_My = np.array([[1, tan_shx, 0, 0],
                      [0,       1, 0, 0],
                      [0, tan_shz, 1, 0],
                      [0,       0, 0, 1]])
    # get the shearing matrix on z axis
    Sh_Mz = np.array([[1, 0, tan_shx, 0],
                      [0, 1, tan_shy, 0],
                      [0, 0,       1, 0],
                      [0, 0,       0, 1]])
    # compute the full shearing matrix
    Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

    # compute the full transform matrix
    M = Identity
    M = np.dot(T_M,  M)
    M = np.dot(R_M,  M)
    M = np.dot(Sc_M, M)
    M = np.dot(Sh_M, M)
    M = np.dot(Hp_M, np.dot(M, H_M))

    # apply the transformation
    # image = cv2.warpPerspective(image, M, (w, h))
    image = WarpFunction.apply(image, M, (w, h))
    return image

def _create_camera_matrix(fov):
    # 将FOV从度转换为弧度
    fov_rad = math.radians(fov)

    # 计算焦距
    f_x = 1 / (torch.tan(torch.tensor(fov_rad / 2)) * 1.4142)  # sqrt(2) = 1.4142
    f_y = 1 / (torch.tan(torch.tensor(fov_rad / 2)) * 1.4142)

    # 主点坐标（单位化图像的中心）S
    c_x = 0.5
    c_y = 0.5

    # 构造内参矩阵
    K = torch.tensor([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    # 扩展为4维
    camera_matrix = torch.eye(4)
    camera_matrix[:3, :3] = K

    return camera_matrix

def world2img(p, matrix):
    """Project a 3D point in world coordinates to 2D image coordinates using a projection matrix."""
    p = torch.matmul(matrix, p)  # Multiply by camera intrinsics to compute point on the sensor
    p = p / p[2]  # Normalize its world coordinates
    return [float(p[0]), float(p[1])]

def _pad_with_black(input_tensor, target_size):
    """
    将输入的张量填充到指定的尺寸，并在四周填充黑色。
    
    参数：
    - input_tensor: 输入的张量，大小为 [N, C, H, W]
    - target_size: 目标尺寸，格式为 (target_height, target_width)
    
    返回值：
    - padded_tensor: 填充后的张量，大小为 [N, C, target_height, target_width]
    """
    # 获取输入张量的尺寸
    _, input_height, input_width = input_tensor.shape[-3:]
    # 获取目标尺寸的高度和宽度
    target_height, target_width = target_size
    
    # 计算垂直和水平方向上的填充量
    pad_vertical = max(target_height - input_height, 0)
    pad_horizontal = max(target_width - input_width, 0)
    
    # 计算上下左右填充的数量
    pad_top_bottom = pad_vertical // 2
    pad_left_right = pad_horizontal // 2
    
    # 使用 pad 函数在四周填充黑色
    padded_tensor = F.pad(input_tensor, (pad_left_right, pad_top_bottom), fill=0)
    
    return padded_tensor

def _pad_with_black_batch(input_tensor, target_size):
    """
    将输入的张量填充到指定的尺寸，并在四周填充黑色。
    
    参数：
    - input_tensor: 输入的张量，大小为 [N, C, H, W]
    - target_size: 目标尺寸，格式为 (target_height, target_width)
    
    返回值：
    - padded_tensor: 填充后的张量，大小为 [N, C, target_height, target_width]
    """
    # 获取输入张量的尺寸
    batch_size, channels, input_height, input_width = input_tensor.shape
    # 获取目标尺寸的高度和宽度
    target_height, target_width = target_size
    
    # 计算垂直和水平方向上的填充量
    pad_vertical = max(target_height - input_height, 0)
    pad_horizontal = max(target_width - input_width, 0)
    
    # 计算上下左右填充的数量
    pad_top = pad_vertical // 2
    pad_bottom = pad_vertical - pad_top
    pad_left = pad_horizontal // 2
    pad_right = pad_horizontal - pad_left

    # 使用 pad 函数在四周填充黑色
    padded_tensor = F.pad(input_tensor, (pad_left, pad_bottom, pad_right, pad_top), fill=0)
    
    return padded_tensor


def _rotate_about_axis(points, angle_rad, axis):
    """
    Rotate a 3D point about a specified axis passing through the given center.

    return:
    - points
    """
    angle_rad = angle_rad + 1e-5  # 防止转换后无法进行透视变换
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    
    # Rotation matrices around x, y, z axes
    if axis == 'x':
        R = torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]], dtype=torch.float)
    elif axis == 'y':
        R = torch.tensor([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]], dtype=torch.float)
    elif axis == 'z':
        R = torch.tensor([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # translation matrix
    bais = torch.tensor([0., 0., 4., 0.])
    center = sum(points) / len(points) + bais
    T_neg = torch.eye(4)
    T_neg[:3, 3] = -center[:3]
    T_pos = torch.eye(4)
    T_pos[:3, 3] = center[:3]

    # Combine translation and rotation into a single transformation matrix
    K = torch.matmul(T_pos, torch.matmul(R, T_neg))
    return [K @ p for p in points]

def rotate_by_angle(img_tensor, horizontal, vertical, canvas_size=(128, 128), interpolation=transforms.InterpolationMode.NEAREST):
    """
    horizontal, vertical unit degree, 'image' is  patch
    """
    device = img_tensor.device
    mask = torch.ones_like(img_tensor, device=device)
    

    img_tensor = _pad_with_black(img_tensor, canvas_size)
    mask = _pad_with_black(mask, canvas_size)

    canvas_h, canvas_w = canvas_size

    orig_points = [[0, 0], [0, canvas_w], [canvas_h, canvas_w], [canvas_h, 0]]
    rotated_points = calculate_rotated_points(canvas_size, horizontal, vertical)

    # print(orig_points, rotated_points)
    transformed_img_tensor = F.perspective(img_tensor, orig_points, rotated_points, interpolation=interpolation, fill=0)
    transformed_mask = F.perspective(mask, orig_points, rotated_points, interpolation=interpolation, fill=0)


    return transformed_img_tensor, transformed_mask

def calculate_rotated_points(image_size, horizontal, vertical, radius=300, FOV=60):

    w, h = image_size

    points_world = [
        torch.tensor([-w/2, -h/2, radius, 1.]),
        torch.tensor([-w/2, h/2, radius, 1.]),
        torch.tensor([w/2, h/2, radius, 1.]),
        torch.tensor([w/2, -h/2, radius, 1.])
    ]
    
    camera_matrix = _create_camera_matrix(FOV)
    rotated_points_world = _rotate_about_axis(points_world, horizontal, 'y')  # rotate + project + unnormal
    rotated_points_world = _rotate_about_axis(rotated_points_world, vertical, 'x')

    antinorm = lambda x: (x[0]*w, x[1]*h)
    rotated_points = [antinorm(world2img(p, camera_matrix)) for p in rotated_points_world]
    return rotated_points
    

def rotate_by_points(img_tensor, rotated_points, canvas_size=(128, 128), interpolation=transforms.InterpolationMode.NEAREST):
    """
    use rotated_points
    """
    device = img_tensor.device
    mask = torch.ones_like(img_tensor, device=device)
    

    img_tensor = _pad_with_black(img_tensor, canvas_size)
    mask = _pad_with_black(mask, canvas_size)

    canvas_h, canvas_w = canvas_size

    points_orig = [[0, 0], [0, canvas_w], [canvas_h, canvas_w], [canvas_h, 0]]

    # print(points_orig, rotated_points)
    transformed_img_tensor = F.perspective(img_tensor, points_orig, rotated_points, interpolation=interpolation, fill=0)
    transformed_mask = F.perspective(mask, points_orig, rotated_points, interpolation=interpolation, fill=0)


    return transformed_img_tensor, transformed_mask


def rotate_by_points_batch(img_tensors, rotated_points_batch, canvas_size=(128, 128), interpolation=transforms.InterpolationMode.NEAREST):
    """
    Rotate a batch of image tensors by transforming their corners to new rotated points.
    
    Args:
        img_tensors (torch.Tensor): Batch of image tensors to be transformed, shape (B, C, H, W).
        rotated_points_batch (list of list of list of int): Batch of target points to map the original image corners to, shape (B, 4, 2).
        canvas_size (tuple): The size of the canvas to pad the image to.
        interpolation (transforms.InterpolationMode): The interpolation mode for the transformation.

    Returns:
        torch.Tensor: Batch of transformed image tensors.
        torch.Tensor: Batch of transformed mask tensors.
    """
    device = img_tensors.device
    batch_size = img_tensors.shape[0]
    canvas_h, canvas_w = canvas_size

    points_orig = torch.tensor([[0, 0], [0, canvas_w], [canvas_h, canvas_w], [canvas_h, 0]], dtype=torch.float32)
    
    img_tensors_padded = _pad_with_black_batch(img_tensors, canvas_size)
    masks = _pad_with_black_batch(torch.ones_like(img_tensors, device=device), canvas_size)

    rotated_points_batch = torch.tensor(rotated_points_batch, dtype=torch.float32, device=device)

    transformed_img_tensors = []
    transformed_masks = []

    for i in range(batch_size):        
        # 进行透视变换
        transformed_img = F.perspective(img_tensors_padded[i], points_orig, rotated_points_batch[i], interpolation=interpolation, fill=0)
        transformed_mask = F.perspective(masks[i], points_orig, rotated_points_batch[i], interpolation=interpolation, fill=0)
        
        # 添加到列表中
        transformed_img_tensors.append(transformed_img)
        transformed_masks.append(transformed_mask)

    # 将列表转换为张量
    transformed_img_tensors = torch.stack(transformed_img_tensors)
    transformed_masks = torch.stack(transformed_masks)

    return transformed_img_tensors, transformed_masks


if __name__ == '__main__':
    xs = torch.ones(1, 3, 128, 128).cuda()
    transform(xs, rotation=(.0, .0, .0))


    tensor_image = torch.rand((3, 32, 64))

    i = 0
    while True:
        i += 10
        transformed_image, transformed_mask = rotate_by_angle(tensor_image, horizontal=i, vertical=0, canvas_size=(128, 128))
        save_tensor_image(transformed_image, 'debug/transformed_image.png')
        save_tensor_image(transformed_mask, 'debug/transformed_mask.png')
        time.sleep(0.1)
