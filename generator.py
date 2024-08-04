import torch
import torch.nn.functional as F

# 计算颜色概率映射图
def prob_fix_color(original_circles, coordinates, colors, fig_size_h, fig_size_w,blur=1):
    # original_circle 控制点信息
    # coordinate 全套衣服的坐标（经过长宽等比缩小后的图），这样一个像素点代表多个像素点
    # colors预设的颜色
    # fig_size_h, fig_size_w  mesh渲染图案的长宽大小
    assert original_circles.shape[0] == colors.shape[0]
    coordinates = coordinates.expand(original_circles.shape[1],-1,-1,-1).permute(1,2,0,3)
    # circles = original_circles * fig_size_h
    circle0 = original_circles[...,0]*fig_size_h   # 改成坐标信息了？？？？
    circle1 = original_circles[...,1]*fig_size_w   # 改成坐标？？？
    circles = torch.stack([circle0,circle1],dim=-1)
    dist_sum = torch.zeros([colors.shape[0],fig_size_h,fig_size_w]).to(coordinates.device)
    for color_idx in range(colors.shape[0]):
        dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
        # dist = torch.norm(coordinates-circles[color_idx,:,:2],dim=-1)
        # dist = dist / (circles[color_idx,:,2]+1)
        dist_sum[color_idx] = torch.exp(-dist/blur).sum(dim=-1)
        # print(dist_sum[color_idx])
    # print(dist_sum[0])
    dist_sum = dist_sum/(dist_sum.sum(dim=0)+1e-8)
    return dist_sum


# 根据概率值大小计算颜色
def gumbel_color_fix_seed(prob_map, seed, color, tau=0.3, type='gumbel'):
    # print(prob_map.shape, seed.shape, color.shape)
    if type == 'gumbel':
        color_map = F.softmax((torch.log(prob_map) + seed)/tau, dim=-1)
    elif type == 'determinate':
        color_ind = (torch.log(prob_map) + seed).max(-1)[1]
        color_map = F.one_hot(color_ind, prob_map.shape[-1]).to(prob_map)
    else:
        raise ValueError
    tex = torch.matmul(color_map, color).unsqueeze(0)
    return tex


# 计算维诺图控制点之间的距离损失，扩大子区域空间
def ctrl_loss(circles, fig_h, fig_w, sigma=40):
    circles = circles.repeat(circles.shape[1],1,1,1).permute(1,0,2,3)
    diff = circles - circles.permute(0,2,1,3)
    diff_ell2 = (diff[...,0] * diff[...,0]*fig_h*fig_h + diff[...,1] * diff[..., 1]*fig_w*fig_w)
    loss_c = torch.exp(-diff_ell2/(sigma**2)).mean() - 1/circles.shape[1]
    return loss_c

