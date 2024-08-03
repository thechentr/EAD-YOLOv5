import torch
class LocalGradientsSmoothing:
    def __init__(self, window_size: int,
                 overlap: int,
                 smoothing_factor: float,
                 threshold: float,):
        self.window_size = window_size
        self.overlap = overlap
        self.smoothing_factor = smoothing_factor
        self.threshold = threshold
        self.grad = Gradient()
        self.stride = self.window_size - self.overlap
        self.fold = torch.nn.functional.fold
        self.unfold = torch.nn.functional.unfold
        
    def normalized_grad(self, img: torch.Tensor) -> torch.Tensor:
        img_grad = self.grad(img)
        max_grad = torch.amax(img_grad, dim=(2, 3), keepdim=True)
        min_grad = torch.amin(img_grad, dim=(2, 3), keepdim=True)
        img_grad = (img_grad - min_grad) / (max_grad - min_grad + 1e-7)
        return img_grad

    def get_mask(self, img_t: torch.Tensor) -> torch.Tensor:
        grad = self.normalized_grad(img_t)
        grad_unfolded = self.unfold(grad, self.window_size, stride=self.stride)
        mask_unfolded = torch.mean(grad_unfolded, dim=1, keepdim=True) > self.threshold
        mask_unfolded = mask_unfolded.repeat(1, grad_unfolded.shape[1], 1)
        mask_unfolded = mask_unfolded.float()
        mask_folded = self.fold(mask_unfolded, grad.shape[2:], kernel_size=self.window_size, stride=self.stride)
        mask_folded = (mask_folded >= 1).float()
        grad *= mask_folded
        grad = torch.clamp(self.smoothing_factor * grad, 0, 1)
        return grad

    def __call__(self, img: torch.Tensor):
        img = img.permute(0,3,1,2)/255
        img_gray = torch.sum(img,dim=1)
        mask = self.get_mask(img_gray)
        mask = mask.repeat((1, 3, 1, 1))
        img = img * (1 - mask)
        img = img.permute(0,2,3,1)*255
        return img
    
    def to(self, device: torch.device):
        self.grad = self.grad.to(device)
        return self




class Gradient(torch.nn.Module):
    r'''
    Compute the first-order local gradient
    '''

    def __init__(self) -> None:
        super().__init__()

        self.d_x = torch.nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
        self.d_y = torch.nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)
        self.zero_pad_x = torch.nn.ZeroPad2d((0, 1, 0, 0))
        self.zero_pad_y = torch.nn.ZeroPad2d((0, 0, 0, 1))
        self.update_weight()

    def update_weight(self):
        first_order_diff = torch.FloatTensor([[1, -1],])
        kernel_dx = first_order_diff.unsqueeze(0).unsqueeze(0)
        kernel_dy = first_order_diff.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        self.d_x.weight = torch.nn.Parameter(kernel_dx).requires_grad_(False)
        self.d_y.weight = torch.nn.Parameter(kernel_dy).requires_grad_(False)

    def forward(self, img):
        batch_size = img.shape[0]
        img_aux = img.reshape(-1, img.shape[-2], img.shape[-1])
        img_aux.unsqueeze_(1)
        grad_x = self.d_x(img_aux)
        grad = self.zero_pad_x(grad_x).pow(2)
        grad_y = self.d_y(img_aux)
        grad += self.zero_pad_y(grad_y).pow(2)
        grad.sqrt_()
        grad.squeeze_(1)
        grad = grad.reshape(batch_size, -1, img_aux.shape[-2], img_aux.shape[-1])
        return grad