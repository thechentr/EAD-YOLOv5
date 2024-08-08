import gym
from EG3Drender.render import EG3DRender
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch  
import numpy as np  
from PIL import Image  
import time
import random
import torchvision.transforms.functional as TF
import utils.modelTool as modelTool
from gym import spaces
import os
import pygame
from utils.post_process import post_process_pred, draw_boxes_on_grid_image
from gym.envs.registration import register

VIEW_WIDTH = 256
VIEW_HEIGHT = 256
VIEW_FOV = 90

def tensor_to_image(tensor, file_path):  
    """  
    Converts a tensor to an image and saves it as a PNG file.  

    Args:  
        tensor (torch.Tensor): [3, w, h], [0, 1]
        file_path (str): The file path to save the PNG image.  
    """  
    tensor = tensor.cpu().detach().numpy()  
    tensor = np.transpose(tensor, (1, 2, 0))  
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255  
    tensor = tensor.astype(np.uint8)  
    image = Image.fromarray(tensor)  
    image.save(file_path)  

def opoints(img):
    return [[0, 0], 
            [0, img.shape[1]], 
            [img.shape[2], img.shape[1]], 
            [img.shape[1], 0]]

class EG3DEnv(gym.Env):
    def __init__(self, batch_size, max_step, device=torch.device('cuda')):
        super(EG3DEnv, self).__init__()
        veritcal_scale = 15 # TODO
        horizontal_scale = 60 # TODO

        self.batch_size = batch_size
        self.max_step = max_step
        self.device = device


        self.sensory = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
        self.sensory.eval()
        modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolov5_2000.pt', detModel=self.sensory)

        self.action_space = spaces.Box(
            low=np.array([[-horizontal_scale, -veritcal_scale]]*batch_size), 
            high=np.array([[horizontal_scale, veritcal_scale]]*batch_size), 
            shape=(batch_size, 2),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(batch_size, max_step + 1, 64, 32, 56), 
            dtype=np.float32)
        
        
        self.dist = torch.ones(batch_size).cuda()*1.3
        self.elev_centre = torch.ones(batch_size).cuda()*5 # TODO
        self.azim_centre = torch.ones(batch_size).cuda()*(-90) # TODO
        self.curr_step = 0
            
        self.client = EG3DRender(device=device)
        pygame.init()
        self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("EG3DEnv Visualization")

        self.patch_tensor = torch.rand((batch_size, 3, 256, 256), device=device)

    @torch.no_grad()
    def step(self, actions):
        self.curr_step += 1

        elev = self.elev_centre + torch.tensor(actions[:,0]).cuda() + 15
        azim = self.azim_centre + torch.tensor(actions[:,1]).cuda()
        elev = torch.clamp(elev, self.elev_centre, self.elev_centre + 30)
        azim = torch.clamp(azim, self.azim_centre - 60, self.azim_centre + 60)
        # print(f'elev: {elev[0].item()}, azim: {azim[0].item() + 90}')

        imgs_tensor, rpoints = self.client.step(actions)
        annotation = self._annotate(imgs_tensor)

        for i in range(imgs_tensor.shape[0]):
            patch = TF.perspective(self.patch_tensor[i], opoints(self.patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
            imgs_tensor[i] = torch.where(patch.mean(0) == -1, imgs_tensor[i], patch)

        
        # process imgs
        seqimgs_tensor = imgs_tensor.permute(0, 2, 3, 1).unsqueeze(1) * 255
        feature = self.sensory.ead_stage_1(seqimgs_tensor)
        preds = self.sensory.ead_stage_2(feature[:,-1,:,:,:])

        post_process_pred(preds, imgs_tensor)
        # draw_boxes_on_grid_image(imgs_tensor*255, annotation)
        
        
        
        # draw the first img on pygame box
        self.render(imgs_tensor[0])
        pygame.display.flip()
        pygame.event.pump()
       

        self.features[:,self.curr_step:self.curr_step+1] = feature.cpu().numpy() # [B, S[curr_step], F] <- [B, 1, F]
        self.annotations[:,self.curr_step:self.curr_step+1] = annotation.unsqueeze(1).cpu().numpy()
        
        state = self.features
        reward = np.zeros(self.batch_size) # [B]
        done = (self.curr_step >= self.max_step)
        info = {'step':self.curr_step, 'annotations':self.annotations}
        return state, reward, done, info
                           
    @torch.no_grad()
    def reset(self, car_idx):
        self.curr_step = -1
        self.features = np.zeros((self.batch_size, self.max_step+1, 64, 32, 32), dtype=np.float32) # [B, S, F]
        self.annotations = np.zeros((self.batch_size, self.max_step+1, 6), dtype=np.float32) # [B, S, 6]

        init_state = torch.zeros((self.batch_size, 2), dtype=torch.float32, requires_grad=True, device=self.device)
        img_tensor, rpoints = self.client.reset([car_idx], init_state)

        state, reward, done, info = self.step(init_state)
        return state, info
    
    def render(self, img_tensor):
        image = img_tensor.mul(255).byte().cpu().numpy()  
        
        

        pil_image = Image.fromarray(np.transpose(image, (1, 2, 0)))  
        image_path = os.path.join(f'pygame_image.png')  
        pil_image.save(image_path)

        image = np.transpose(image, (2, 1, 0))
        surface = pygame.surfarray.make_surface(image)  
        self.display.blit(surface, (0, 0))  
        pygame.display.flip()

    @torch.no_grad
    def _annotate(self, imgs_tensor):
        label = []
        for i in range(imgs_tensor.shape[0]):
            box = [0, *self._calculate_box(imgs_tensor[i].permute(1,2,0))]
            if box[-1] + box[-2] < 1.8:  # 过滤掉没有物体的box
                label.append([i, *box])
        label = torch.tensor(label, dtype=torch.float32, device=self.device)

        return label

    @torch.no_grad
    def _calculate_box(self, img_tensor):
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
    
register(
    id='EG3DEnv-v0',
    entry_point='EG3DEnv:EG3DEnv',
)
        

if __name__ == '__main__':
    
    device = torch.device(f'cuda:0')

    env = EG3DEnv(4, 4, device)
    env.reset()
   
    try:  
        while True:  
            actions = torch.randn((env.batch_size, 2)).to(device) * 20  
            state, reward, done, info = env.step(actions)  
            print(info['step'])
            if done:  
                print('done')
                time.sleep(0.5)  
                env.reset()  
                
            time.sleep(0.3)  
    except KeyboardInterrupt:  
        pygame.quit()