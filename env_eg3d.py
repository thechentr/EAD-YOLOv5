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

import pygame


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
    def __init__(self, batch_size, max_step, device):
        super(EG3DEnv, self).__init__()
        veritcal_scale = 15 # TODO
        horizontal_scale = 60 # TODO

        self.batch_size = batch_size
        self.max_step = max_step
        self.device = device


        # self.sensory = modelTool.get_det_model(pretrain_weights='checkpoints/yolov5n.pt', freeze = 17, device=device)
        # self.sensory.eval()
        # modelTool.transfer_paramaters(pretrain_weights='checkpoints/freeze17_7000_4step_usap_501.pt', detModel=self.sensory)

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

        img_tensor, rpoints = self.client.step(actions)

        for i in range(img_tensor.shape[0]):
            patch = TF.perspective(self.patch_tensor[i], opoints(self.patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
            img_tensor[i] = torch.where(patch.mean(0) == -1, img_tensor[i], patch)

        self.render(img_tensor[0])


        pygame.display.flip()
        pygame.event.pump()

        self.annotations = torch.tensor([1, 2])  # TODO
        self.features = torch.tensor([1, 2])  # TODO
        # feature = self.sensory.ead_stage_1(img_tensor)

        # collect_data = collect_data[:,-1].permute(0,3,1,2)/255
        # preds = self.sensory.ead_stage_2(feature[:,-1,:,:,:])
        # post_process_pred(preds, collect_data[0:1])
        

        # self.features[:,self.curr_step:self.curr_step+1] = feature.unsqueeze(1).cpu().numpy() # [B, S[curr_step], F] <- [B, 1, F]
        state = self.features
        reward = np.zeros(self.batch_size) # [B]
        done = (self.curr_step >= self.max_step)
        info = {'step':self.curr_step, 'annotations':self.annotations}
        return state, reward, done, info
                           
    @torch.no_grad()
    def reset(self):
        self.curr_step = 0
        seed = list(range(10000, 13400, 17))
        random.shuffle(seed)
        persperctive = torch.zeros((self.batch_size, 2), dtype=torch.float32, requires_grad=True, device=self.device)
        img_tensor, rpoints = self.client.reset(seed[:self.batch_size], persperctive)

        for i in range(img_tensor.shape[0]):
            patch = TF.perspective(self.patch_tensor[i], opoints(self.patch_tensor[i]), rpoints[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
            img_tensor[i] = torch.where(patch.mean(0) == -1, img_tensor[i], patch)

        self.render(img_tensor[0])

        return img_tensor
    
    def render(self, img_tensor):
        image = img_tensor.mul(255).byte().cpu().numpy()  
        image = np.transpose(image, (2, 1, 0))
        surface = pygame.surfarray.make_surface(image)  
        self.display.blit(surface, (0, 0))  
        pygame.display.flip()
        

# env 需要返回features，内部需要进行yolo推理并计算reward
if __name__ == '__main__':
    
    device = torch.device(f'cuda:0')

    env = EG3DEnv(4, 4, device)
    env.reset()
   
    try:  
        while True:  
            actions = torch.randn((env.batch_size, 2)).to(device) * 20  
            state, reward, done, info = env.step(actions)  
            if done:  
                print('done')
                time.sleep(0.5)  
                env.reset()  
                
            time.sleep(0.3)  
    except KeyboardInterrupt:  
        pygame.quit()