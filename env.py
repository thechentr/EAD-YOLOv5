import cv2
import gym
import numpy as np
import torch
import math
import torch.nn.functional as F
from gym import spaces
from gym.envs.registration import register
import weakref
from utils.post_process import post_process_pred

import utils.modelTool as modelTool
import carla
import pygame
import json
import time
from generate_test_datatset import ClientSideBoundingBoxes

VIEW_WIDTH = 256
VIEW_HEIGHT = 256
VIEW_FOV = 90

class EADEnv(gym.Env):
    def __init__(self, batch_size, max_step):
        super(EADEnv, self).__init__()
        veritcal_scale = 15 # TODO
        horizontal_scale = 60 # TODO
        
        device = torch.device('cuda:0')

        self.sensory = modelTool.get_det_model(pretrain_weights='checkpoints/yolo_carla.pt', freeze = 17, device=device)
        self.sensory.eval()
        modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolo_carla.pt', detModel=self.sensory)



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

        self.batch_size = batch_size
        self.max_step = max_step
        self.dist = torch.ones(batch_size).cuda()*1.3
        self.elev_centre = torch.ones(batch_size).cuda()*5 # TODO
        self.azim_centre = torch.ones(batch_size).cuda()*(-90) # TODO
        self.curr_step = 0

        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.clear_all()

        self.camera = None
        self.car = None

        self.image = None
        self.capture = True

        with open('dataset/state/test_state.json', 'r') as file:
            self.test_state = json.load(file)
            print(len(self.test_state))

    def render_carla(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            self.collect_data = array
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.02
        self.world.apply_settings(settings)

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        camera_bp.set_attribute('blur_amount', str(0.0))
        camera_bp.set_attribute('motion_blur_max_distortion', str(0.0))
        return camera_bp
    
    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def setup_car(self, idx):
        """
        Spawns actor-vehicle to be controled.
        """
        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[idx]
        transform = carla.Transform(carla.Location(x=self.test_state[idx]['loc'][0],
                                                   y=self.test_state[idx]['loc'][1],
                                                   z=self.test_state[idx]['loc'][2]), 
                                    carla.Rotation(pitch=self.test_state[idx]['rot'][0],
                                                   yaw=self.test_state[idx]['rot'][1],
                                                   roll=self.test_state[idx]['rot'][2]))
        self.car = self.world.spawn_actor(car_bp, transform)
        return car_bp.get_attribute('base_type')
    
    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        self.camera = self.world.spawn_actor(self.camera_blueprint(), carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.set_camera_relative_position(elev=-5, azim=-60, dist=6)

    def set_camera_relative_position(self, azim, elev, dist):
        """
        根据方位角、仰角和距离设置相机相对于其父对象的位置。
        参数:
        - camera: 要调整的相机对象。
        - azim: 方位角（度），定义相机围绕Z轴的旋转。
        - elev: 仰角（度），定义相机围绕Y轴的旋转。
        - dist: 从父对象到相机的距离。
        """
        azim = -azim
        # 将角度转换为弧度
        rad_azim = math.radians(azim)
        rad_elev = math.radians(elev)

        # 计算相机相对于父对象的位置
        x = dist * math.cos(rad_elev) * math.cos(rad_azim)
        y = dist * math.cos(rad_elev) * math.sin(rad_azim)
        z = dist * math.sin(rad_elev)

        # 创建相对位置的Transform对象
        relative_location = carla.Location(x=x, y=y, z=z)
        # 确定相机应该朝向父对象的中心，计算适当的旋转
        pitch = -math.degrees(rad_elev)
        yaw = -math.degrees(math.pi - rad_azim)
        relative_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0)
        relative_transform = carla.Transform(relative_location, relative_rotation)

        # 设置相机的新Transform
        self.camera.set_transform(relative_transform)

    def set_dist(self, car_type):
        if car_type == 'car':
            dist = 6
        elif car_type == 'van':
            dist = 8
        elif car_type == 'motorcycle':
            dist = 5
        elif car_type == 'truck':
            dist = 8
        elif car_type == 'Bus':
            dist = 12
        elif car_type == 'bicycle':
            dist = 5
        else:
            dist = 5

        return dist
    
    @torch.no_grad()
    def step(self, action):
        self.curr_step += 1

        elev = self.elev_centre + torch.tensor(action[:,0]).cuda() + 15
        azim = self.azim_centre + torch.tensor(action[:,1]).cuda()
        elev = torch.clamp(elev, self.elev_centre, self.elev_centre + 30)
        azim = torch.clamp(azim, self.azim_centre - 60, self.azim_centre + 60)
        print(f'elev: {elev[0].item()}, azim: {azim[0].item() + 90}')

        self.set_camera_relative_position(elev=elev, azim=azim, dist=self.dist)
        for i in range(5):
            self.capture = True
            self.world.tick()

        self.render_carla(self.display)

        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles, self.camera, 'vehicle')
        patch_bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles, self.camera, 'patch')
        bbox, rpoints = ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes, patch_bounding_boxes)
        self.annotations[:,self.curr_step:self.curr_step+1] = torch.tensor((0,0)+bbox).unsqueeze(0).unsqueeze(0).numpy()
        # print(self.annotations)


        cv2.imwrite(f'ppo_{self.curr_step}.png', self.collect_data)

        pygame.display.flip()
        pygame.event.pump()

        collect_data = torch.tensor(self.collect_data[:,:,::-1].copy()).unsqueeze(0).unsqueeze(0)

        feature = self.sensory.ead_stage_1(collect_data.cuda()) # [B, F]

        collect_data = collect_data[:,-1].permute(0,3,1,2)/255
        preds = self.sensory.ead_stage_2(feature[:,-1,:,:,:])
        post_process_pred(preds, collect_data[0:1])
        

        self.features[:,self.curr_step:self.curr_step+1] = feature.unsqueeze(1).cpu().numpy() # [B, S[curr_step], F] <- [B, 1, F]
        state = self.features.copy()
        reward = np.zeros(self.batch_size) # [B]
        done = (self.curr_step == self.max_step)
        print('self.curr_step',self.curr_step,'done',done)
        info = {'step':self.curr_step, 'annotations':self.annotations.copy()}
        return state, reward, done, info

    @torch.no_grad()
    def reset(self, car_idx):
        self.clear()
        self.car_idx = car_idx
        car_type = self.setup_car(car_idx)
        self.setup_camera()
        self.vehicles = self.world.get_actors().filter('vehicle.*')
        time.sleep(2.0)
        self.set_synchronous_mode(True)

        self.curr_step = 0
        self.features = np.zeros((self.batch_size, self.max_step+1, 64, 32, 56), dtype=np.float32) # [B, S, F]
        self.annotations = np.zeros((self.batch_size, self.max_step+1, 6), dtype=np.float32) # [B, S, 6]
        self.use_patch = torch.randint(low=0,high=100,size=(self.batch_size,))


        elev = self.elev_centre
        azim = self.azim_centre
        self.dist = self.set_dist(car_type)

        print(f'elev: {elev[0].item()}, azim: {azim[0].item() + 90}')


        self.set_camera_relative_position(elev=elev, azim=azim, dist=self.dist)
        for i in range(5):
            self.capture = True
            self.world.tick()

        self.render_carla(self.display)

        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles, self.camera, 'vehicle')
        patch_bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles, self.camera, 'patch')
        bbox, rpoints = ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes, patch_bounding_boxes)
        self.annotations[:,self.curr_step:self.curr_step+1] = torch.tensor((0,0)+bbox).unsqueeze(0).unsqueeze(0).numpy()
        # print(self.annotations)

        cv2.imwrite(f'ppo_{self.curr_step}.png', self.collect_data)

        pygame.display.flip()
        pygame.event.pump()

        collect_data = torch.tensor(self.collect_data[:,:,::-1].copy()).unsqueeze(0).unsqueeze(0) # [B, S, C, H, W]

        feature = self.sensory.ead_stage_1(collect_data.cuda()) # [B, S, FC, FH, FW]
        
        preds = self.sensory.ead_stage_2(feature[:,-1,:,:,:])
        collect_data = collect_data[:,-1].permute(0,3,1,2)/255
        post_process_pred(preds, collect_data[0:1])
        
        self.features[:,self.curr_step:self.curr_step+1] = feature.unsqueeze(1).cpu().numpy() # [B, S[0], FC, FH, FW] <- [B, 1, FC, FH, FW]
        
        state = self.features.copy()
        info = {'step':self.curr_step, 'annotations':self.annotations.copy()}
        return state, info
    
    def clear(self):
        self.set_synchronous_mode(False)
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
        if self.camera is not None and self.car.is_alive:
            self.car.destroy()

    def clear_all(self):
        self.set_synchronous_mode(False)
        self._clear_all_actors(['sensor.camera.rgb', 
                                'vehicle.*'])
        
    def close(self):
        self.clear_all()
        pygame.quit()

    def _clear_all_actors(self, actor_filters):                                 # 删除carla中的所有actor，各种传感器也是carla中的sub-actor
        """Clear specific actors."""
        for actor_filter in actor_filters:                                      # 对于每一类actor
            for actor in self.world.get_actors().filter(actor_filter):          # 找到carla世界中所有属于这类的actor
                if actor.is_alive:                                              # 如果actor实例处于激活状态
                    if actor.type_id == 'controller.ai.walker':                 # 如果是AI控制器
                        actor.stop()                                            # 先停止控制器
                    actor.destroy()                                             # 销毁actor
        

register(
    id='EADEnv-v0',
    entry_point='env:EADEnv',
)

# env = EADEnv(1,3)
# for car_idx in range(1):
#     env.reset(car_idx)
#     for i in range(3):
#         env.step(np.random.rand(1,2)*np.array([30,120])+np.array([5,-60]))
# env.close()
