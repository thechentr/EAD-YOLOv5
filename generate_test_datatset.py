import carla
import math
import weakref
import random
import cv2
import time
import os
import pygame
import numpy as np
import json
from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w


VIEW_WIDTH = 256
VIEW_HEIGHT = 256
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)
PATCH_COLOR = (64, 248, 24)

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera, bbx_type):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera, bbx_type) for vehicle in vehicles]
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, patch_bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """
        # print(bounding_boxes)
        # print(patch_bounding_boxes)
        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            x_min = VIEW_WIDTH
            x_max = 0
            y_min = VIEW_HEIGHT
            y_max = 0
            for point in points:
                if point[0] < x_min:
                    x_min = point[0]
                if point[0] > x_max:
                    x_max = point[0]
                if point[1] < y_min:
                    y_min = point[1]
                if point[1] > y_max:
                    y_max = point[1]

            pygame.draw.line(bb_surface, BB_COLOR, (x_min, y_min), (x_min, y_max))
            pygame.draw.line(bb_surface, BB_COLOR, (x_min, y_max), (x_max, y_max))
            pygame.draw.line(bb_surface, BB_COLOR, (x_max, y_max), (x_max, y_min))
            pygame.draw.line(bb_surface, BB_COLOR, (x_max, y_min), (x_min, y_min))

        for bbox in patch_bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            pygame.draw.line(bb_surface, PATCH_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, PATCH_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, PATCH_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, PATCH_COLOR, points[1], points[5])

            
        display.blit(bb_surface, (0, 0))
        width = (x_max - x_min)/VIEW_WIDTH
        height = (y_max - y_min)/VIEW_HEIGHT
        x = (x_max + x_min)/2/VIEW_WIDTH
        y = (y_max + y_min)/2/VIEW_HEIGHT
        return (x, y, width, height), (points[5], points[1], points[0], points[4])

    @staticmethod
    def get_bounding_box(vehicle, camera, bbx_type):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        if bbx_type == 'vehicle':
            bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        elif bbx_type == 'patch':
            bb_cords = ClientSideBoundingBoxes._create_patch_bb_points(vehicle)
        else:
            raise NotImplementedError
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords
    
    def _create_patch_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        extent_z = cords[4, 2]
        
        cords[:,0]*=0.5
        cords[:,2]*=0.5
        cords[:,2]-=extent_z*0.25
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True

        with open('dataset/state/test_state.json', 'r') as file:
            self.test_state = json.load(file)
            print(len(self.test_state))

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

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.02
        self.world.apply_settings(settings)

    def setup_car(self, idx):
        """
        Spawns actor-vehicle to be controled.
        """
        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[idx]
        # transform = random.choice(self.world.get_map().get_spawn_points())
        # with open('dataset/state/test_state.json', 'r') as file:
        #     states = json.load(file)
        # states.append({'loc':[transform.location.x, transform.location.y, transform.location.z],
        #                'rot':[transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],})
        # with open('dataset/state/test_state.json', 'w') as file:
        #     json.dump(states, file, indent=4)
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

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

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

    def render(self, display):
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

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)



            for car_idx in range(0, len(self.world.get_blueprint_library().filter('vehicle.*'))):
                car_path = f'dataset/test/{str(car_idx).zfill(2)}'
                if not os.path.exists(car_path):
                    os.mkdir(car_path)
                car_type = self.setup_car(car_idx)
                print(car_idx, car_type)
                self.setup_camera()
                vehicles = self.world.get_actors().filter('vehicle.*')
                print(vehicles)
                time.sleep(2.0)
                self.set_synchronous_mode(True)

                dist = self.set_dist(car_type)
                azim_centre = -90
                view_idx = 0
                for elev in np.linspace(5, 35, 5):
                    for azim in np.linspace(-60, 60, 20):
                        azim = azim + azim_centre
                        self.set_camera_relative_position(elev=elev, azim=azim, dist=dist)
                        for i in range(3):
                            self.capture = True
                            self.world.tick()
                        
                        self.render(self.display)

                        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera, 'vehicle')
                        patch_bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera, 'patch')
                        bbox, rpoints = ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes, patch_bounding_boxes)

                        cv2.imwrite(os.path.join(car_path, f'{str(view_idx).zfill(3)}.png'), self.collect_data)
                        label = {'state':[elev, azim - azim_centre, dist], 'bbox':bbox, 'rpoints':rpoints}
                        with open(os.path.join(car_path, f'{str(view_idx).zfill(3)}.json'), 'w') as file:
                            json.dump(label, file)

                        pygame.display.flip()

                        pygame.event.pump()
                        if self.control(self.car):
                            return
                        view_idx += 1
                self.set_synchronous_mode(False)
                self.camera.destroy()
                self.car.destroy()

        finally:
            self.set_synchronous_mode(False)
            if self.camera.is_alive:
                self.camera.destroy()
            if self.car.is_alive:
                self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()