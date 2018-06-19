import math
import numpy as np
import transform
import torch
from torch.autograd import Variable

class Camera:
    def __init__(self,
                 position,
                 look_at,
                 up,
                 cam_to_world,
                 fov,
                 clip_near,
                 clip_far,
                 resolution,
                 fisheye = False):
        self.resolution = resolution
        self.num_pixels = resolution[0] * resolution[1]

        if cam_to_world is None:
            self.position = position
            self.look_at = look_at
            self.up = up
            self.cam_to_world = transform.gen_look_at_matrix(position, look_at, up)
        else:
            self.cam_to_world = cam_to_world
        self.world_to_cam = torch.inverse(self.cam_to_world)

        aspect_ratio = resolution[0] / resolution[1]
        s = Variable(torch.from_numpy(\
            np.array([-0.5, -0.5 * aspect_ratio, 1.0], dtype=np.float32)))
        t = Variable(torch.from_numpy(\
            np.array([-1.0, -1.0 / aspect_ratio, 0.0], dtype=np.float32)))
        s = transform.gen_scale_matrix(s)
        t = transform.gen_translate_matrix(t)
        p = transform.gen_perspective_matrix(fov, clip_near, clip_far)
        self.cam_to_sample = s @ t @ p
        self.sample_to_cam = torch.inverse(self.cam_to_sample)

        self.fov = fov
        self.fov_factor = torch.tan(transform.radians(0.5 * fov))
        self.aspect_ratio = aspect_ratio
        self.clip_near = clip_near
        self.clip_far = clip_far

        self.fisheye = fisheye
