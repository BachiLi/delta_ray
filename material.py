import torch
from torch.autograd import Variable
import numpy as np

class Material:
    def __init__(self,
    			 diffuse_reflectance,
    			 specular_reflectance = Variable(torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32))),
    			 roughness = Variable(torch.from_numpy(np.array([1.0],dtype=np.float32))),
    			 diffuse_uv_scale = Variable(torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32))),
    			 specular_uv_scale = Variable(torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32))),
    			 roughness_uv_scale = Variable(torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32))),
    			 two_sided=False):
        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.roughness = roughness
        self.diffuse_uv_scale = diffuse_uv_scale
        self.specular_uv_scale = diffuse_uv_scale
        self.roughness_uv_scale = diffuse_uv_scale
        self.two_sided = two_sided