import torch
import torch.optim
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np
import math
import random
import load_obj

def safe_asin(x):
    """
        return pi/2 if x == 1, otherwise return asin(x)
    """
    safe_x = torch.where(x < 1, x, torch.zeros_like(x))
    return torch.where(x < 1, torch.asin(safe_x), (math.pi/2) * torch.ones_like(x))

def length(x):
    return torch.sqrt(torch.sum(x * x, 1))

def compute_vertex_normal(vertices, indices):
    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    normals = torch.zeros_like(vertices)
    v = [vertices[indices[:, 0]],
         vertices[indices[:, 1]],
         vertices[indices[:, 2]]]
    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / e1_len.view([-1, 1])
        side_b = e2 / e2_len.view([-1, 1])
        if i == 0:
            n = torch.cross(side_a, side_b)
            n = n / length(n).view([-1, 1])
        angle = torch.where(torch.sum(side_a * side_b, 1) < 0, 
                            math.pi - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
                            2.0 * safe_asin(0.5 * length(side_b - side_a)))
        sin_angle = torch.sin(angle)
        
        normals[indices[:, i]] += n * (sin_angle / (e1_len * e2_len)).view([-1, 1])

    normals = normals / length(normals).view([-1, 1])
    return normals

resolution = [256, 256]
cam = camera.Camera(position     = np.array([0, 3, -6], dtype=np.float32),
                    look_at      = np.array([0, 0,  0], dtype=np.float32),
                    up           = np.array([0, 1,  0], dtype=np.float32),
                    cam_to_world = None,
                    fov          = 45.0,
                    clip_near    = 0.01,
                    clip_far     = 10000.0,
                    resolution   = resolution)
mat_grey=material.Material(diffuse_reflectance=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
mat_black=material.Material(diffuse_reflectance=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)))
materials=[mat_grey,mat_black]
# plane_vertices, plane_indices=generate_plane([32, 32])
# shape_plane=shape.Shape(plane_vertices,plane_indices,None,None,0)
indices, vertices, uvs, normals = load_obj.load_obj('results/heightfield_gan/model.obj')
indices = Variable(torch.from_numpy(indices.astype(np.int64)))
vertices = Variable(torch.from_numpy(vertices))
normals = compute_vertex_normal(vertices, indices)
shape_plane=shape.Shape(vertices,indices,None,normals,0)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.1,50,-0.1],[-0.1,50,0.1],[0.1,50,-0.1],[0.1,50,0.1]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,2,1],[1,2,3]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,1)
shapes=[shape_plane,shape_light]
light_intensity=torch.from_numpy(\
    np.array([100000,100000,100000],dtype=np.float32))
light=light.Light(1,light_intensity)
lights=[light]

render = render_pytorch.RenderFunction.apply
args = render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,4,1)
img = render(random.randint(0, 1048576), *args)
image.imwrite(img.data.numpy(), 'results/heightfield_gan/test.exr')