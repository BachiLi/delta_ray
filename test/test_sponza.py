import torch
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np
import load_obj

resolution = [256, 256]
cam = camera.Camera(position     = np.array([14.4499, 6.89415, -1.18547], dtype=np.float32),
                    look_at      = np.array([13.4563, 6.78438, -1.15861], dtype=np.float32),
                    up           = np.array([-0.109708, 0.993957, 0.00374357], dtype=np.float32),
                    cam_to_world = None,
                    fov          = 90.4701,
                    clip_near    = 0.01,
                    clip_far     = 10000.0,
                    resolution   = resolution)

mat_grey = \
	material.Material(albedo=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
materials = [mat_grey]
sponza_indices, sponza_vertices, sponza_uvs = \
	load_obj.load_obj('results/sponza/sponza.obj')
sponza_indices=torch.from_numpy(sponza_indices)
sponza_vertices=torch.from_numpy(sponza_vertices)
sponza_uvs=torch.from_numpy(sponza_uvs)
shape_sponza=shape.Shape(sponza_vertices,sponza_indices,sponza_uvs,0)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.1,17,-0.1],[-0.1,17,0.1],[0.1,17,-0.1],[0.1,17,0.1]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,2,1],[1,2,3]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,0)
shapes = [shape_sponza, shape_light]
light_intensity=torch.from_numpy(\
    np.array([2000,2000,2000],dtype=np.float32))
light=light.Light(1, light_intensity)
lights=[light]
args=render_pytorch.RenderFunction.serialize_scene(\
	cam,materials,shapes,lights,resolution,4)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'results/test_sponza/sponza.exr')