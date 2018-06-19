import torch
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np

position = Variable(torch.from_numpy(np.array([0, 0, -2.5], dtype=np.float32)))
look_at = Variable(torch.from_numpy(np.array([0, 0, 0], dtype=np.float32)))
up = Variable(torch.from_numpy(np.array([0, 1, 0], dtype=np.float32)))
fov = Variable(torch.from_numpy(np.array([90.0], dtype=np.float32)))
clip_near = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32)))
clip_far = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32)))
resolution = [128, 128]
cam = camera.Camera(position     = position,
                    look_at      = look_at,
                    up           = up,
                    cam_to_world = None,
                    fov          = fov,
                    clip_near    = clip_near,
                    clip_far     = clip_far,
                    resolution   = resolution)
mat_grey=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
materials=[mat_grey]
mesh_resolution = (256, 256)
mesh_resj = list(map(lambda x: x*1j, mesh_resolution))
vertices = np.reshape(\
    np.mgrid[-1.5:1.5:mesh_resj[0], -1.5:1.5:mesh_resj[1], 0:0:1j].astype(np.float32),
    (3, mesh_resolution[0] * mesh_resolution[1]))
vertices = np.swapaxes(vertices, 0, 1)
indices = []
for y in range(mesh_resolution[0] - 1):
    for x in range(mesh_resolution[1] - 1):
        left_top     = x * mesh_resolution[1] + y
        right_top    = left_top + 1
        left_bottom  = left_top + mesh_resolution[0]
        right_bottom = left_top + mesh_resolution[0] + 1
        indices.append([left_top, right_top, left_bottom])
        indices.append([left_bottom, right_top, right_bottom])
vertices = Variable(torch.from_numpy(vertices))
indices = torch.from_numpy(np.array(indices, dtype=np.int32))
# vertices=Variable(torch.from_numpy(\
#     np.array([[-1.5,-1.5,0.0],
#               [-1.5, 1.5,0.0],
#               [ 1.5,-1.5,0.0],
#               [ 1.5, 1.5,0.0]],
#               dtype=np.float32)))
# indices=torch.from_numpy(np.array([[0,1,2],[1,3,2]],dtype=np.int32))
shape_triangle=shape.Shape(vertices,indices,None,None,0)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.01,-0.01,-0.3],
              [ 0.01,-0.01,-0.3],
              [-0.01, 0.01,-0.3],
              [ 0.01, 0.01,-0.3]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,1,2],[1,3,2]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,0)
shapes=[shape_triangle,shape_light]
light_intensity=torch.from_numpy(\
    np.array([2500,2500,2500],dtype=np.float32))
light=light.Light(1,light_intensity)
lights=[light]
args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,16,1)

render = render_pytorch.RenderFunction.apply
img = render(0, *args)
img = -img
img = img.data.numpy()
print(np.min(img))
print(np.max(img))
vmin = -0.1
vmax =  0.1
img = np.clip((img[:, :, 0] - vmin) / (vmax - vmin), 0.0, 1.0)
import matplotlib.cm as cm
import skimage.io
img = cm.viridis(img)
skimage.io.imsave('results/test_plane_pointlight/dx.png', img)

# image.imwrite(img, 'results/test_plane_pointlight/plane.png')
