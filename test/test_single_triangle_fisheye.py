import torch
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np

resolution = [256, 256]
position = Variable(torch.from_numpy(np.array([0, 0, -1], dtype=np.float32)))
look_at = Variable(torch.from_numpy(np.array([0, 0, 0], dtype=np.float32)))
up = Variable(torch.from_numpy(np.array([0, 1, 0], dtype=np.float32)))
fov = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32)))
clip_near = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32)))
clip_far = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32)))
cam = camera.Camera(position     = position,
                    look_at      = look_at,
                    up           = up,
                    cam_to_world = None,
                    fov          = fov,
                    clip_near    = clip_near,
                    clip_far     = clip_far,
                    resolution   = resolution,
                    fisheye      = True)
mat_grey=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
materials=[mat_grey]
vertices=Variable(torch.from_numpy(\
    np.array([[-1.7,1.0,0.0], [1.0,1.0,0.0], [-0.5,-1.0,0.0]],dtype=np.float32)))
# vertices=Variable(torch.from_numpy(\
#     np.array([[-1.6,1.0,0.0], [1.1,1.0,0.0], [-0.4,-1.0,0.0]],dtype=np.float32)),
#     requires_grad=True)
indices=torch.from_numpy(np.array([[0,1,2]],dtype=np.int32))
shape_triangle=shape.Shape(vertices,indices,None,None,0)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-1,-1,-9],[1,-1,-9],[-1,1,-9],[1,1,-9]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,1,2],[1,3,2]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,0)
shapes=[shape_triangle,shape_light]
light_intensity=torch.from_numpy(\
    np.array([30,30,30],dtype=np.float32))
light=light.Light(1,light_intensity)
lights=[light]
args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,4,1)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/test_single_triangle_fisheye/target.exr')
target = Variable(torch.from_numpy(image.imread('test/results/test_single_triangle_fisheye/target.exr')))
position = Variable(torch.from_numpy(np.array([0.5, -0.5, -3.0], dtype=np.float32)),
                    requires_grad=True)

optimizer = torch.optim.Adam([position], lr=5e-2)
for t in range(200):
    print('iteration:', t)
    # To apply our Function, we use Function.apply method. We alias this as 'render'.
    cam = camera.Camera(position     = position,
                        look_at      = look_at,
                        up           = up,
                        cam_to_world = None,
                        fov          = fov,
                        clip_near    = clip_near,
                        clip_far     = clip_far,
                        resolution   = resolution,
                        fisheye      = True)
    args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,4,1)

    optimizer.zero_grad()
    # Forward pass: render the image
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'test/results/test_single_triangle_fisheye/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', position.grad)

    optimizer.step()
    print('vertices:', position)

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/test_single_triangle_fisheye/iter_%d.png",
    "-vb", "20M",
    "test/results/test_single_triangle_fisheye/out.mp4"])