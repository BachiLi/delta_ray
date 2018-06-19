import torch
import torch.optim
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np

resolution = [256, 256]
position = Variable(torch.from_numpy(np.array([0, 2, -5], dtype=np.float32)))
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
                    resolution   = resolution)
mat_grey=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
mat_black=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)))
materials=[mat_grey,mat_black]
floor_vertices=Variable(torch.from_numpy(\
    np.array([[-2.0,0.0,-2.0],[-2.0,0.0,2.0],[2.0,0.0,-2.0],[2.0,0.0,2.0]],dtype=np.float32)))
floor_indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
shape_floor=shape.Shape(floor_vertices,floor_indices,None,None,0)
blocker_vertices=Variable(torch.from_numpy(\
    np.array([[-0.5,3.0,-0.5],[-0.5,3.0,0.5],[0.5,3.0,-0.5],[0.5,3.0,0.5]],dtype=np.float32)))
blocker_indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
shape_blocker=shape.Shape(blocker_vertices,blocker_indices,None,None,0)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.1,5,-0.1],[-0.1,5,0.1],[0.1,5,-0.1],[0.1,5,0.1]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,2,1],[1,2,3]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,1)
shapes=[shape_floor,shape_blocker,shape_light]
light_intensity=torch.from_numpy(\
    np.array([1000,1000,1000],dtype=np.float32))
light=light.Light(2,light_intensity)
lights=[light]

args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,256,1)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/test_shadow/target.exr')
image.imwrite(img.data.numpy(), 'test/results/test_shadow/target.png')
target = Variable(torch.from_numpy(image.imread('test/results/test_shadow/target.exr')))
shape_blocker.vertices=Variable(torch.from_numpy(\
    np.array([[-0.2,3.5,-0.8],[-0.8,3.0,0.3],[0.4,2.8,-0.8],[0.3,3.2,1.0]],dtype=np.float32)),
    requires_grad=True)
args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,256,1)
img = render(1, *args)
image.imwrite(img.data.numpy(), 'test/results/test_shadow/init.png')
diff = torch.abs(target - img)
image.imwrite(diff.data.numpy(), 'test/results/test_shadow/init_diff.png')

optimizer = torch.optim.Adam([shape_blocker.vertices], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    # To apply our Function, we use Function.apply method. We alias this as 'render'.
    render = render_pytorch.RenderFunction.apply

    optimizer.zero_grad()
    # Forward pass: render the image
    args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,4,1)
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'test/results/test_shadow/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', shape_blocker.vertices.grad)

    optimizer.step()
    print('shape_blocker.vertices:', shape_blocker.vertices)

args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,256,1)
img = render(202, *args)
image.imwrite(img.data.numpy(), 'test/results/test_shadow/final.exr')
image.imwrite(img.data.numpy(), 'test/results/test_shadow/final.png')
image.imwrite(np.abs(target.data.numpy() - img.data.numpy()), 'test/results/test_shadow/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/test_shadow/iter_%d.png", "-vb", "20M",
    "test/results/test_shadow/out.mp4"])