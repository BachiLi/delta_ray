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
position = Variable(torch.from_numpy(np.array([0, 0, -5], dtype=np.float32)))
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
mat_checker_board=material.Material(\
    diffuse_reflectance=torch.from_numpy(image.imread('test/results/test_texture/checker_board.exr')))
materials=[mat_grey, mat_checker_board]
vertices=Variable(torch.from_numpy(\
    np.array([[-1.0,-1.0,0.0], [-1.0,1.0,0.0], [1.0,-1.0,0.0], [1.0,1.0,0.0]],dtype=np.float32)))
indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
uvs=torch.from_numpy(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],dtype=np.float32))
shape_plane=shape.Shape(vertices,indices,uvs,None,1)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-1,-1,-7],[1,-1,-7],[-1,1,-7],[1,1,-7]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,1,2],[1,3,2]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,0)
shapes=[shape_plane,shape_light]
light_intensity=torch.from_numpy(\
    np.array([20,20,20],dtype=np.float32))
light=light.Light(1,light_intensity)
lights=[light]

args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,256,1)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/test_texture/target.exr')
image.imwrite(img.data.numpy(), 'test/results/test_texture/target.png')
target = Variable(torch.from_numpy(image.imread('test/results/test_texture/target.exr')))
shape_plane.vertices = Variable(torch.from_numpy(\
    np.array([[-1.1,-1.2,0.0], [-1.3,1.1,0.0], [1.1,-1.1,0.0], [0.8,1.2,0.0]],dtype=np.float32)),
    requires_grad=True)
args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,256,1)
img = render(1, *args)
image.imwrite(img.data.numpy(), 'test/results/test_texture/init.png')
diff = torch.abs(target - img)
image.imwrite(diff.data.numpy(), 'test/results/test_texture/init_diff.png')

optimizer = torch.optim.Adam([shape_plane.vertices], lr=5e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()

    # Forward pass: render the image
    args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,4,1)
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'test/results/test_texture/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', shape_plane.vertices.grad)

    optimizer.step()
    print('position:', shape_plane.vertices)

args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,256,1)
img = render(202, *args)
image.imwrite(img.data.numpy(), 'test/results/test_texture/final.exr')
image.imwrite(img.data.numpy(), 'test/results/test_texture/final.png')
image.imwrite(np.abs(target.data.numpy() - img.data.numpy()), 'test/results/test_texture/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/test_texture/iter_%d.png", "-vb", "20M",
    "test/results/test_texture/out.mp4"])