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
mat_green=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.35,0.75,0.35],dtype=np.float32)))
mat_red=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.75,0.35,0.35],dtype=np.float32)))
mat_black=material.Material(\
    diffuse_reflectance=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)))
materials=[mat_green,mat_red,mat_black]
tri0_vertices=Variable(torch.from_numpy(\
    np.array([[-1.7,1.0,0.0], [1.0,1.0,0.0], [-0.5,-1.0,0.0]],dtype=np.float32)))
tri1_vertices=Variable(torch.from_numpy(\
    np.array([[-1.0,1.5,1.0], [0.2,1.5,1.0], [0.2,-1.5,1.0]],dtype=np.float32)))
# tri0_vertices=Variable(torch.from_numpy(\
#     np.array([[-1.3,1.5,0.1], [1.5,0.7,-0.2], [-0.8,-1.1,0.2]],dtype=np.float32)),
#     requires_grad=True)
tri0_indices=torch.from_numpy(np.array([[0,1,2]],dtype=np.int32))
shape_tri0=shape.Shape(tri0_vertices,tri0_indices,None,None,0)
# tri1_vertices=Variable(torch.from_numpy(\
#     np.array([[-0.5,1.2,1.2], [0.3,1.7,1.0], [0.5,-1.8,1.3]],dtype=np.float32)),
#     requires_grad=True)
tri1_indices=torch.from_numpy(np.array([[0,1,2]],dtype=np.int32))
shape_tri1=shape.Shape(tri1_vertices,tri1_indices,None,None,1)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-1,-1,-7],[1,-1,-7],[-1,1,-7],[1,1,-7]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,1,2],[1,3,2]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,2)
shapes=[shape_tri0,shape_tri1,shape_light]
light_intensity=torch.from_numpy(\
    np.array([20,20,20],dtype=np.float32))
light=light.Light(2,light_intensity)
lights=[light]
args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,256,1)

render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/test_two_triangles/target.exr')
image.imwrite(img.data.numpy(), 'test/results/test_two_triangles/target.png')
shape_tri0.vertices = Variable(torch.from_numpy(\
    np.array([[-1.3,1.5,0.1], [1.5,0.7,-0.2], [-0.8,-1.1,0.2]],dtype=np.float32)),
    requires_grad=True)
shape_tri1.vertices = Variable(torch.from_numpy(\
    np.array([[-0.5,1.2,1.2], [0.3,1.7,1.0], [0.5,-1.8,1.3]],dtype=np.float32)),
    requires_grad=True)
args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,256,1)
img = render(1, *args)
image.imwrite(img.data.numpy(), 'test/results/test_two_triangles/init.png')
args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,4,1)
target = Variable(torch.from_numpy(image.imread('test/results/test_two_triangles/target.exr')))
diff = torch.abs(target - img)
image.imwrite(diff.data.numpy(), 'test/results/test_two_triangles/init_diff.png')

optimizer = torch.optim.Adam([shape_tri0.vertices, shape_tri1.vertices], lr = 1e-2)
for t in range(200):
    # To apply our Function, we use Function.apply method. We alias this as 'render'.
    render = render_pytorch.RenderFunction.apply

    optimizer.zero_grad()
    # Forward pass: render the image
    img = render(t+1, *args)
    image.imwrite(img.data.numpy(), 'test/results/test_two_triangles/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    optimizer.step()

    print('grad:', shape_tri1.vertices.grad)

args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,256,1)
img = render(202, *args)
image.imwrite(img.data.numpy(), 'test/results/test_two_triangles/final.exr')
image.imwrite(img.data.numpy(), 'test/results/test_two_triangles/final.png')
diff = torch.abs(target - img)
image.imwrite(diff.data.numpy(), 'test/results/test_two_triangles/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24",
    "-vb", "20M",
    "-i", "test/results/test_two_triangles/iter_%d.png", "test/results/test_two_triangles/out.mp4"])