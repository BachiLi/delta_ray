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
mat_grey=material.Material(diffuse_reflectance=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)),
                           specular_reflectance=torch.from_numpy(np.array([1.0,1.0,1.0],dtype=np.float32)),
                           roughness=torch.from_numpy(np.array([0.01],dtype=np.float32)))
mat_black=material.Material(diffuse_reflectance=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)))
materials=[mat_grey,mat_black]
floor_vertices=Variable(torch.from_numpy(\
    np.array([[-2.0,0.0,-2.0],[-2.0,0.0,2.0],[2.0,0.0,-2.0],[2.0,0.0,2.0]],dtype=np.float32)))
floor_indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
shape_floor=shape.Shape(floor_vertices,floor_indices,None,None,0)
light_translation = Variable(torch.from_numpy(\
    np.array([0.0,0.0,0.0],dtype=np.float32)))
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.1,5,6.9],[-0.1,5,7.1],[0.1,5,6.9],[0.1,5,7.1]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,2,1],[1,2,3]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,1)
shapes=[shape_floor,shape_light]
light_intensity=torch.from_numpy(\
    np.array([100,100,100],dtype=np.float32))
light=light.Light(1,light_intensity)
lights=[light]
args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,256,1)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/test_glossy/target.exr')
exit()
light_translation = Variable(torch.from_numpy(\
    np.array([-2.0,-0.5,-0.5],dtype=np.float32)), requires_grad=True)

optimizer = torch.optim.Adam([light_translation], lr=5e-2)
for t in range(200):
    print('iteration:', t)
    shape_light.vertices=light_vertices+light_translation
    args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,4,1)
    # To apply our Function, we use Function.apply method. We alias this as 'render'.
    render = render_pytorch.RenderFunction.apply

    optimizer.zero_grad()
    # Forward pass: render the image
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'test/results/test_glossy/iter_{}.png'.format(t))
    target = Variable(torch.from_numpy(image.imread('test/results/test_glossy/target.exr')))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', light_translation.grad)

    optimizer.step()
    print('light_translation:', light_translation)
    print('shape_light.vertices:', shape_light.vertices)

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i", "test/results/test_glossy/iter_%d.png", "-vb", "20M", "test/results/test_glossy/out.mp4"])