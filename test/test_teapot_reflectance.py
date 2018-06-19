import load_mitsuba
import render_pytorch
import image
import transform
import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
import camera

cam, materials, shapes, lights, resolution = \
    load_mitsuba.load_mitsuba('test/scenes/teapot.xml')

materials[-1].diffuse_reflectance = \
    Variable(torch.from_numpy(np.array([0.3, 0.2, 0.2], dtype=np.float32)))
materials[-1].specular_reflectance = \
    Variable(torch.from_numpy(np.array([0.6, 0.6, 0.6], dtype=np.float32)))
materials[-1].roughness = \
    Variable(torch.from_numpy(np.array([0.05], dtype=np.float32)))
args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 256,
    max_bounces = 2)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/teapot_reflectance/target.exr')

cam_position = cam.position
cam_translation = Variable(torch.from_numpy(\
    np.array([-0.1,0.1,-0.1],dtype=np.float32)), requires_grad=True)
materials[-1].diffuse_reflectance = \
    Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5], dtype=np.float32)),
        requires_grad = True)
materials[-1].specular_reflectance = \
    Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5], dtype=np.float32)),
        requires_grad = True)
materials[-1].roughness = \
    Variable(torch.from_numpy(np.array([0.2], dtype=np.float32)),
        requires_grad = True)
target = Variable(torch.from_numpy(image.imread('test/results/teapot_reflectance/target.exr')))
image.imwrite(target.data.numpy(), 'test/results/teapot_reflectance/target.png')
cam = camera.Camera(position     = cam_position + cam_translation,
                    look_at      = cam.look_at,
                    up           = cam.up,
                    cam_to_world = None,
                    fov          = cam.fov,
                    clip_near    = cam.clip_near,
                    clip_far     = cam.clip_far,
                    resolution   = resolution,
                    fisheye      = False)
args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,256,2)
img = render(1, *args)
image.imwrite(img.data.numpy(), 'test/results/teapot_reflectance/init.png')
diff = torch.abs(target - img)
image.imwrite(diff.data.numpy(), 'test/results/teapot_reflectance/init_diff.png')

optimizer = torch.optim.Adam([materials[-1].diffuse_reflectance,
                              materials[-1].specular_reflectance,
                              materials[-1].roughness,
                              cam_translation], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    cam = camera.Camera(position     = cam_position + cam_translation,
                        look_at      = cam.look_at,
                        up           = cam.up,
                        cam_to_world = None,
                        fov          = cam.fov,
                        clip_near    = cam.clip_near,
                        clip_far     = cam.clip_far,
                        resolution   = resolution,
                        fisheye      = False)
    args=render_pytorch.RenderFunction.serialize_scene(\
        cam, materials, shapes, lights, resolution,
        num_samples = 4,
        max_bounces = 2)
    img = render(t+1, *args)
    image.imwrite(img.data.numpy(), 'test/results/teapot_reflectance/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('diffuse_reflectance.grad:', materials[-1].diffuse_reflectance.grad)
    print('specular_reflectance.grad:', materials[-1].specular_reflectance.grad)
    print('roughness.grad:', materials[-1].roughness.grad)

    optimizer.step()
    print('diffuse_reflectance:', materials[-1].diffuse_reflectance)
    print('specular_reflectance:', materials[-1].specular_reflectance)
    print('roughness:', materials[-1].roughness)

args=render_pytorch.RenderFunction.serialize_scene(\
    cam,materials,shapes,lights,resolution,256,2)
img = render(202, *args)
image.imwrite(img.data.numpy(), 'test/results/teapot_reflectance/final.exr')
image.imwrite(img.data.numpy(), 'test/results/teapot_reflectance/final.png')
image.imwrite(np.abs(target.data.numpy() - img.data.numpy()), 'test/results/teapot_reflectance/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/teapot_reflectance/iter_%d.png", "-vb", "20M",
    "test/results/teapot_reflectance/out.mp4"])
