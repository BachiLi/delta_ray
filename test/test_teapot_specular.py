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
    load_mitsuba.load_mitsuba('test/scenes/teapot_specular.xml')

materials[-1].diffuse_reflectance = \
    Variable(torch.from_numpy(np.array([0.15, 0.2, 0.15], dtype=np.float32)))
materials[-1].specular_reflectance = \
    Variable(torch.from_numpy(np.array([0.8, 0.8, 0.8], dtype=np.float32)))
materials[-1].roughness = \
    Variable(torch.from_numpy(np.array([0.0001], dtype=np.float32)))

args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 256,
    max_bounces = 2)
render = render_pytorch.RenderFunction.apply
# img = render(0, *args)
# image.imwrite(img.data.numpy(), 'test/results/teapot_specular/target.exr')
target = Variable(torch.from_numpy(image.imread('test/results/teapot_specular/target.exr')))
image.imwrite(target.data.numpy(), 'test/results/teapot_specular/target.png')
ref_pos = shapes[-1].vertices
translation = Variable(torch.from_numpy(np.array([20.0, 0.0, 2.0], dtype=np.float32)), requires_grad=True)
shapes[-1].vertices = ref_pos + translation
args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 256,
    max_bounces = 2)
# img = render(1, *args)
# image.imwrite(img.data.numpy(), 'test/results/teapot_specular/init.png')
# diff = torch.abs(target - img)
# image.imwrite(diff.data.numpy(), 'test/results/teapot_specular/init_diff.png')

optimizer = torch.optim.Adam([translation], lr=0.5)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()

    shapes[-1].vertices = ref_pos + translation
    args=render_pytorch.RenderFunction.serialize_scene(\
        cam, materials, shapes, lights, resolution,
        num_samples = 4,
        max_bounces = 2)
    img = render(t+1, *args)
    image.imwrite(img.data.numpy(), 'test/results/teapot_specular/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('translation.grad:', translation.grad)

    optimizer.step()
    print('translation:', translation)

shapes[-1].vertices = ref_pos + translation
args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 256,
    max_bounces = 2)
img = render(202, *args)
image.imwrite(img.data.numpy(), 'test/results/teapot_specular/final.exr')
image.imwrite(img.data.numpy(), 'test/results/teapot_specular/final.png')
image.imwrite(np.abs(target.data.numpy() - img.data.numpy()), 'test/results/teapot_specular/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/teapot_specular/iter_%d.png", "-vb", "20M",
    "test/results/teapot_specular/out.mp4"])
