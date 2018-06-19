import load_mitsuba
import render_pytorch
import image
import transform
import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
import scipy.ndimage.filters

cam, materials, shapes, lights, resolution = \
    load_mitsuba.load_mitsuba('test/scenes/room_0/room.xml')
args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 625,
    max_bounces = 1)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'test/results/room_0/target.exr')
image.imwrite(img.data.numpy(), 'test/results/room_0/target.png')

diffuse_reflectance_bases = []
mat_variables = []
# Don't optimize the last 3 materials
for mat_id in range(len(materials)):
    mat = materials[mat_id]
    d = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    diffuse_reflectance_bases.append(\
        Variable(torch.from_numpy(\
            scipy.special.logit(d)), requires_grad = True))

    mat_variables.append(diffuse_reflectance_bases[-1])
lgt_variables = []
lgt_intensity_bases = []
for lgt in lights:
    lgt_intensity_bases.append(\
        Variable(torch.from_numpy(\
            np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            requires_grad = True))
    lgt_variables.append(lgt_intensity_bases[-1])

for i in range(len(materials)):
    materials[i].diffuse_reflectance = torch.sigmoid(diffuse_reflectance_bases[i])
for i in range(len(lights)):
    lights[i].intensity = torch.abs(500 * lgt_intensity_bases[i])

args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 625,
    max_bounces = 1)
img = render(1, *args)
image.imwrite(img.data.numpy(), 'test/results/room_0/init.exr')
image.imwrite(img.data.numpy(), 'test/results/room_0/init.png')
target = Variable(torch.from_numpy(image.imread('test/results/room_0/target.exr')))

mat_optimizer = torch.optim.Adam(mat_variables, lr=5e-3)
lgt_optimizer = torch.optim.Adam(lgt_variables, lr=5e-3)
for t in range(2000):
    print('iteration:', t)
    mat_optimizer.zero_grad()
    lgt_optimizer.zero_grad()

    for i in range(len(materials)):
        materials[i].diffuse_reflectance = torch.sigmoid(diffuse_reflectance_bases[i])
    for i in range(len(lights)):
        lights[i].intensity = torch.abs(500 * lgt_intensity_bases[i])
    args=render_pytorch.RenderFunction.serialize_scene(\
        cam, materials, shapes, lights, resolution,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *args)
    image.imwrite(img.data.numpy(), 'test/results/room_0/iter_%04d.png' % t)

    loss = (img - target).pow(2).sum() / (256*256)
    print('loss:', loss.item())

    loss.backward()
    mat_optimizer.step()
    lgt_optimizer.step()
    print('light intensity:', lights[0].intensity)

for i in range(len(materials)):
    materials[i].diffuse_reflectance = torch.sigmoid(diffuse_reflectance_bases[i])
for i in range(len(lights)):
    lights[i].intensity = torch.abs(500 * lgt_intensity_bases[i])
args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 625,
    max_bounces = 1)
img = render(2002, *args)
image.imwrite(img.data.numpy(), 'test/results/room_0/final.exr')
image.imwrite(img.data.numpy(), 'test/results/room_0/final.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/room_0/iter_%04d.png", "-vb", "20M",
    "test/results/room_0/out.mp4"])
