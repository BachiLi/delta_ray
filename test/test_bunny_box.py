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
    load_mitsuba.load_mitsuba('test/scenes/bunny_box.xml')
shapes[-1].vertices += Variable(torch.from_numpy(np.array([0, 0.01, 0], dtype=np.float32))) 
args=render_pytorch.RenderFunction.serialize_scene(\
    cam, materials, shapes, lights, resolution,
    num_samples = 625,
    max_bounces = 6)
render = render_pytorch.RenderFunction.apply
# img = render(0, *args)
# image.imwrite(img.data.numpy(), 'test/results/bunny_box/target.exr')

bunny_vertices = shapes[-1].vertices.clone()
bunny_translation = Variable(torch.from_numpy(\
    np.array([0.1,0.4,0.1],dtype=np.float32)), requires_grad=True)
bunny_rotation = Variable(torch.from_numpy(\
    np.array([-0.2,0.1,-0.1],dtype=np.float32)), requires_grad=True)
#bunny_translation = Variable(torch.from_numpy(\
#    np.array([0.0485, -0.1651, -0.0795],dtype=np.float32)), requires_grad=True)
#bunny_rotation = Variable(torch.from_numpy(\
#    np.array([-0.2,0.1,-0.1],dtype=np.float32)), requires_grad=True)
target = Variable(torch.from_numpy(image.imread('test/results/bunny_box/target.exr')))


optimizer = torch.optim.Adam([bunny_translation, bunny_rotation], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    bunny_rotation_matrix = transform.torch_rotate_matrix(bunny_rotation)

    shapes[-1].vertices = \
        (bunny_vertices-torch.mean(bunny_vertices, 0))@torch.t(bunny_rotation_matrix) + \
        torch.mean(bunny_vertices, 0) + bunny_translation
    args=render_pytorch.RenderFunction.serialize_scene(\
        cam, materials, shapes, lights, resolution,
        num_samples = 4,
        max_bounces = 6)
    img = render(t+1, *args)
    image.imwrite(img.data.numpy(), 'test/results/bunny_box/iter_{}.png'.format(t))

    dirac = np.zeros([7,7], dtype=np.float32)
    dirac[3,3] = 1.0
    dirac = Variable(torch.from_numpy(dirac))
    f = np.zeros([3, 3, 7, 7], dtype=np.float32)
    gf = scipy.ndimage.filters.gaussian_filter(dirac, 1.0)
    f[0, 0, :, :] = gf
    f[1, 1, :, :] = gf
    f[2, 2, :, :] = gf
    f = Variable(torch.from_numpy(f))
    m = torch.nn.AvgPool2d(2)

    res = 256
    diff_0 = (img - target).view(1, res, res, 3).permute(0, 3, 2, 1)
    diff_1 = m(torch.nn.functional.conv2d(diff_0, f, padding=3)) # 128 x 128
    diff_2 = m(torch.nn.functional.conv2d(diff_1, f, padding=3)) # 64 x 64
    diff_3 = m(torch.nn.functional.conv2d(diff_2, f, padding=3)) # 32 x 32
    diff_4 = m(torch.nn.functional.conv2d(diff_3, f, padding=3)) # 16 x 16
    loss = diff_0.pow(2).sum() / (res*res) + \
           diff_1.pow(2).sum() / ((res/2)*(res/2)) + \
           diff_2.pow(2).sum() / ((res/4)*(res/4)) + \
           diff_3.pow(2).sum() / ((res/8)*(res/8)) + \
           diff_4.pow(2).sum() / ((res/16)*(res/16))
    print('loss:', loss.item())

    loss.backward()
    print('bunny_translation.grad:', bunny_translation.grad)
    print('bunny_rotation.grad:', bunny_rotation.grad)

    optimizer.step()
    print('bunny_translation:', bunny_translation)
    print('bunny_rotation:', bunny_rotation)

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "test/results/bunny_box/iter_%d.png", "-vb", "20M",
    "test/results/bunny_box/out.mp4"])
