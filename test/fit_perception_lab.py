import load_mitsuba
import render_pytorch
import image
import transform
import camera
import torch
import torch.optim
from torch.autograd import Variable
import numpy as np
import scipy.ndimage.filters
import scipy.ndimage.interpolation
import scipy.special
import math
import sys

def round_to_square(x):
    ret = 1
    while True:
        if ret * ret >= x:
            return ret * ret
        ret += 1
    return ret

render = render_pytorch.RenderFunction.apply

cam, materials, shapes, lights, resolution = \
    load_mitsuba.load_mitsuba('test/scenes/perception_lab/perception_lab.xml')
cam.fisheye = True
#args=render_pytorch.RenderFunction.serialize_scene(\
#   cam, materials, shapes, lights, resolution, 256, 1)
#img = render(0, *args)
#image.imwrite(img.data.numpy(), 'test/results/perception_lab/target.exr')
#exit()
cam_variables = []
#cam.look_at.requires_grad = True
#cam.position.requires_grad = True
#variables.append(cam.look_at)
#variables.append(cam.position)
look_at_base = Variable(0.01 * cam.look_at.data, requires_grad=True)
position_base = Variable(0.01 * cam.position.data, requires_grad=True)
cam_variables.append(look_at_base)
cam_variables.append(position_base)
cam.up.requires_grad = True
cam_variables.append(cam.up)
diffuse_reflectance_bases = []
specular_reflectance_bases = []
roughness_bases = []
mat_variables = []
# Don't optimize the last 3 materials
for mat_id in range(len(materials) - 4):
    mat = materials[mat_id]
    #mat.specular_reflectance = Variable(torch.from_numpy(\
    #    np.array([0.1, 0.1, 0.1], dtype=np.float32)))
    #mat.roughness = Variable(torch.from_numpy(\
    #    np.array([0.15], dtype=np.float32)))
    # mat.diffuse_reflectance.requires_grad = True
    #mat.specular_reflectance.requires_grad = True
    #mat.roughness.requires_grad = True
    d = mat.diffuse_reflectance.data.numpy()
    diffuse_reflectance_bases.append(\
        Variable(torch.from_numpy(\
            scipy.special.logit(d)), requires_grad = True))
    s = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    specular_reflectance_bases.append(\
        Variable(torch.from_numpy(\
            scipy.special.logit(s)), requires_grad = True))
    r = np.array([0.15], dtype=np.float32)
    roughness_bases.append(\
        Variable(torch.from_numpy(\
            scipy.special.logit(r)), requires_grad = True))

    mat_variables.append(diffuse_reflectance_bases[-1])
    mat_variables.append(specular_reflectance_bases[-1])
    mat_variables.append(roughness_bases[-1])
lgt_variables = []
lgt_intensity_bases = []
for lgt in lights:
    lgt_intensity_bases.append(\
        Variable(torch.from_numpy(\
            0.1 * np.array([16, 13, 11], dtype=np.float32)),
            requires_grad = True))
    lgt_variables.append(lgt_intensity_bases[-1])

cam.position = 100 * position_base
cam.look_at = 100 * look_at_base

cam = camera.Camera(position     = cam.position,
                look_at      = cam.look_at,
                up           = cam.up,
                cam_to_world = None,
                fov          = cam.fov,
                clip_near    = cam.clip_near,
                clip_far     = cam.clip_far,
                resolution   = (512, 512),
                fisheye      = True)
for i in range(len(materials) - 4):
    materials[i].diffuse_reflectance = torch.sigmoid(diffuse_reflectance_bases[i])
    materials[i].specular_reflectance = torch.sigmoid(specular_reflectance_bases[i])
    materials[i].roughness = torch.sigmoid(roughness_bases[i])
for i in range(len(lights)):
    lights[i].intensity = torch.abs(lgt_intensity_bases[i] * 10.0)
#args=render_pytorch.RenderFunction.serialize_scene(\
#                cam, materials, shapes, lights, cam.resolution,
#                num_samples = 16384,
#                max_bounces = 16)
#img = render(0, *args)
#image.imwrite(img.data.numpy(), 'test/results/perception_lab/init.exr')
#exit()

print('load target')
org_target = image.imread('test/scenes/perception_lab/off_monitor_30_final.hdr')
#target = image.imread('test/results/perception_lab/target.exr')
num_scales = 1
for scale in range(num_scales):
    # linearly scale from 32x32 to 512x512
    downscale_factor = 512.0 / ((512.0 / num_scales) * scale + 32.0)
    print('downscale_factor', downscale_factor)
    res = round(512 / downscale_factor)
    print('scaling target')
    if scale < num_scales - 1:
        target = scipy.ndimage.interpolation.zoom(org_target, (1.0/downscale_factor, 1.0/downscale_factor, 1.0), order=1)
    else:
        downscale_factor = 1
        res = 512
        target = org_target
    print('target.shape:', target.shape)
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xv, yv = np.meshgrid(x, y)
    weight = (xv * xv + yv * yv < 1.0).astype(np.float32)
    weight = Variable(torch.from_numpy(np.stack([weight, weight, weight], axis=-1)))
    #image.imwrite(weight, 'weight.exr')
    #exit()
    image.imwrite(target, 'test/results/perception_lab/target_{}.exr'.format(scale))
    target = Variable(torch.from_numpy(target))

    cam_optimizer = torch.optim.Adam(cam_variables, lr=2e-3)
    mat_optimizer = torch.optim.Adam(mat_variables, lr=2e-3)
    lgt_optimizer = torch.optim.Adam(lgt_variables, lr=2e-3)
    base_num_iter = 50
    num_iter = base_num_iter
    if scale == num_scales - 1:
        num_iter = 200
    for t in range(num_iter):
        print('iteration: ({}, {})'.format(scale, t))
        cam_optimizer.zero_grad()
        mat_optimizer.zero_grad()
        lgt_optimizer.zero_grad()
        cam.position = 100 * position_base
        cam.look_at = 100 * look_at_base
    
        cam = camera.Camera(position     = cam.position,
                            look_at      = cam.look_at,
                            up           = cam.up,
                            cam_to_world = None,
                            fov          = cam.fov,
                            clip_near    = cam.clip_near,
                            clip_far     = cam.clip_far,
                            resolution   = (res, res),
                            fisheye      = True)
        for i in range(len(materials) - 4):
            materials[i].diffuse_reflectance = torch.sigmoid(diffuse_reflectance_bases[i])
            materials[i].specular_reflectance = torch.sigmoid(specular_reflectance_bases[i])
            materials[i].roughness = torch.sigmoid(roughness_bases[i])
        for i in range(len(lights)):
            lights[i].intensity = torch.abs(lgt_intensity_bases[i] * 10.0)
        
        num_samples = round_to_square(4 * downscale_factor)
        print('num_samples:', num_samples)
        args=render_pytorch.RenderFunction.serialize_scene(\
            cam, materials, shapes, lights, cam.resolution,
            num_samples = num_samples,
            max_bounces = 16)
        img = render(scale * num_iter + t + 1, *args)
        image.imwrite(img.data.numpy(), 'test/results/perception_lab/iter_{}.exr'.format(scale * base_num_iter + t))
    
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
    
        #res = 512
        #diff = torch.clamp(img, 0, 2) - torch.clamp(target, 0, 2)
        #diff = weight * (torch.log(img+1e-3) - torch.log(target+1e-3))
        img_grey = np.sum(img.data.numpy(), axis=-1)
        target_grey = np.sum(target.data.numpy(), axis=-1)
        clamp = Variable(torch.from_numpy(\
            np.logical_and(img_grey < 6.0, target_grey < 6.0).astype(np.float32))).view(res, res, 1)
        diff = (img - target) * clamp * weight
        #loss = (torch.abs(diff) / (torch.abs(target) + 1e-3)).sum() / (res ** 2)
        loss = torch.abs(diff).sum() / (res ** 2)
        print('loss_highres:', loss.item())
        # match DC
        loss = loss + 0.1 * torch.abs(diff.sum()) / (res ** 2)
        #diff = diff[160:, :, :]
        image.imwrite(torch.abs(diff).data.numpy(), 'test/results/perception_lab/diff_{}.exr'.format(\
                scale * base_num_iter + t))
        image.imwrite((torch.abs(diff) / (torch.abs(target) + 1e-3)).data.numpy(),
            'test/results/perception_lab/rel_diff_{}.exr'.format(scale * base_num_iter + t))
        diff = diff.view(1, res, res, 3).permute(0, 3, 2, 1)
        #for i in range(scale):
        diff = m(torch.nn.functional.conv2d(diff, f, padding=3))
        target_batch = target.view(1, res, res, 3).permute(0, 3, 2, 1)
        target_lowres = m(torch.nn.functional.conv2d(target_batch, f, padding=3))
        loss_lowres = (torch.abs(diff) / (torch.abs(target_lowres) + 1e-3)).sum() / ((res/2) ** 2)
        loss = loss + loss_lowres
        print('DC:', torch.abs(diff.sum()).item() / (res ** 2))
        print('loss_lowres:', loss_lowres.item())
        print('loss:', loss.item())
    
        loss.backward()
        #print('position_base.grad:', position_base.grad)
        #print('look_at_base.grad:', look_at_base.grad)
        #print('cam.up.grad:', cam.up.grad)
    
        cam_optimizer.step()
        mat_optimizer.step()
        lgt_optimizer.step()

        sys.stdout.flush()

        if scale == num_scales - 1 and t == num_iter - 1:
            print('cam.position:', cam.position)
            print('cam.look_at', cam.look_at)
            print('cam.up:', cam.up)
            for mat in materials:
                print('mat.diffuse_reflectance:', mat.diffuse_reflectance)
                print('mat.specular_reflectance:', mat.specular_reflectance)
                print('mat.roughness:', mat.roughness)
            for lgt in lights:
                print('lgt.intensity:', lgt.intensity)
            f = open('scene.log', 'w')
            f.write('cam.position:\n')
            f.write(str(cam.position))
            f.write('\n')
            f.write('cam.look_at:\n')
            f.write(str(cam.look_at))
            f.write('\n')
            f.write('cam.up:\n')
            f.write(str(cam.up))
            f.write('\n')
            for mat in materials:
                f.write('mat.diffuse_reflectance:\n')
                f.write(str(mat.diffuse_reflectance))
                f.write('\n')
                f.write('mat.specular_reflectance:\n')
                f.write(str(mat.specular_reflectance))
                f.write('\n')
                f.write('mat.roughness:\n')
                f.write(str(mat.roughness))
                f.write('\n')
            for lgt in lights:
                f.write('lgt.intensity:\n')
                f.write(str(lgt.intensity))
                f.write('\n')
            f.close()
            print('Rendering final')
            sys.stdout.flush()
            args=render_pytorch.RenderFunction.serialize_scene(\
                    cam, materials, shapes, lights, cam.resolution,
                    num_samples = 16384,
                    max_bounces = 16)
            img = render(scale * num_iter + t + 1, *args)
            image.imwrite(img.data.numpy(), 'test/results/perception_lab/final.exr')

