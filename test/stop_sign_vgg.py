import torch
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np
import load_obj
import math
import transform
import torchvision.transforms as transforms
import torchvision.models as models
import load_mitsuba

cam, materials, shapes, lights, resolution = \
    load_mitsuba.load_mitsuba('test/scenes/street.xml')
args=render_pytorch.RenderFunction.serialize_scene(\
                cam, materials, shapes, lights, cam.resolution,
                num_samples = 1024,
                max_bounces = 1)
render = render_pytorch.RenderFunction.apply
# img = render(0, *args)
# image.imwrite(img.data.numpy(), 'test/results/stop_sign/init.exr')

cam_pos = cam.position
cam_pos.requires_grad = True
cam_lookat = cam.look_at
cam_lookat.requires_grad = True
cam_up = cam.up
cam_up.requires_grad = True
org_light_pos = shapes[0].vertices.clone()
light_translation = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0], 
    dtype=np.float32)), requires_grad = True)
intensity0 = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0], 
    dtype=np.float32)), requires_grad = True)
intensity1 = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0], 
    dtype=np.float32)), requires_grad = True)
org_intensity0 = lights[0].intensity.clone()
org_intensity1 = lights[1].intensity.clone()

optimizer = torch.optim.SGD([light_translation, intensity0], lr=1.0)
net = models.vgg16(pretrained=True).eval()
m = torch.nn.Softmax()
for t in range(1000):
    print('iteration:', t)
    optimizer.zero_grad()

    cam_ = camera.Camera(position     = cam_pos,
                        look_at      = cam_lookat,
                        up           = cam_up,
                        cam_to_world = None,
                        fov          = cam.fov,
                        clip_near    = cam.clip_near,
                        clip_far     = cam.clip_far,
                        resolution   = (224, 224))
    shapes[0].vertices = org_light_pos + light_translation
    lights[0].intensity = org_intensity0 + intensity0
    lights[1].intensity = org_intensity1 + intensity1
    args=render_pytorch.RenderFunction.serialize_scene(\
                    cam_, materials, shapes, lights, cam.resolution,
                    num_samples = 4,
                    max_bounces = 1)
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'test/results/stop_sign/render_%04d.exr' % (t))
    nimg = img.permute(2, 1, 0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    nimg = normalize(nimg)
    nimg = nimg.unsqueeze(0)
    vec = m(net(nimg))
    tk = torch.topk(vec, 5)

    loss = vec[0, 919] + vec[0, 920]
    loss.backward()

    print('light_translation:', light_translation)
    print('intensity0:', intensity0)
    print('intensity1:', intensity1)
    print('tk:', tk)
    print('loss:', loss)

    optimizer.step()

# resolution = [299, 299]
# position = Variable(torch.from_numpy(np.array([0, 2, -5], dtype=np.float32)))
# look_at = Variable(torch.from_numpy(np.array([0, 0, 0], dtype=np.float32)))
# up = Variable(torch.from_numpy(np.array([0, 1, 0], dtype=np.float32)))
# fov = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32)))
# clip_near = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32)))
# clip_far = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32)))
# cam = camera.Camera(position     = position,
#                     look_at      = look_at,
#                     up           = up,
#                     cam_to_world = None,
#                     fov          = fov,
#                     clip_near    = clip_near,
#                     clip_far     = clip_far,
#                     resolution   = resolution)

# mat_grey=material.Material(\
#     diffuse_reflectance=torch.from_numpy(np.array([0.381592,0.381592,0.381592],dtype=np.float32)))
# mat_sign=material.Material(\
#     diffuse_reflectance=torch.from_numpy(image.imread('results/stop_sign/sign.png')))
# materials=[mat_grey, mat_sign]
# pole_indices, pole_vertices, pole_uvs, pole_normals =\
#     load_obj.load_obj('results/stop_sign/stop_sign_rest.obj')
# pole_indices=torch.from_numpy(pole_indices)
# pole_vertices=torch.from_numpy(pole_vertices)
# pole_uvs=torch.from_numpy(pole_uvs)
# shape_pole=shape.Shape(pole_vertices, pole_indices, pole_uvs, pole_normals, 0)
# sign_indices, sign_vertices, sign_uvs, sign_normals =\
#     load_obj.load_obj('results/stop_sign/stop_sign_sign.obj')
# sign_uvs[:,1] = 1.0 - sign_uvs[:,1]
# sign_indices=torch.from_numpy(sign_indices)
# sign_vertices=torch.from_numpy(sign_vertices)
# sign_uvs=torch.from_numpy(sign_uvs)
# shape_sign=shape.Shape(sign_vertices, sign_indices, sign_uvs, sign_normals, 1)
# light_vertices=Variable(torch.from_numpy(\
#     np.array([[-0.1,-0.1,-12],[0.1,-0.1,-12],[-0.1,0.1,-12],[0.1,0.1,-12]],dtype=np.float32)))
# light_indices=torch.from_numpy(\
#     np.array([[0,1,2],[1,3,2]],dtype=np.int32))
# shape_light=shape.Shape(light_vertices,light_indices,None,None,0)
# shapes=[shape_pole,shape_sign,shape_light]
# light_intensity=torch.from_numpy(\
#     np.array([2000,2000,2000],dtype=np.float32))
# light=light.Light(2,light_intensity)
# lights=[light]
# angles=Variable(torch.from_numpy(np.array([0.0, -0.36*math.pi, 0.0], np.float32)), requires_grad=True)
# translation=Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0], np.float32)), requires_grad=True)
# lgt_translation=Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0], np.float32)), requires_grad=True)
# # angles=Variable(torch.from_numpy(np.array([0.0022, -1.5708,  0.0022], np.float32)), requires_grad=True)
# # translation=Variable(torch.from_numpy(1e-04 * np.array([-6.5446,  6.4331, -1.7775], np.float32)), requires_grad=True)
# #net = models.vgg16_bn(pretrained=True).eval()
# net = models.inception_v3(pretrained=True).eval()
# m = torch.nn.Softmax()

# optimizer = torch.optim.SGD([angles, translation, lgt_translation], lr=1e-5)
# for t in range(10000):
#     optimizer.zero_grad()
#     rot = transform.torch_rotate_matrix(angles)
#     shape_pole.vertices = pole_vertices@torch.t(rot) + translation
#     shape_sign.vertices = sign_vertices@torch.t(rot) + translation
#     shape_light.vertices = light_vertices + lgt_translation
#     args=render_pytorch.RenderFunction.serialize_scene(\
#         cam,materials,shapes,lights,resolution,1,1)
#     render = render_pytorch.RenderFunction.apply
#     img = render(t, *args)
#     img = img.permute(2, 1, 0)
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     nimg = normalize(img)
#     nimg = nimg.unsqueeze(0)
#     vec = m(net(nimg))
#     tk = torch.topk(vec, 5)
#     image.imwrite(img.data.numpy(), 'results/stop_sign/render_%04d_%f_%f.exr' % (t, vec[0, 919], vec[0, 620]))
#     print(tk)
#     loss = vec[0, 919]

#     print('loss:', loss.item())

#     loss.backward()
#     print('angles.grad:', angles.grad)
#     print('translation.grad:', translation.grad)

#     optimizer.step()
#     print('angles:', angles)
#     print('translation:', translation)

#     #if tk[1][0,0].item() != 919:
#     #    exit()
