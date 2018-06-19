import torch
import torch.optim
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import transform
import numpy as np

dtype = torch.FloatTensor
resolution = [256, 256]
cam = camera.Camera(position     = np.array([0, 2, -5], dtype=np.float32),
                    look_at      = np.array([0, 0,  0], dtype=np.float32),
                    up           = np.array([0, 1,  0], dtype=np.float32),
                    cam_to_world = None,
                    fov          = 45.0,
                    clip_near    = 0.01,
                    clip_far     = 10000.0,
                    resolution   = resolution)
mat_grey=material.Material(albedo=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
mat_red=material.Material(albedo=torch.from_numpy(np.array([0.9,0.15,0.15],dtype=np.float32)))
mat_green=material.Material(albedo=torch.from_numpy(np.array([0.15,0.9,0.15],dtype=np.float32)))
mat_black=material.Material(albedo=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)))
materials=[mat_grey,mat_red,mat_green,mat_black]
floor_vertices=Variable(torch.from_numpy(\
    np.array([[-2.0,0.0,-2.0],[-2.0,0.0,2.0],[2.0,0.0,-2.0],[2.0,0.0,2.0]],dtype=np.float32)))
floor_indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
shape_floor=shape.Shape(floor_vertices,floor_indices,None,None,0)
red_reflector_vertices=Variable(torch.from_numpy(\
    np.array([[-4.0,4.0,2.0],[-4.0,8.0,2.0],[0.0,4.0,2.0],[0.0,8.0,2.0]],dtype=np.float32)))
red_reflector_indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
shape_red_reflector=shape.Shape(red_reflector_vertices,red_reflector_indices,None,None,1)
green_reflector_vertices=Variable(torch.from_numpy(\
    np.array([[0.0,4.0,2.0],[0.0,8.0,2.0],[4.0,4.0,2.0],[4.0,8.0,2.0]],dtype=np.float32)))
green_reflector_indices=torch.from_numpy(np.array([[0,1,2], [1,3,2]],dtype=np.int32))
shape_green_reflector=shape.Shape(green_reflector_vertices,green_reflector_indices,None,None,2)
light_translation=Variable(torch.from_numpy(\
    np.array([0.0,5.0,-2.0],dtype=np.float32)), requires_grad=True)
light_rotation=Variable(torch.from_numpy(\
    np.array([2.5,0.0,0.0], dtype=np.float32)), requires_grad=True)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.1,0,-0.1],[-0.1,0,0.1],[0.1,0,-0.1],[0.1,0,0.1]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,2,1],[1,2,3]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,3)
shapes=[shape_floor,shape_red_reflector,shape_green_reflector,shape_light]
light_intensity=torch.from_numpy(\
    np.array([10000,10000,10000],dtype=np.float32))
light=light.Light(3,light_intensity)
lights=[light]

optimizer = torch.optim.Adam([light_rotation], lr = 1e-2)
for t in range(100):
    print('iteration:', t)
    print('light_rotation', light_rotation)
    light_rotation_matrix=transform.torch_rotate_matrix(light_rotation)
    shape_light.vertices=light_vertices@torch.t(light_rotation_matrix)+light_translation
    args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,4,32)

    # To apply our Function, we use Function.apply method. We alias this as 'render'.
    render = render_pytorch.RenderFunction.apply

    optimizer.zero_grad()
    # Forward pass: render the image
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'results/test_gi/iter_{}.png'.format(t))
    target = Variable(torch.from_numpy(image.imread('results/test_gi/target.exr')))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.data[0])

    loss.backward()
    print('grad:', light_rotation.grad)

    optimizer.step()
