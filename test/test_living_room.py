import load_mitsuba
import render_pytorch
import image
import transform
import torch
import torch.optim
from torch.autograd import Variable
import numpy as np

cam, materials, shapes, lights, resolution = \
	load_mitsuba.load_mitsuba('results/living-room-3/scene.xml')
args=render_pytorch.RenderFunction.serialize_scene(\
	cam, materials, shapes, lights, resolution, 64, 32)
render = render_pytorch.RenderFunction.apply
img = render(0, *args)
image.imwrite(img.data.numpy(), 'results/test_living_room/living_room.exr')
target_luminance = torch.mean(0.212671 * img[:,:,0] + 0.715160 * img[:,:,1] + 0.072169 * img[:,:,2])
print('target_luminance:', target_luminance)
light_translation=Variable(torch.from_numpy(\
    np.array([0.0,0.0,0.0],dtype=np.float32)), requires_grad=True)
light_rotation=Variable(torch.from_numpy(\
    np.array([0.0,0.0,0.0], dtype=np.float32)), requires_grad=True)
light_vertices=shapes[-1].vertices.clone()

optimizer = torch.optim.Adam([light_translation, light_rotation], lr = 5e-2)
for t in range(100):
    print('iteration:', t)
    print('light_translation', light_translation)
    print('light_rotation', light_rotation)
    light_rotation_matrix=transform.torch_rotate_matrix(light_rotation)
    shapes[-1].vertices=light_vertices@torch.t(light_rotation_matrix)+light_translation
    args=render_pytorch.RenderFunction.serialize_scene(cam,materials,shapes,lights,resolution,4,32)

    # To apply our Function, we use Function.apply method. We alias this as 'render'.
    render = render_pytorch.RenderFunction.apply

    optimizer.zero_grad()
    # Forward pass: render the image
    img = render(t, *args)
    image.imwrite(img.data.numpy(), 'results/test_living_room/iter_{}.png'.format(t))
    img_x = torch.mean(0.412453 * img[:,:,0] + 0.357580 * img[:,:,1] + 0.180423 * img[:,:,2])
    img_y = torch.mean(0.212671 * img[:,:,0] + 0.715160 * img[:,:,1] + 0.072169 * img[:,:,2])
    img_z = torch.mean(0.019334 * img[:,:,0] + 0.119193 * img[:,:,1] + 0.119193 * img[:,:,2])
    print('luminance:', img_y)
    loss = 10 * (img_y - target_luminance).pow(2).sum() + img_x.pow(2).sum() + img_z.pow(2).sum()
    print('loss:', loss.data[0])

    loss.backward()
    print('light_translation.grad:', light_translation.grad)
    print('light_rotation.grad:', light_rotation.grad)

    optimizer.step()
