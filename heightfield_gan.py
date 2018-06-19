import torch
import torch.optim
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import render_pytorch
import image
import camera
import material
import light
import shape
import numpy as np
import math
import random
import glob
import os

def generate_plane(resolution):
    mesh_resj = list(map(lambda x: x*1j, resolution))
    vertices = np.reshape(\
        np.mgrid[-5.0:5.0:mesh_resj[0], 0:0:1j, -5.0:5.0:mesh_resj[1]].astype(np.float32),
        [3, resolution[0] * resolution[1]])
    vertices = np.swapaxes(vertices, 0, 1)
    indices = []
    for y in range(resolution[0] - 1):
        for x in range(resolution[1] - 1):
            left_top     = x * resolution[1] + y
            right_top    = left_top + 1
            left_bottom  = left_top + resolution[0]
            right_bottom = left_top + resolution[0] + 1
            indices.append([left_top, right_top, left_bottom])
            indices.append([left_bottom, right_top, right_bottom])
    return (Variable(torch.from_numpy(vertices)), Variable(torch.from_numpy(np.array(indices))))

def generate_heightfield(resolution, amplitude, freq_x, freq_y, phase_x, phase_y):
    mesh_resj = list(map(lambda x: x*1j, resolution))
    coords = np.mgrid[-5.0:5.0:mesh_resj[0], -5.0:5.0:mesh_resj[1]].astype(np.float32)
    height = amplitude * np.sin(freq_x * coords[0,:,:] + phase_x) * freq_y * np.cos(coords[1] + phase_x)
    vertices = np.stack([np.zeros(resolution, dtype=np.float32),
                         height,
                         np.zeros(resolution, dtype=np.float32)], axis=-1)
    vertices = np.reshape(vertices, [-1, 3])
    return vertices

def safe_asin(x):
    """
        return pi/2 if x == 1, otherwise return asin(x)
    """
    safe_x = torch.where(x < 1, x, torch.zeros_like(x))
    return torch.where(x < 1, torch.asin(safe_x), (math.pi/2) * torch.ones_like(x))

def length(x):
    return torch.sqrt(torch.sum(x * x, 1))

def compute_vertex_normal(vertices, indices):
    normals = torch.zeros_like(vertices)
    v = [vertices[indices[:, 0]],
         vertices[indices[:, 1]],
         vertices[indices[:, 2]]]
    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / e1_len.view([-1, 1])
        side_b = e2 / e2_len.view([-1, 1])
        if i == 0:
            n = torch.cross(side_a, side_b)
            n = n / length(n).view([-1, 1])
        angle = torch.where(torch.sum(side_a * side_b, 1) < 0, 
                            math.pi - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
                            2.0 * safe_asin(0.5 * length(side_b - side_a)))
        # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
        #sin_angle = torch.sin(angle)
        #normals[indices[:, i]] += n * (sin_angle / (e1_len * e2_len)).view([-1, 1])
        # Thurmer and Wuthrich 1998
        normals[indices[:, i]] += n * angle.view([-1, 1])

    normals = normals / length(normals).view([-1, 1])
    return normals

resolution = [32, 32]
lowres_cam = camera.Camera(\
                    position     = Variable(torch.from_numpy(np.array([0, 3, -6], dtype=np.float32))),
                    look_at      = Variable(torch.from_numpy(np.array([0, 0,  0], dtype=np.float32))),
                    up           = Variable(torch.from_numpy(np.array([0, 1,  0], dtype=np.float32))),
                    cam_to_world = None,
                    fov          = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32))),
                    clip_near    = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32))),
                    clip_far     = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32))),
                    resolution   = resolution)
highres_cam = camera.Camera(\
                    position     = Variable(torch.from_numpy(np.array([0, 3, -6], dtype=np.float32))),
                    look_at      = Variable(torch.from_numpy(np.array([0, 0,  0], dtype=np.float32))),
                    up           = Variable(torch.from_numpy(np.array([0, 1,  0], dtype=np.float32))),
                    cam_to_world = None,
                    fov          = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32))),
                    clip_near    = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32))),
                    clip_far     = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32))),
                    resolution   = [256, 256])
heightfield_res = [32, 32]
cam = lowres_cam
mat_grey=material.Material(diffuse_reflectance=torch.from_numpy(np.array([0.5,0.5,0.5],dtype=np.float32)))
mat_black=material.Material(diffuse_reflectance=torch.from_numpy(np.array([0.0,0.0,0.0],dtype=np.float32)))
materials=[mat_grey,mat_black]
plane_vertices, plane_indices=generate_plane(heightfield_res)
shape_plane=shape.Shape(plane_vertices,plane_indices,None,None,0)
light_vertices=Variable(torch.from_numpy(\
    np.array([[-0.1,10,-0.1],[-0.1,10,0.1],[0.1,10,-0.1],[0.1,10,0.1]],dtype=np.float32)))
light_indices=torch.from_numpy(\
    np.array([[0,2,1],[1,2,3]],dtype=np.int32))
shape_light=shape.Shape(light_vertices,light_indices,None,None,1)
shapes=[shape_plane,shape_light]
light_intensity=torch.from_numpy(\
    np.array([4000,4000,4000],dtype=np.float32))
light=light.Light(1,light_intensity)
lights=[light]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngf = 64
        ngf = self.ngf
        self.preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * ngf),
            nn.ReLU(True),
        )
        self._4_to_8 = nn.Sequential(
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2, bias=None),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
        )
        self._8_to_16 = nn.Sequential(
            nn.ConvTranspose2d(2 * ngf, ngf, 2, stride=2, bias=None),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self._4_to_4 = nn.ConvTranspose2d(4 * ngf, 1, 1)
        self._8_to_8 = nn.ConvTranspose2d(2 * ngf, 1, 1)
        self._16_to_16 = nn.ConvTranspose2d(ngf, 1, 1)
        self._16_to_32 = nn.ConvTranspose2d(ngf, 1, 2, stride=2)

        self.resolution = resolution
        self.xz = None
        self.save_heightfield = False
        self.iteration = 0

    def forward(self, input):
        tanh = nn.Tanh()
        height_batch = self.preprocess(input)
        height_batch = height_batch.view(-1, 4 * self.ngf, 4, 4)
        _4x4 = height_batch
        _8x8 = self._4_to_8(_4x4)
        _16x16 = self._8_to_16(_8x8)
        upsample = nn.Upsample(size=(32, 32), mode='bilinear')
        height_batch = (tanh(self._16_to_32(_16x16)) + \
                        upsample(tanh(self._16_to_16(_16x16))) + \
                        upsample(tanh(self._8_to_8(_8x8))) + \
                        upsample(tanh(self._4_to_4(_4x4)))) / 4.0
        height_batch = height_batch.permute(0, 3, 2, 1)
        if np.any(np.isnan(height_batch.data.numpy())):
            print('NANNANNAN')
            exit()
        if self.save_heightfield:
            height_batch_np = height_batch.data.numpy()
            height_flatten = np.zeros([32 * 8, 32 * 8, 1])
            for i in range(8):
                for j in range(8):
                    img = height_batch_np[8 * i + j, :, :, :]
                    height_flatten[32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1), :] = img
            image.imwrite(height_flatten.squeeze(),
                'results/heightfield_gan/heightfield_%06d.png' % iteration)

        output = Variable(torch.zeros([input.shape[0], 1, 32, 32]))
        for i in range(input.shape[0]):
            height = torch.stack([\
                Variable(torch.from_numpy(np.zeros(heightfield_res, dtype=np.float32))),
                height_batch[i, :, :, 0],
                Variable(torch.from_numpy(np.zeros(heightfield_res, dtype=np.float32)))],
                dim=-1)
            height = height.view([-1, 3])
            shape_plane.vertices = plane_vertices + height
            if self.save_heightfield:
                v = shape_plane.vertices.data.numpy()
                ind = shape_plane.indices.data.numpy() + 1
                with open('results/heightfield_gan/model_%06d_%03d.obj' \
                        % (self.iteration, i), 'w') as f:
                    for vid in range(v.shape[0]):
                        f.write('v %f %f %f\n' % (v[vid, 0], v[vid, 1], v[vid, 2]))
                    for iid in range(ind.shape[0]):
                        f.write('f %d %d %d\n' % (ind[iid, 0], ind[iid, 1], ind[iid, 2]))

            shape_plane.normals = compute_vertex_normal(shape_plane.vertices, shape_plane.indices)
            cam = camera.Camera(\
                    position     = Variable(torch.from_numpy(np.array([self.xz[i][0], 3, self.xz[i][1]], dtype=np.float32))),
                    look_at      = Variable(torch.from_numpy(np.array([0, 0,  0], dtype=np.float32))),
                    up           = Variable(torch.from_numpy(np.array([0, 1,  0], dtype=np.float32))),
                    cam_to_world = None,
                    fov          = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32))),
                    clip_near    = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32))),
                    clip_far     = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32))),
                    resolution   = self.resolution)
            args = render_pytorch.RenderFunction.serialize_scene(\
                cam,materials,shapes,lights,self.resolution,4,1)
            render = render_pytorch.RenderFunction.apply
            img = render(random.randint(0, 1048576), *args)
            img = img.permute([2, 1, 0])
            output[i, :, :, :] = img[0, :, :]
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64
        self.ndf = ndf
        self.main = nn.Sequential(
            # in: 3x32x32
            # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(1, ndf, 3, 2, 1),
            nn.LeakyReLU(inplace = True), # 8 x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1),
            nn.LeakyReLU(inplace = True), # 16 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1),
            nn.LeakyReLU(inplace = True), # 32 x 4 x 4
        )
        self.linear = nn.Linear(4*4*4*ndf, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.ndf)
        output = self.linear(output)
        return output

start_iteration = 0
netG = Generator()
netD = Discriminator()
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
checkpoints = glob.glob('results/heightfield_gan/chkpt_*.pth.tar')
checkpoints.sort(key=lambda x: os.path.getmtime(x))
fixed_noise = Variable(torch.randn(64, 128))
fixed_xz = []
for i in range(fixed_noise.shape[0]):
    phase = random.random() * 2 * math.pi
    x = 6.0 * math.cos(phase)
    z = 6.0 * math.sin(phase)
    fixed_xz.append((x, z))

if len(checkpoints) > 0:
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint = torch.load(latest_checkpoint)
    start_iteration = latest_checkpoint['iteration']
    netD.load_state_dict(latest_checkpoint['netD.state_dict'])
    netG.load_state_dict(latest_checkpoint['netG.state_dict'])
    optimizerD.load_state_dict(latest_checkpoint['optimizerD'])
    optimizerG.load_state_dict(latest_checkpoint['optimizerG'])
    fixed_noise = latest_checkpoint['fixed_noise']    
    fixed_xz = latest_checkpoint['fixed_xz']

LAMBDA = 10 # Gradient penalty lambda hyperparameter

one = torch.FloatTensor([1])
mone = one * -1

batch_size = 4

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(1)
    alpha = alpha.expand(real_data.nelement()).contiguous().view(batch_size, 1, 32, 32)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(\
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# Training iteration
for iteration in range(start_iteration, 1000000):
    ############################
    # (1) Update D network
    ###########################
    # render a "real" data
    real_img_batch = np.zeros([batch_size, 1, 32, 32], dtype=np.float32)
    for i in range(batch_size):
        height = generate_heightfield(heightfield_res,
            0.2 * random.random() + 0.5,
            0.5 * random.random() + 0.5,
            0.5 * random.random() + 0.5,
            math.pi * random.random(),
            math.pi * random.random())
        shape_plane.vertices = plane_vertices+Variable(torch.from_numpy(height))
        shape_plane.normals = compute_vertex_normal(\
            shape_plane.vertices, shape_plane.indices)
        phase = random.random() * 2 * math.pi
        x = 6.0 * math.cos(phase)
        z = 6.0 * math.sin(phase)
        cam = camera.Camera(\
                    position     = Variable(torch.from_numpy(np.array([x, 3, z], dtype=np.float32))),
                    look_at      = Variable(torch.from_numpy(np.array([0, 0,  0], dtype=np.float32))),
                    up           = Variable(torch.from_numpy(np.array([0, 1,  0], dtype=np.float32))),
                    cam_to_world = None,
                    fov          = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32))),
                    clip_near    = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32))),
                    clip_far     = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32))),
                    resolution   = resolution)
        args = render_pytorch.RenderFunction.serialize_scene(\
            cam,materials,shapes,lights,resolution,4,1)
        render = render_pytorch.RenderFunction.apply
        real_img = render(random.randint(0, 1048576), *args)
        real_img = real_img.permute([2, 1, 0])
        real_img_batch[i, :, :, :] = real_img[0, :, :]
    real_img_batch = Variable(torch.from_numpy(real_img_batch))

    netD.zero_grad()
    # real data cost
    D_real = netD(real_img_batch)
    D_real = D_real.mean()
    D_real.backward(mone)

    # fake data cost
    # generate fake data
    noise = torch.randn(batch_size, 128)
    netG.xz = []
    for i in range(batch_size):
        phase = random.random() * 2 * math.pi
        x = 6.0 * math.cos(phase)
        z = 6.0 * math.sin(phase)
        netG.xz.append((x, z))
    fake = netG(noise)
    D_fake = netD(fake)
    D_fake = D_fake.mean()
    D_fake.backward(one)
    gradient_penalty = calc_gradient_penalty(netD, real_img_batch.data, fake.data)
    gradient_penalty.backward()

    D_cost = D_fake - D_real + gradient_penalty
    print('D fake:', D_fake.item())
    print('D real:', D_real.item())
    print('GP:', gradient_penalty.item())
    print('D cost:', D_cost.item())

    optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    netG.zero_grad()
    fake = netG(noise)
    G = netD(fake).mean()
    G_cost = -G
    print('G cost:', G_cost.item())

    G.backward(mone)
    optimizerG.step()

    if iteration % 100 == 0:
        netG.xz = fixed_xz
        netG.iteration = iteration
        netG.save_heightfield = True
        fake = netG(fixed_noise).data.numpy()
        netG.save_heightfield = False
        fake_flatten = np.zeros([32 * 8, 32 * 8, 1])
        for i in range(8):
            for j in range(8):
                img = fake[8 * i + j, :, :, :].transpose([2, 1, 0])
                fake_flatten[32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1), :] = img
        if np.any(np.isnan(fake_flatten)):
            print('NANNANNAN')
            exit()
        image.imwrite(fake_flatten.squeeze(),
            'results/heightfield_gan/generated_%06d.png' % iteration)

        #netG.resolution = [256, 256]
        #netG.x = 0.0
        #netG.z = -6.0
        #fake = netG(fixed_noise)
        #netG.resolution = resolution
        #image.imwrite(fake.data.numpy(),
        #    'results/heightfield_gan/highres_%06d.exr' % iteration)

        # render a "real" data
        #height = generate_heightfield(heightfield_res,
        #    0.2 * random.random() + 0.5,
        #    0.5 * random.random() + 0.5,
        #    0.5 * random.random() + 0.5,
        #    math.pi * random.random(),
        #    math.pi * random.random())
        #shape_plane.vertices = plane_vertices+height
        #shape_plane.normals = compute_vertex_normal(shape_plane.vertices, shape_plane.indices)
        #phase = random.random() * 2 * math.pi
        #x = 6.0 * math.cos(phase)
        #z = 6.0 * math.sin(phase)
        #cam = camera.Camera(position = np.array([x, 3, z], dtype=np.float32),
        #            look_at      = np.array([0, 0,  0], dtype=np.float32),
        #            up           = np.array([0, 1,  0], dtype=np.float32),
        #            cam_to_world = None,
        #            fov          = 45.0,
        #            clip_near    = 0.01,
        #            clip_far     = 10000.0,
        #            resolution   = [256, 256])
        #args = render_pytorch.RenderFunction.serialize_scene(\
        #    cam,materials,shapes,lights,[256, 256],4,1)
        #render = render_pytorch.RenderFunction.apply
        #real_img = render(random.randint(0, 1048576), *args)
        #real_img = height.view(32, 32, 3)
        #image.imwrite(real_img.data.numpy(),
        #    'results/heightfield_gan/highres_real_%06d.exr' % iteration)
        #cam = camera.Camera(position = np.array([x, 3, z], dtype=np.float32),
        #            look_at      = np.array([0, 0,  0], dtype=np.float32),
        #            up           = np.array([0, 1,  0], dtype=np.float32),
        #            cam_to_world = None,
        #            fov          = 45.0,
        #            clip_near    = 0.01,
        #            clip_far     = 10000.0,
        #            resolution   = resolution)
        #args = render_pytorch.RenderFunction.serialize_scene(\
        #    cam,materials,shapes,lights,resolution,4,1)
        #render = render_pytorch.RenderFunction.apply
        #real_img = render(random.randint(0, 1048576), *args)
        #image.imwrite(real_img.data.numpy(),
        #    'results/heightfield_gan/real_%06d.exr' % iteration)

        torch.save({'iteration': iteration,
                    'netD.state_dict': netD.state_dict(),
                    'netG.state_dict': netG.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'fixed_noise': fixed_noise,
                    'fixed_xz': fixed_xz},
                    'results/heightfield_gan/chkpt_%06d.pth.tar' % iteration)
