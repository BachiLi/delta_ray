import torch
from torch.autograd import Variable
import delta_ray
import numpy as np
import image

class RenderFunction(torch.autograd.Function):
    """
        Render a scene
    """

    @staticmethod
    def serialize_scene(cam,
                        materials,
                        shapes,
                        lights,
                        resolution,
                        num_samples,
                        max_bounces):
        num_materials = len(materials)
        num_shapes = len(shapes)
        num_lights = len(lights)
        args = []
        args.append(num_materials)
        args.append(num_shapes)
        args.append(num_lights)
        args.append(cam.cam_to_world)
        args.append(cam.world_to_cam)
        args.append(cam.sample_to_cam)
        args.append(cam.cam_to_sample)
        args.append(cam.fov_factor.item())
        args.append(cam.aspect_ratio)
        args.append(cam.clip_near.item())
        args.append(cam.fisheye)
        for material in materials:
            args.append(material.diffuse_reflectance)
            args.append(material.specular_reflectance)
            args.append(material.roughness)
            args.append(material.diffuse_uv_scale)
            args.append(material.specular_uv_scale)
            args.append(material.roughness_uv_scale)
            args.append(material.two_sided)
        for shape in shapes:
            args.append(shape.vertices)
            args.append(shape.indices)
            args.append(shape.uvs)
            args.append(shape.normals)
            args.append(shape.mat_id)
        for light in lights:
            args.append(light.shape_id)
            args.append(light.intensity)
        args.append(resolution)
        args.append(num_samples)
        args.append(max_bounces)
        return args

    @staticmethod
    def forward(ctx,
                seed,
                *args):
        # Unpack arguments
        current_index = 0
        num_materials = args[current_index]
        current_index += 1
        num_shapes = args[current_index]
        current_index += 1
        num_lights = args[current_index]
        current_index += 1
        cam_to_world = args[current_index]
        current_index += 1
        world_to_cam = args[current_index]
        current_index += 1
        sample_to_cam = args[current_index]
        current_index += 1
        cam_to_sample = args[current_index]
        current_index += 1
        fov_factor = args[current_index]
        current_index += 1
        aspect_ratio = args[current_index]
        current_index += 1
        clip_near = args[current_index]
        current_index += 1
        fisheye = args[current_index]
        current_index += 1
        diffuse_reflectance_list = []
        specular_reflectance_list = []
        roughness_list = []
        diffuse_uv_scale_list = []
        specular_uv_scale_list = []
        roughness_uv_scale_list = []
        two_sided_list = []
        for i in range(num_materials):
            diffuse_reflectance_list.append(args[current_index])
            current_index += 1
            specular_reflectance_list.append(args[current_index])
            current_index += 1
            roughness_list.append(args[current_index])
            current_index += 1
            diffuse_uv_scale_list.append(args[current_index])
            current_index += 1
            specular_uv_scale_list.append(args[current_index])
            current_index += 1
            roughness_uv_scale_list.append(args[current_index])
            current_index += 1
            two_sided_list.append(args[current_index])
            current_index += 1
        vertices_list = []
        indices_list = []
        uvs_list = []
        normals_list = []
        material_id_list = []
        for i in range(num_shapes):
            vertices_list.append(args[current_index])
            current_index += 1
            indices_list.append(args[current_index])
            current_index += 1
            uvs_list.append(args[current_index])
            current_index += 1
            normals_list.append(args[current_index])
            current_index += 1
            material_id_list.append(args[current_index])
            current_index += 1
        light_shape_id_list = []
        light_intensity_list = []
        for i in range(num_lights):
            light_shape_id_list.append(args[current_index])
            current_index += 1
            light_intensity_list.append(args[current_index])
            current_index += 1
        resolution = args[current_index]
        current_index += 1
        num_samples = args[current_index]
        current_index += 1
        max_bounces = args[current_index]
        current_index += 1

        cam = delta_ray.Camera(cam_to_world.data.numpy(),
                               world_to_cam.data.numpy(),
                               sample_to_cam.data.numpy(),
                               cam_to_sample.data.numpy(),
                               fov_factor,
                               aspect_ratio,
                               clip_near,
                               fisheye)
        materials = []
        for diffuse_reflectance, specular_reflectance, roughness, \
                diffuse_uv_scale, specular_uv_scale, roughness_uv_scale, two_sided in \
                zip(diffuse_reflectance_list, specular_reflectance_list,
                    roughness_list, diffuse_uv_scale_list, specular_uv_scale_list,
                    roughness_uv_scale_list, two_sided_list):
            materials.append(delta_ray.Material(\
                diffuse_reflectance.data.numpy(),
                specular_reflectance.data.numpy(),
                roughness.data.numpy(),
                diffuse_uv_scale.data.numpy(),
                specular_uv_scale.data.numpy(),
                roughness_uv_scale.data.numpy(),
                two_sided))

        shapes = []
        for vertices, indices, uvs, normals, material_id in \
                zip(vertices_list, indices_list, uvs_list, normals_list, material_id_list):
            mat = materials[material_id]
            if uvs is not None:
                uvs = uvs.numpy()
            if normals is not None:
                normals = normals.data.numpy()
            shapes.append(delta_ray.Shape(\
                vertices.data.numpy(), indices.data.numpy(), uvs, normals, mat, None))

        lights = []
        for light_shape_id, light_intensity in zip(light_shape_id_list, light_intensity_list):
            light_mesh = shapes[light_shape_id]
            light = delta_ray.Light(light_mesh,
                                    light_intensity.data.numpy())
            light_mesh.light = light
            lights.append(light)

        # d_img = np.ones([resolution[1], resolution[0], 3], dtype=np.float32)
        d_img = np.array(0.0, dtype=np.float32)

        print('forward pass')
        result = \
            delta_ray.render(cam,
                             shapes,
                             materials,
                             lights,
                             resolution,
                             d_img,
                             num_samples,
                             max_bounces,
                             seed,
                             True)
        if False:
            import matplotlib.cm as cm
            dx = result.dx_image
            image.imwrite(dx, 'dx.exr')

            #width = 0.02
            #dx = np.clip(dx, -width, width)
            #dx = (dx + width) / (2.0 * width)
            #dx = cm.viridis(dx[:, :, 0])
            #image.imwrite(dx, 'dx.png')
            exit()

        # dy = result.dy_image
        # print('max(dy):', np.max(dy))
        # print('min(dy):', np.min(dy))
        # print('sum(dy):', np.sum(dy))
        # dy = dy# / np.max(dy)
        # image.imwrite(dy, 'fwd_dy.exr')
        # dy = -dy# / np.min(dy)
        # image.imwrite(dy, 'fwd_inv_dy.exr')
        # dx = result.dx_image
        # print('max(dx):', np.max(dx))
        # print('min(dx):', np.min(dx))
        # print('sum(dx):', np.sum(dx))
        # dx = dx# / np.max(dx)
        # image.imwrite(dx, 'fwd_dx.exr')
        # dx = -dx# / np.min(dx)
        # image.imwrite(dx, 'fwd_inv_dx.exr')
        # exit()

        ctx.cam = cam
        ctx.shapes = shapes
        ctx.materials = materials
        ctx.lights = lights
        ctx.resolution = resolution
        ctx.num_samples = num_samples
        ctx.max_bounces = max_bounces
        ctx.seed = seed
        img = torch.from_numpy(result.image)
        return img

    @staticmethod
    def backward(ctx, grad_img):
        cam = ctx.cam
        shapes = ctx.shapes
        materials = ctx.materials
        lights = ctx.lights
        resolution = ctx.resolution
        num_samples = ctx.num_samples
        max_bounces = ctx.max_bounces
        seed = ctx.seed

        print('backward pass')
        result = \
            delta_ray.render(cam,
                             shapes,
                             materials,
                             lights,
                             resolution,
                             grad_img.data.numpy(),
                             num_samples,
                             max_bounces,
                             seed,
                             True)
        if False:
            image.imwrite(result.image, 'img.exr')
            n = grad_img.data.numpy().copy()
            n = n / np.max(n)
            image.imwrite(n, 'grad_img.exr')
            n = n / np.min(n)
            image.imwrite(n, 'inv_grad_img.exr')
            #dy = result.dy_image
            #print('max(dy):', np.max(dy))
            #print('min(dy):', np.min(dy))
            #print('sum(dy):', np.sum(dy))
            #dy = dy / np.max(dy)
            #image.imwrite(dy, 'dy.exr')
            #dy = dy / np.min(dy)
            #image.imwrite(dy, 'inv_dy.exr')
            dx = result.dx_image
            print('max(dx):', np.max(dx))
            print('min(dx):', np.min(dx))
            print('sum(dx):', np.sum(dx))
            dx = dx# / np.max(dx)
            image.imwrite(dx, 'dx.exr')
            dx = -dx# / np.min(dx)
            image.imwrite(dx, 'inv_dx.exr')
            exit()

        ret_list = []
        ret_list.append(None) # seed
        ret_list.append(None) # num_materials
        ret_list.append(None) # num_shapes
        ret_list.append(None) # num_lights
        ret_list.append(Variable(torch.from_numpy(\
            result.d_camera.d_cam_to_world))) # cam_to_world
        ret_list.append(Variable(torch.from_numpy(\
            result.d_camera.d_world_to_cam))) # world_to_cam
        ret_list.append(Variable(torch.from_numpy(\
            result.d_camera.d_sample_to_cam))) # sample_to_cam
        ret_list.append(Variable(torch.from_numpy(\
            result.d_camera.d_cam_to_sample))) # cam_to_sample
        ret_list.append(None) # fov_factor
        ret_list.append(None) # aspect_ratio
        ret_list.append(None) # clip_near
        ret_list.append(None) # fisheye
        for d_material in result.d_materials:
            d_diffuse = Variable(torch.from_numpy(d_material.diffuse_reflectance))
            d_specular = Variable(torch.from_numpy(d_material.specular_reflectance))
            d_roughness = Variable(torch.from_numpy(d_material.roughness))
            d_diffuse_uv_scale = Variable(torch.from_numpy(d_material.diffuse_uv_scale))
            d_specular_uv_scale = Variable(torch.from_numpy(d_material.specular_uv_scale))
            d_roughness_uv_scale = Variable(torch.from_numpy(d_material.roughness_uv_scale))
            ret_list.append(d_diffuse) # diffuse_reflection
            ret_list.append(d_specular) # specular_reflection
            ret_list.append(d_roughness) # roughness
            ret_list.append(d_diffuse_uv_scale)
            ret_list.append(d_specular_uv_scale)
            ret_list.append(d_roughness_uv_scale)
            ret_list.append(None) # two-sided
        for d_shape in result.d_shapes:
            d_vertices = Variable(torch.from_numpy(d_shape.vertices))
            ret_list.append(d_vertices) # vertices
            ret_list.append(None) # indices
            ret_list.append(None) # uvs
            if d_shape.normals.ndim != 2:
                ret_list.append(None) # normal
            else:
                d_normals = Variable(torch.from_numpy(d_shape.normals))
                ret_list.append(d_normals) # normal
            ret_list.append(None) # material id
        for d_light in result.d_lights:
            ret_list.append(None) # light shape id
            # intensity
            ret_list.append(Variable(torch.from_numpy(d_light.intensity)))
        ret_list.append(None) # resolution
        ret_list.append(None) # num_samples
        ret_list.append(None) # max_bounces

        return tuple(ret_list)
