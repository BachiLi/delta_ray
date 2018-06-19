import torch
from torch.autograd import Variable
import xml.etree.ElementTree as etree
import numpy as np
import camera
import material
import shape
import light
import load_obj
import load_serialized
import image
import os
import transform

def parse_transform(node):
    ret = np.identity(4, dtype=np.float32)
    for child in node:
        if child.tag == 'matrix':
            value = np.reshape(\
                np.fromstring(child.attrib['value'], dtype=np.float32, sep=' '),
                (4, 4))
            ret = value @ ret
        elif child.tag == 'translate':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = \
                transform.gen_translate_matrix(np.array([x, y, z], dtype=np.float32))
            ret = value @ ret
        elif child.tag == 'scale':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = \
                transform.gen_scale_matrix(np.array([x, y, z], dtype=np.float32))
            ret = value @ ret
    return ret

def parse_vector(str):
    v = np.fromstring(str, dtype=np.float32, sep=',')
    if v.shape[0] != 3:
        v = np.fromstring(str, dtype=np.float32, sep=' ')
    assert(v.ndim == 1)
    return v

def parse_camera(node):
    fov = Variable(torch.from_numpy(np.array([45.0], dtype=np.float32)))
    to_world = np.identity(4, dtype=np.float32)
    position = None
    look_at = None
    up = None
    clip_near = Variable(torch.from_numpy(np.array([0.01], dtype=np.float32)))
    clip_far = Variable(torch.from_numpy(np.array([10000.0], dtype=np.float32)))
    resolution = [256, 256]
    for child in node:
        if 'name' in child.attrib:
            if child.attrib['name'] == 'fov':
                fov = Variable(torch.from_numpy(\
                    np.array([float(child.attrib['value'])], dtype=np.float32)))
            elif child.attrib['name'] == 'toWorld':
                has_lookat = False
                for grandchild in child:
                    if grandchild.tag == 'lookat':
                        has_lookat = True
                        position = Variable(torch.from_numpy(\
                            parse_vector(grandchild.attrib['origin'])))
                        look_at = Variable(torch.from_numpy(\
                            parse_vector(grandchild.attrib['target'])))
                        up = Variable(torch.from_numpy(\
                            parse_vector(grandchild.attrib['up'])))
                if not has_lookat:
                    to_world = parse_transform(child)
        if child.tag == 'film':
            for grandchild in child:
                if 'name' in grandchild.attrib:
                    if grandchild.attrib['name'] == 'width':
                        resolution[0] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'height':
                        resolution[1] = int(grandchild.attrib['value'])
    if position is not None:
        to_world = None
    else:
        to_world = Variable(torch.from_numpy(to_world))

    return camera.Camera(position     = position,
                         look_at      = look_at,
                         up           = up,
                         cam_to_world = to_world,
                         fov          = fov,
                         clip_near    = clip_near,
                         clip_far     = clip_far,
                         resolution   = resolution), resolution

def parse_material(node, two_sided = False):
    node_id = None
    if 'id' in node.attrib:
        node_id = node.attrib['id']
    if node.attrib['type'] == 'diffuse':
        diffuse_reflectance = Variable(torch.from_numpy(\
            np.array([0.5, 0.5, 0.5], dtype=np.float32)))
        diffuse_uv_scale = Variable(torch.from_numpy(\
            np.array([1.0, 1.0], dtype=np.float32)))
        specular_reflectance = Variable(torch.from_numpy(\
            np.array([0.0, 0.0, 0.0], dtype=np.float32)))
        specular_uv_scale = Variable(torch.from_numpy(\
            np.array([1.0, 1.0], dtype=np.float32)))
        roughness = Variable(torch.from_numpy(\
            np.array([1.0], dtype=np.float32)))
        for child in node:
            if child.attrib['name'] == 'reflectance':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            diffuse_reflectance = Variable(torch.from_numpy(\
                                image.imread(grandchild.attrib['value'])))
                        elif grandchild.attrib['name'] == 'uscale':
                            diffuse_uv_scale.data[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            diffuse_uv_scale.data[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb':
                    diffuse_reflectance = \
                        Variable(torch.from_numpy(\
                            parse_vector(child.attrib['value'])))
            elif child.attrib['name'] == 'specular':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            specular_reflectance = Variable(torch.from_numpy(\
                                image.imread(grandchild.attrib['value'])))
                        elif grandchild.attrib['name'] == 'uscale':
                            specular_uv_scale.data[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            specular_uv_scale.data[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb':
                    specular_reflectance = \
                        Variable(torch.from_numpy(\
                            parse_vector(child.attrib['value'])))
            elif child.attrib['name'] == 'roughness':
                roughness = \
                    Variable(torch.from_numpy(\
                        float(child.attrib['value'])))
        return (node_id, material.Material(diffuse_reflectance,
                diffuse_uv_scale = diffuse_uv_scale,
                specular_reflectance = specular_reflectance,
                specular_uv_scale = specular_uv_scale,
                roughness = roughness,
                two_sided = two_sided))
    elif node.attrib['type'] == 'twosided':
        ret = parse_material(node[0], True)
        return (node_id, ret[1])

def parse_shape(node, material_dict, shape_id):
    if node.attrib['type'] == 'obj' or node.attrib['type'] == 'serialized':
        to_world = np.identity(4, dtype=np.float32)
        serialized_shape_id = 0
        mat_id = -1
        light_intensity = None
        filename = ''
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'filename':
                    filename = child.attrib['value']
                elif child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
                elif child.attrib['name'] == 'shapeIndex':
                    serialized_shape_id = int(child.attrib['value'])
            if child.tag == 'ref':
                mat_id = material_dict[child.attrib['id']]
            elif child.tag == 'emitter':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = np.array([light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]], dtype=np.float32)
                        light_intensity = Variable(torch.from_numpy(light_intensity))

        if node.attrib['type'] == 'obj':
            indices, vertices, uvs, normals = load_obj.load_obj(filename)
        else:
            assert(node.attrib['type'] == 'serialized')
            indices, vertices, uvs, normals = \
                load_serialized.load_serialized(filename, serialized_shape_id)
            if uvs.shape[0] == 0:
                uvs = None
            if normals.shape[0] == 0:
                normals = None

        # Transform the vertices and normals
        vertices = \
            np.concatenate(\
                (vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)), 1)
        vertices = vertices @ np.transpose(to_world)
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3]
        if normals is not None:
            normals = normals @ (np.linalg.inv(np.transpose(to_world))[:3, :3])
        assert(vertices is not None)
        assert(indices is not None)
        lgt = None
        if light_intensity is not None:
            lgt = light.Light(shape_id, light_intensity)
        vertices = Variable(torch.from_numpy(vertices))
        indices = Variable(torch.from_numpy(indices))
        if uvs is not None:
            uvs = Variable(torch.from_numpy(uvs))
        if normals is not None:
            normals = Variable(torch.from_numpy(normals))
        return shape.Shape(vertices, indices, uvs, normals, mat_id), lgt
    elif node.attrib['type'] == 'rectangle':
        indices = np.array([[0,2,1],[1,2,3]])
        vertices = np.array([[-1,-1,0],[-1,1,0],[1,-1,0],[1,1,0]],dtype=np.float32)
        uvs = None
        normals = None
        to_world = np.identity(4, dtype=np.float32)
        mat_id = -1
        light_intensity = None
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
            if child.tag == 'ref':
                mat_id = material_dict[child.attrib['id']]
            elif child.tag == 'emitter':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = np.array([light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]], dtype=np.float32)
                        light_intensity = Variable(torch.from_numpy(light_intensity))
        # Transform the vertices
        vertices = \
            np.concatenate(\
                (vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)), 1)
        vertices = vertices @ np.transpose(to_world)
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3]
        if normals is not None:
            normals = normals @ (np.linalg.inv(np.transpose(to_world))[:3, :3])
        assert(vertices is not None)
        assert(indices is not None)
        lgt = None
        if light_intensity is not None:
            lgt = light.Light(shape_id, light_intensity)
        vertices = Variable(torch.from_numpy(vertices))
        indices = Variable(torch.from_numpy(indices))
        if uvs is not None:
            uvs = Variable(torch.from_numpy(uvs))
        if normals is not None:
            normals = Variable(torch.from_numpy(normals))
        return shape.Shape(vertices, indices, uvs, normals, mat_id), lgt
    else:
        assert(False)

def parse_scene(node):
    cam = None
    resolution = None
    materials = []
    material_dict = {}
    shapes = []
    lights = []
    for child in node:
        if child.tag == 'sensor':
            cam, resolution = parse_camera(child)
        elif child.tag == 'bsdf':
            node_id, material = parse_material(child)
            if node_id is not None:
                material_dict[node_id] = len(materials)
                materials.append(material)
        elif child.tag == 'shape':
            shape, light = parse_shape(child, material_dict, len(shapes))
            shapes.append(shape)
            if light is not None:
                lights.append(light)
    return cam, materials, shapes, lights, resolution

def load_mitsuba(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(filename))
    ret = parse_scene(root)
    os.chdir(cwd)
    return ret
