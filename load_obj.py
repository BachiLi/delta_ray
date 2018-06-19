import torch
from torch.autograd import Variable
import numpy as np
import re

def load_obj(filename):
    """
        load from a Wavefront obj file 
    """

    vertices_pool = []
    uvs_pool = []
    normals_pool = []
    indices = []
    vertices = []
    normals = []
    uvs = []
    vertices_map = {}
    for line in open(filename, 'r'):
        line = line.strip()
        splitted = re.split('\ +', line)
        if splitted[0] == 'v':
            vertices_pool.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
        elif splitted[0] == 'vt':
            uvs_pool.append([float(splitted[1]), float(splitted[2])])
        elif splitted[0] == 'vn':
            normals_pool.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
        elif splitted[0] == 'f':
            def num_indices(x):
                return len(re.split('/', x))
            def get_index(x, i):
                return int(re.split('/', x)[i])
            def parse_face_index(x, i):
                f = get_index(x, i)
                if f < 0:
                    if (i == 0):
                        f += len(vertices)
                    if (i == 1):
                        f += len(uvs)
                else:
                    f -= 1
                return f
            assert(len(splitted) <= 5)
            def get_vertex_id(indices):
                pi = parse_face_index(indices, 0)
                uvi = None
                if (num_indices(indices) > 1 and re.split('/', indices)[1] != ''):
                    uvi = parse_face_index(indices, 1)
                ni = None
                if (num_indices(indices) > 2 and re.split('/', indices)[2] != ''):
                    ni = parse_face_index(indices, 2)
                key = (pi, uvi, ni)
                if key in vertices_map:
                    return vertices_map[key]

                vertex_id = len(vertices)
                vertices_map[key] = vertex_id
                vertices.append(vertices_pool[pi])
                if uvi is not None:
                    uvs.append(uvs_pool[uvi])
                if ni is not None:
                    normals.append(normals_pool[ni])
                return vertex_id
            vid0 = get_vertex_id(splitted[1])
            vid1 = get_vertex_id(splitted[2])
            vid2 = get_vertex_id(splitted[3])

            indices.append([vid0, vid1, vid2])
            if (len(splitted) == 5):
                vid3 = get_vertex_id(splitted[4])
                indices.append([vid0, vid2, vid3])
    
    indices = np.array(indices, dtype=np.int32)
    vertices = np.array(vertices, dtype=np.float32)
    if len(uvs) == 0:
        uvs = None
    else:
        uvs = np.array(uvs, dtype=np.float32)
    if len(normals) == 0:
        normals = None
    else:
        normals = np.array(normals, dtype=np.float32)

    return (indices, vertices, uvs, normals)