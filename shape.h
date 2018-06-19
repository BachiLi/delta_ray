#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Material;
struct Light;

struct Shape {
    py::array_t<float> vertices;
    py::array_t<int> indices;
    py::array_t<float> uvs;
    py::array_t<float> normals;
    const Material *material;
    const Light *light;
};

struct DShape {
    py::array_t<float> vertices;
    py::array_t<float> normals;
};

inline auto get_vertex(const Shape &shape, int index) {
    auto vertices_accessor = shape.vertices.unchecked<2>();
    return make_vector3(vertices_accessor(index, 0),
                        vertices_accessor(index, 1),
                        vertices_accessor(index, 2));
}

inline auto get_indices(const Shape &shape, int tri_index) {
    auto indices_accessor = shape.indices.unchecked<2>();
    return make_vector3(indices_accessor(tri_index, 0),
                        indices_accessor(tri_index, 1),
                        indices_accessor(tri_index, 2));
}

inline auto has_uvs(const Shape &shape) {
    return shape.uvs.ndim() == 2;
}

inline auto get_uv(const Shape &shape, int vertex_index) {
    auto uv_accessor = shape.uvs.unchecked<2>();
    return make_vector2(uv_accessor(vertex_index, 0),
                        uv_accessor(vertex_index, 1));
}

inline auto has_shading_normals(const Shape &shape) {
    return shape.normals.ndim() == 2;
}

inline auto get_shading_normal(const Shape &shape, int vertex_index) {
    auto normal_accessor = shape.normals.unchecked<2>();
    return make_vector3(normal_accessor(vertex_index, 0),
                        normal_accessor(vertex_index, 1),
                        normal_accessor(vertex_index, 2));
}

inline auto num_vertices(const Shape &shape) {
    return shape.vertices.shape()[0];
}

inline auto num_triangles(const Shape &shape) {
    return shape.indices.shape()[0];
}

inline auto get_normal(const Shape &shape, int tri_index) {
    auto indices = get_indices(shape, tri_index);
    auto v0 = get_vertex(shape, std::get<0>(indices));
    auto v1 = get_vertex(shape, std::get<1>(indices));
    auto v2 = get_vertex(shape, std::get<2>(indices));
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    return normalize(cross(e1, e2));
}

inline void accumulate_derivative(
        DShape &d_shape, int vertex_id, const Vector3f &dv) {
    auto vertices_accessor = d_shape.vertices.mutable_unchecked<2>();
    vertices_accessor(vertex_id, 0) += dv[0];
    vertices_accessor(vertex_id, 1) += dv[1];
    vertices_accessor(vertex_id, 2) += dv[2];
}


inline void accumulate_derivative(
        DShape &d_shape, int vertex_id, const Vector3f &dv, const Vector3f &dn) {
    auto vertices_accessor = d_shape.vertices.mutable_unchecked<2>();
    vertices_accessor(vertex_id, 0) += dv[0];
    vertices_accessor(vertex_id, 1) += dv[1];
    vertices_accessor(vertex_id, 2) += dv[2];
    if (d_shape.normals.ndim() == 2) {
        auto normal_accessor = d_shape.normals.mutable_unchecked<2>();
        normal_accessor(vertex_id, 0) += dn[0];
        normal_accessor(vertex_id, 1) += dn[1];
        normal_accessor(vertex_id, 2) += dn[2];
    }
}
