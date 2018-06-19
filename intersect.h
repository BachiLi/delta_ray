#pragma once

#include "vector.h"
#include "shape.h"

struct Intersection {
    const Shape *shape = nullptr;
    int tri_id;

    bool valid() const {
        return shape != nullptr;
    }
};

template <typename T>
struct TSurfacePoint {
    TVector3<T> position;
    TVector3<T> geom_normal;
    TFrame<T> shading_frame;
    TVector2<T> uv;
};

template <typename T>
struct TRay {
    TVector3<T> org;
    TVector3<T> dir;
};

template <typename T>
struct TCone {
    TRay<T> ray;
    T cos_angle;
};

using SurfacePoint = TSurfacePoint<float>;
using Ray = TRay<float>;
using Cone = TCone<float>;

template <typename T>
inline auto make_surface_point() {
    return TSurfacePoint<T>{};
}

template <typename T>
inline auto make_surface_point(
        const TVector3<T> &position,
        const TVector3<T> &geom_normal,
        const TFrame<T>   &shading_frame,
        const TVector2<T> &uv) {
    return TSurfacePoint<T>{position, geom_normal, shading_frame, uv};
}

template <typename T>
inline auto make_ray(const TVector3<T> &org, const TVector3<T> &dir) {
    return TRay<T>{org, dir};
}

template <typename TOut, typename TIn>
inline TSurfacePoint<TOut> convert(const TSurfacePoint<TIn> &v) {
    return make_surface_point(convert<TOut>(v.position),
                              convert<TOut>(v.geom_normal),
                              convert<TOut>(v.shading_frame),
                              convert<TOut>(v.uv));
}

template <typename TOut, typename TIn>
inline TRay<TOut> convert(const TRay<TIn> &v) {
    return make_ray(convert<TOut>(v.org), convert<TOut>(v.dir));
}

template <typename T0, typename T1, typename T2, typename T3>
inline auto intersect(const TVector3<T0> &v0,
                      const TVector3<T1> &v1,
                      const TVector3<T2> &v2,
                      const TRay<T3> &ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = cross(ray.dir, e2);
    auto divisor = dot(pvec, e1);
    if (divisor == 0.f) {
        // XXX HACK!!! XXX
        divisor = 1e-8f;
    }
    auto s = ray.org - v0;
    auto u = dot(s, pvec) / divisor;
    auto qvec = cross(s, e1);
    auto v = dot(ray.dir, qvec) / divisor;
    auto t = dot(e2, qvec) / divisor;
    return std::make_tuple(u, v, t);
}

template <typename T>
struct IntersectParameter {
    const Shape *shape = nullptr;
    int tri_id;
    TVector3<T> v0, v1, v2; // positions
    TVector3<T> n0, n1, n2; // shading normal
};

template <typename T>
inline auto intersect(const Intersection &isect,
                      const TRay<T> &ray,
                      std::vector<IntersectParameter<T>> *params = nullptr) {
    const Shape &shape = *isect.shape;
    auto ind = get_indices(shape, isect.tri_id);
    auto v0 = convert<T>(get_vertex(shape, ind[0]));
    auto v1 = convert<T>(get_vertex(shape, ind[1]));
    auto v2 = convert<T>(get_vertex(shape, ind[2]));
    TVector2<T> uvs0, uvs1, uvs2;
    if (has_uvs(shape)) {
        uvs0 = convert<T>(get_uv(shape, ind[0]));
        uvs1 = convert<T>(get_uv(shape, ind[1]));
        uvs2 = convert<T>(get_uv(shape, ind[2]));
    } else {
        uvs0 = make_vector2(T(0), T(0));
        uvs1 = make_vector2(T(1), T(0));
        uvs2 = make_vector2(T(1), T(1));
    }
    if (params != nullptr) {
        params->push_back(IntersectParameter<T>{&shape, isect.tri_id, v0, v1, v2});
    }
    auto [u, v, t] = intersect(v0, v1, v2, ray);
    auto w = 1.f - (u + v);
    auto uv = w * uvs0 + u * uvs1 + v * uvs2;
    auto hit_pos = ray.org + ray.dir * t;
    auto geom_normal = normalize(cross(v1 - v0, v2 - v0));
    auto shading_normal = geom_normal;
    if (has_shading_normals(shape)) {
        auto n0 = convert<T>(get_shading_normal(shape, ind[0]));
        auto n1 = convert<T>(get_shading_normal(shape, ind[1]));
        auto n2 = convert<T>(get_shading_normal(shape, ind[2]));
        if (params != nullptr) {
            params->back().n0 = n0;
            params->back().n1 = n1;
            params->back().n2 = n2;
        }
        shading_normal = normalize(w * n0 + u * n1 + v * n2);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    return make_surface_point(
        convert<T>(hit_pos),
        convert<T>(geom_normal),
        make_frame(convert<T>(shading_normal)),
        convert<T>(uv));
}
