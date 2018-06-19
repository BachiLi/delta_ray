#pragma once

#include "vector.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template <typename T>
inline auto xfm_point(const pybind11::array_t<float> &xform,
                      const TVector3<T> &pt) {
    auto acc = xform.unchecked<2>();
    auto tpt = make_vector4(
        acc(0, 0) * pt[0] + acc(0, 1) * pt[1] + acc(0, 2) * pt[2] + acc(0, 3),
        acc(1, 0) * pt[0] + acc(1, 1) * pt[1] + acc(1, 2) * pt[2] + acc(1, 3),
        acc(2, 0) * pt[0] + acc(2, 1) * pt[1] + acc(2, 2) * pt[2] + acc(2, 3),
        acc(3, 0) * pt[0] + acc(3, 1) * pt[1] + acc(3, 2) * pt[2] + acc(3, 3));
    auto inv_w = 1.f / tpt[3];
    return make_vector3(tpt[0], tpt[1], tpt[2]) * inv_w;
}

inline void d_xfm_point(const pybind11::array_t<float> &xform,
                        const Vector3f &pt,
                        const Vector3f &d_out,
                        Matrix4x4f &d_xform) {
    auto acc = xform.unchecked<2>();
    auto tpt = make_vector4(
        acc(0, 0) * pt[0] + acc(0, 1) * pt[1] + acc(0, 2) * pt[2] + acc(0, 3),
        acc(1, 0) * pt[0] + acc(1, 1) * pt[1] + acc(1, 2) * pt[2] + acc(1, 3),
        acc(2, 0) * pt[0] + acc(2, 1) * pt[1] + acc(2, 2) * pt[2] + acc(2, 3),
        acc(3, 0) * pt[0] + acc(3, 1) * pt[1] + acc(3, 2) * pt[2] + acc(3, 3));
    auto inv_w = 1.f / tpt[3];
    // backprop ret = make_vector3(tpt[0], tpt[1], tpt[2]) * inv_w;
    auto dtpt0 = d_out[0] * inv_w;
    auto dtpt1 = d_out[1] * inv_w;
    auto dtpt2 = d_out[2] * inv_w;
    auto dinv_w = d_out[0] * tpt[0] + d_out[1] * tpt[1] + d_out[2] * tpt[2];
    // backprop inv_w = 1.f / tpt[3];
    auto dtpt3 = - dinv_w * inv_w * inv_w;
    // backprop tpt0 = acc(0, 0) * pt[0] + acc(0, 1) * pt[1] + acc(0, 2) * pt[2] + acc(0, 3)
    d_xform(0, 0) += dtpt0 * pt[0];
    d_xform(0, 1) += dtpt0 * pt[1];
    d_xform(0, 2) += dtpt0 * pt[2];
    d_xform(0, 3) += dtpt0;
    // backprop tpt1 = acc(1, 0) * pt[0] + acc(1, 1) * pt[1] + acc(1, 2) * pt[2] + acc(1, 3)
    d_xform(1, 0) += dtpt1 * pt[0];
    d_xform(1, 1) += dtpt1 * pt[1];
    d_xform(1, 2) += dtpt1 * pt[2];
    d_xform(1, 3) += dtpt1;
    // backprop tpt2 = acc(2, 0) * pt[0] + acc(2, 1) * pt[1] + acc(2, 2) * pt[2] + acc(2, 3)
    d_xform(2, 0) += dtpt2 * pt[0];
    d_xform(2, 1) += dtpt2 * pt[1];
    d_xform(2, 2) += dtpt2 * pt[2];
    d_xform(2, 3) += dtpt2;
    // backprop tpt3 = acc(3, 0) * pt[0] + acc(3, 1) * pt[1] + acc(3, 2) * pt[2] + acc(3, 3)
    d_xform(3, 0) += dtpt3 * pt[0];
    d_xform(3, 1) += dtpt3 * pt[1];
    d_xform(3, 2) += dtpt3 * pt[2];
    d_xform(3, 3) += dtpt3;
}

template <typename T>
inline auto xfm_point(const TMatrix4x4<T> &xform,
                      const TVector3<T> &pt) {
    auto tpt = make_vector4(
        xform(0, 0) * pt[0] + xform(0, 1) * pt[1] + xform(0, 2) * pt[2] + xform(0, 3),
        xform(1, 0) * pt[0] + xform(1, 1) * pt[1] + xform(1, 2) * pt[2] + xform(1, 3),
        xform(2, 0) * pt[0] + xform(2, 1) * pt[1] + xform(2, 2) * pt[2] + xform(2, 3),
        xform(3, 0) * pt[0] + xform(3, 1) * pt[1] + xform(3, 2) * pt[2] + xform(3, 3));
    auto inv_w = 1.f / tpt[3];
    return make_vector3(tpt[0], tpt[1], tpt[2]) * inv_w;
}

template <typename T>
inline auto xfm_vector(const pybind11::array_t<float> &xform,
                       const TVector3<T> &vec) {
    auto acc = xform.unchecked<2>();
    return make_vector3(
        acc(0, 0) * vec[0] + acc(0, 1) * vec[1] + acc(0, 2) * vec[2],
        acc(1, 0) * vec[0] + acc(1, 1) * vec[1] + acc(1, 2) * vec[2],
        acc(2, 0) * vec[0] + acc(2, 1) * vec[1] + acc(2, 2) * vec[2]);
}

template <typename T>
inline auto xfm_vector(const TMatrix4x4<T> &xform,
                       const TVector3<T> &vec) {
    return make_vector3(
        xform(0, 0) * vec[0] + xform(0, 1) * vec[1] + xform(0, 2) * vec[2],
        xform(1, 0) * vec[0] + xform(1, 1) * vec[1] + xform(1, 2) * vec[2],
        xform(2, 0) * vec[0] + xform(2, 1) * vec[1] + xform(2, 2) * vec[2]);
}

template <typename T>
inline auto xfm_normal(const pybind11::array_t<float> &inv_xform,
                       const TVector3<T> &vec) {
    auto acc = inv_xform.unchecked<2>();
    return TVector3<T>(
        acc(0, 0) * vec[0] + acc(1, 0) * vec[1] + acc(2, 0) * vec[2],
        acc(0, 1) * vec[0] + acc(1, 1) * vec[1] + acc(2, 1) * vec[2],
        acc(0, 2) * vec[0] + acc(1, 2) * vec[1] + acc(2, 2) * vec[2]);
}

template <typename T>
inline auto xfm_normal(const TMatrix4x4<T> &inv_xform,
                       const TVector3<T> &vec) {
    return TVector3<T>(
        inv_xform(0, 0) * vec[0] + inv_xform(1, 0) * vec[1] + inv_xform(2, 0) * vec[2],
        inv_xform(0, 1) * vec[0] + inv_xform(1, 1) * vec[1] + inv_xform(2, 1) * vec[2],
        inv_xform(0, 2) * vec[0] + inv_xform(1, 2) * vec[1] + inv_xform(2, 2) * vec[2]);
}

template <typename T>
inline auto rotate(const TVector3<T> &v,
                   const T &angle) {
    auto c = cos(angle);
    auto s = sin(angle);
    return TMatrix4x4<T>(
        c+(1-c)*v[0]*v[0],(1-c)*v[0]*v[1]-v[2]*s,(1-c)*v[0]*v[2]+v[1]*s,0,
        (1-c)*v[0]*v[1]+v[2]*s,c+(1-c)*v[1]*v[1],(1-c)*v[1]*v[2]-v[0]*s,0,
        (1-c)*v[0]*v[2]-v[1]*s,(1-c)*v[1]*v[2]+v[0]*s,c+(1-c)*v[2]*v[2],0,
        0,0,0,1);
}
