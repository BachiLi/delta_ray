#pragma once

#include "autodiff.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <iostream>

using namespace autodiff;
using std::sqrt;
using std::sin;
using std::cos;
using std::isnan;

template <typename T>
using TVector2 = std::array<T, 2>;
template <typename T>
using TVector3 = std::array<T, 3>;
template <typename T>
using TVector4 = std::array<T, 4>;
using Vector2f = TVector2<float>;
using Vector2i = TVector2<int>;
using Vector3f = TVector3<float>;
using Vector4f = TVector4<float>;
using Vector2a = TVector2<AReal>;
using Vector3a = TVector3<AReal>;
using Vector4a = TVector4<AReal>;

template <typename T>
using TFrame = std::array<TVector3<T>, 3>;
using Framef = TFrame<float>;

template <typename T>
struct TMatrix2x2 {
    const T& operator()(int i, int j) const {
        return data[i][j];
    }
    T& operator()(int i, int j) {
        return data[i][j];
    }
    T data[2][2];
};
template <typename T>
struct TMatrix3x3 {
    const T& operator()(int i, int j) const {
        return data[i][j];
    }
    T& operator()(int i, int j) {
        return data[i][j];
    }
    T data[3][3];
};
template <typename T>
struct TMatrix4x4 {
    TMatrix4x4() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = T(0);
            }
        }
    }
    TMatrix4x4(const pybind11::array_t<float> &mat) {
        auto acc = mat.unchecked<2>();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = acc(i, j);
            }
        }
    }
    TMatrix4x4(T v00, T v01, T v02, T v03,
               T v10, T v11, T v12, T v13,
               T v20, T v21, T v22, T v23,
               T v30, T v31, T v32, T v33) {
        data[0][0] = v00;
        data[0][1] = v01;
        data[0][2] = v02;
        data[0][3] = v03;
        data[1][0] = v10;
        data[1][1] = v11;
        data[1][2] = v12;
        data[1][3] = v13;
        data[2][0] = v20;
        data[2][1] = v21;
        data[2][2] = v22;
        data[2][3] = v23;
        data[3][0] = v30;
        data[3][1] = v31;
        data[3][2] = v32;
        data[3][3] = v33;
    }
    const T& operator()(int i, int j) const {
        return data[i][j];
    }
    T& operator()(int i, int j) {
        return data[i][j];
    }
    T data[4][4];
};

using Matrix2x2f = TMatrix2x2<float>;
using Matrix3x3f = TMatrix3x3<float>;
using Matrix4x4f = TMatrix4x4<float>;

template <typename T>
inline auto make_vector2(const T &x, const T &y) {
    return TVector2<T>{{x, y}};
}

template <typename T>
inline auto make_vector2(const pybind11::array_t<float> &arr) {
    auto acc = arr.unchecked<1>();
    return make_vector2(T(acc(0)), T(acc(1)));
}

template <typename T>
inline auto make_vector3(const T &x, const T &y, const T &z) {
    return TVector3<T>{{x, y, z}};
}

inline auto make_vector3(const pybind11::array_t<float> &arr) {
    auto acc = arr.unchecked<1>();
    return make_vector3(acc(0), acc(1), acc(2));
}

template <typename T>
inline auto make_vector4(const T &x, const T &y, const T &z, const T &w) {
    return TVector4<T>{{x, y, z, w}};
}

inline auto make_matrix2x2(float m00, float m01, float m10, float m11) {
    Matrix2x2f m;
    m.data[0][0] = m00;
    m.data[0][1] = m01;
    m.data[1][0] = m10;
    m.data[1][1] = m11;
    return m;
}

inline auto make_matrix3x3(const Framef &f) {
    Matrix3x3f m;
    m.data[0][0] = f[0][0];
    m.data[0][1] = f[0][1];
    m.data[0][2] = f[0][2];
    m.data[1][0] = f[1][0];
    m.data[1][1] = f[1][1];
    m.data[1][2] = f[1][2];
    m.data[2][0] = f[2][0];
    m.data[2][1] = f[2][1];
    m.data[2][2] = f[2][2];
    return m;
}

inline auto make_matrix3x3(const std::array<float, 9> &v) {
    Matrix3x3f m;
    m.data[0][0] = v[0];
    m.data[0][1] = v[1];
    m.data[0][2] = v[2];
    m.data[1][0] = v[3];
    m.data[1][1] = v[4];
    m.data[1][2] = v[5];
    m.data[2][0] = v[6];
    m.data[2][1] = v[7];
    m.data[2][2] = v[8];
    return m;
}

template <typename T>
inline auto make_frame(const TVector3<T> &u,
                       const TVector3<T> &v,
                       const TVector3<T> &w) {
    return TFrame<T>{{u, v, w}};
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const TVector2<T> &v) {
    return os << "(" << v[0] << ", " << v[1] << ")";
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const TVector3<T> &v) {
    return os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
}

inline std::ostream& operator<<(std::ostream &os, const Matrix2x2f &m) {
    return os << "(" << m(0, 0) << ", " << m(0, 1) << "," << std::endl <<
                        m(1, 0) << ", " << m(1, 1) << ")";
}

inline std::ostream& operator<<(std::ostream &os, const Matrix3x3f &m) {
    return os << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << std::endl <<
                 m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << std::endl <<
                 m(2, 0) << " " << m(2, 1) << " " << m(2, 2);;
}

inline std::ostream& operator<<(std::ostream &os, const Matrix4x4f &m) {
    return os << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << " " << m(0, 3) << std::endl <<
                 m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << " " << m(1, 3) << std::endl <<
                 m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << " " << m(2, 3) << std::endl <<
                 m(3, 0) << " " << m(3, 1) << " " << m(3, 2) << " " << m(3, 3);
}

template <typename T0, typename T1>
inline auto operator+(const TVector2<T0> &v0,
                      const TVector2<T1> &v1) {
    return make_vector2(v0[0] + v1[0], v0[1] + v1[1]);
}

template <typename T0, typename T1>
inline auto& operator+=(TVector2<T0> &v0,
                        const TVector2<T1> &v1) {
    v0[0] += v1[0];
    v0[1] += v1[1];
    return v0;
}

template <typename T0, typename T1>
inline auto operator+(const T0 &v0,
                      const TVector3<T1> &v1) {
    return make_vector3(v0 + v1[0], v0 + v1[1], v0 + v1[2]);
}

template <typename T0, typename T1>
inline auto operator+(const TVector3<T0> &v0,
                      const T1 &v1) {
    return make_vector3(v0[0] + v1, v0[1] + v1, v0[2] + v1);
}

template <typename T0, typename T1>
inline auto operator+(const TVector3<T0> &v0,
                      const TVector3<T1> &v1) {
    return make_vector3(v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2]);
}

template <typename T0, typename T1>
inline auto& operator+=(TVector3<T0> &v0,
                        const TVector3<T1> &v1) {
    v0[0] += v1[0];
    v0[1] += v1[1];
    v0[2] += v1[2];
    return v0;
}

template <typename T>
inline auto operator-(const TVector2<T> &v) {
    return make_vector2(-v[0], -v[1]);
}

template <typename T0, typename T1>
inline auto operator-(const TVector2<T0> &v0,
                      const TVector2<T1> &v1) {
    return make_vector2(v0[0] - v1[0], v0[1] - v1[1]);
}

template <typename T0, typename T1>
inline auto& operator-=(TVector2<T0> &v0,
                        const TVector2<T1> &v1) {
    v0[0] -= v1[0];
    v0[1] -= v1[1];
    return v0;
}

template <typename T0, typename T1>
inline auto operator-(const T0 &v0,
                      const TVector3<T1> &v1) {
    return make_vector3(v0 - v1[0], v0 - v1[1], v0 - v1[2]);
}

template <typename T0, typename T1>
inline auto operator-(const TVector3<T0> &v0,
                      const TVector3<T1> &v1) {
    return make_vector3(v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]);
}

template <typename T>
inline auto operator-(const TVector3<T> &v) {
    return make_vector3(-v[0], -v[1], -v[2]);
}

template <typename T0, typename T1>
inline auto& operator-=(TVector3<T0> &v0,
                        const TVector3<T1> &v1) {
    v0[0] -= v1[0];
    v0[1] -= v1[1];
    v0[2] -= v1[2];
    return v0;
}

template <typename T0, typename T1>
inline auto operator*(const TVector2<T0> &v0,
                      const T1 &s) {
    return make_vector2(v0[0] * s,
                        v0[1] * s);
}

template <typename T0, typename T1>
inline auto operator*(const T0 &s,
                      const TVector2<T1> &v0) {
    return make_vector2(s * v0[0],
                        s * v0[1]);
}

template <typename T0, typename T1>
inline auto operator*(const TVector2<T0> &v0,
                      const TVector2<T1> &v1) {
    return make_vector2(v0[0] * v1[0],
                        v0[1] * v1[1]);
}

template <typename T0, typename T1>
inline auto& operator*=(TVector2<T0> &v0, const T1 &s) {
    v0[0] *= s;
    v0[1] *= s;
    return v0;
}

template <typename T0, typename T1>
inline auto operator*(const TVector3<T0> &v0,
                      const T1 &s) {
    return make_vector3(v0[0] * s,
                        v0[1] * s,
                        v0[2] * s);
}

template <typename T0, typename T1>
inline auto operator*(const T0 &s,
                      const TVector3<T1> &v0) {
    return make_vector3(s * v0[0],
                        s * v0[1],
                        s * v0[2]);
}

template <typename T0, typename T1>
inline auto operator*(const TVector3<T0> &v0,
                      const TVector3<T1> &v1) {
    return make_vector3(v0[0] * v1[0],
                        v0[1] * v1[1],
                        v0[2] * v1[2]);
}

template <typename T0, typename T1>
inline auto& operator*=(TVector3<T0> &v0, const T1 &s) {
    v0[0] *= s;
    v0[1] *= s;
    v0[2] *= s;
    return v0;
}

template <typename T0, typename T1>
inline auto& operator*=(TVector3<T0> &v0,
                       const TVector3<T1> &v1) {
    v0[0] *= v1[0];
    v0[1] *= v1[1];
    v0[2] *= v1[2];
    return v0;
}

template <typename T0, typename T1>
inline auto operator/(const TVector2<T0> &v0,
                      const T1 &s) {
    return make_vector2(v0[0] / s,
                        v0[1] / s);
}

template <typename T0, typename T1>
inline auto& operator/=(TVector2<T0> &v0, const T1 &s) {
    v0[0] /= s;
    v0[1] /= s;
    return v0;
}

template <typename T0, typename T1>
inline auto operator/(const TVector3<T0> &v0,
                      const T1 &s) {
    auto inv_s = 1.f / s;
    return v0 * inv_s;
}

template <typename T0, typename T1>
inline auto& operator/=(TVector3<T0> &v0, const T1 &s) {
    auto inv_s = 1.f / s;
    return v0 *= inv_s;
}

template <typename T0, typename T1>
inline auto dot(const TVector2<T0> &v0,
                const TVector2<T1> &v1) {
    return v0[0] * v1[0] +
           v0[1] * v1[1];
}

template <typename T0, typename T1>
inline auto dot(const TVector3<T0> &v0,
                const TVector3<T1> &v1) {
    return v0[0] * v1[0] +
           v0[1] * v1[1] +
           v0[2] * v1[2];
}

template <typename T0, typename T1>
inline auto cross(const TVector3<T0> &v0,
                  const TVector3<T1> &v1) {
    return make_vector3(
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0]);
}

template <typename T>
inline auto length_squared(const TVector2<T> &v0) {
    return square(v0[0]) + square(v0[1]);
}

template <typename T>
inline auto length_squared(const TVector3<T> &v0) {
    return square(v0[0]) + square(v0[1]) + square(v0[2]);
}

template <typename T>
inline auto length(const TVector2<T> &v0) {
    return sqrt(length_squared(v0));
}

template <typename T>
inline auto length(const TVector3<T> &v0) {
    return sqrt(length_squared(v0));
}

template <typename T0, typename T1>
inline auto distance_squared(const TVector3<T0> &v0,
                     const TVector3<T1> &v1) {
    return length_squared(v1 - v0);
}

template <typename T0, typename T1>
inline auto distance(const TVector3<T0> &v0,
                     const TVector3<T1> &v1) {
    return length(v1 - v0);
}

template <typename T0, typename T1>
inline auto distance(const TVector2<T0> &v0,
                     const TVector2<T1> &v1) {
    return length(v1 - v0);
}

template <typename T>
inline auto normalize(const TVector3<T> &v0) {
    return v0 / length(v0);
}

// backprop the normalize operation
inline auto d_normalize(const Vector3f &v0,
                        const Vector3f &d_out) {
    // return v0 / length(v0);
    auto l = length(v0);
    // inv_l = 1.f / l;
    // out = v0 * inv_l
    auto d_inv_l = dot(d_out, v0);
    auto d_l = -d_inv_l / (l*l);
    // d_length_dx = x / length
    return (d_out + d_l * v0) / l;
}

template <typename T>
inline auto normalize(const TVector2<T> &v0) {
    return v0 / length(v0);
}

template <typename T>
inline auto fabs(const TVector3<T> &v0) {
    return make_vector3(fabs(v0[0]),
                        fabs(v0[1]),
                        fabs(v0[2]));
}

template <typename T>
inline auto luminance(const TVector3<T> &v0) {
    return 0.212671f * v0[0] +
           0.715160f * v0[1] +
           0.072169f * v0[2];
}

template <typename T>
inline auto make_frame(const TVector3<T> &n) {
    if (n[2] < -1.f + 1e-6f) {
        return make_frame(
            make_vector3(T(0), T(-1), T(0)),
            make_vector3(T(-1), T(0), T(0)),
            n
        );
    }
    auto a = 1.f / (1.f + n[2]);
    auto b = -n[0] * n[1] * a;
    return make_frame(
        make_vector3(1.f - square(n[0]) * a, b, -n[0]),
        make_vector3(b, 1.f - square(n[1]) * a, -n[1]),
        n
    );
}

template <typename T0, typename T1>
inline auto to_local(const TFrame<T0> &frame,
                     const TVector3<T1> &v) {
    return make_vector3(dot(v, std::get<0>(frame)),
                        dot(v, std::get<1>(frame)),
                        dot(v, std::get<2>(frame)));
}

template <typename T0, typename T1>
inline auto to_world(const TFrame<T0> &frame,
                     const TVector3<T1> &v) {
    return std::get<0>(frame) * std::get<0>(v) +
           std::get<1>(frame) * std::get<1>(v) +
           std::get<2>(frame) * std::get<2>(v);
}

template <typename T>
inline auto make_normal(const TVector2<T> &v) {
    return make_vector2(std::get<1>(v), -std::get<0>(v));
}

inline auto uniform_hemisphere(const Vector2f &sample) {
    auto z = sample[0];
    auto tmp = sqrt(1.f - z * z);
    auto phi = 2.f * float(M_PI) * sample[1];
    auto sin_phi = sin(phi);
    auto cos_phi = cos(phi);
    return make_vector3(cos_phi * tmp, sin_phi * tmp, z);
}

inline auto cos_hemisphere(const Vector2f &sample) {
    auto phi = 2.f * float(M_PI) * sample[0];
    auto tmp = sqrt(1.f - sample[1]);
    return make_vector3(
        cos(phi) * tmp, sin(phi) * tmp, sqrt(sample[1]));
}

inline auto operator*(const Matrix2x2f &m, const Vector2f &v) {
    return make_vector2(m(0, 0) * std::get<0>(v) + m(0, 1) * std::get<1>(v),
                        m(1, 0) * std::get<0>(v) + m(1, 1) * std::get<1>(v));
}

inline auto invert(const Matrix2x2f &m) {
    auto det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    return make_matrix2x2( m(1, 1) / det, -m(0, 1) / det,
                          -m(1, 0) / det,  m(0, 0) / det);
}

inline auto isnan(const Vector2f &v) {
    return std::isnan(std::get<0>(v)) || std::isnan(std::get<1>(v));
}

inline auto isnan(const Vector3f &v) {
    return std::isnan(std::get<0>(v)) || std::isnan(std::get<1>(v)) || std::isnan(std::get<2>(v));
}

inline auto isfinite(const Vector3a &v) {
    return std::isfinite(std::get<0>(v).val) &&
           std::isfinite(std::get<1>(v).val) &&
           std::isfinite(std::get<2>(v).val);
}

inline auto isfinite(const Vector3f &v) {
    return std::isfinite(std::get<0>(v)) && std::isfinite(std::get<1>(v)) && std::isfinite(std::get<2>(v));
}

template <typename TOutput, typename TInput>
inline auto convert(const TVector2<TInput> &v) {
    return make_vector2(convert<TOutput>(v[0]),
                        convert<TOutput>(v[1]));
}

template <typename TOutput, typename TInput>
inline auto convert(const TVector3<TInput> &v) {
    return make_vector3(convert<TOutput>(v[0]),
                        convert<TOutput>(v[1]),
                        convert<TOutput>(v[2]));
}

template <typename TOutput, typename TInput>
inline auto convert(const TFrame<TInput> &v) {
    return make_frame(convert<TOutput>(v[0]),
                      convert<TOutput>(v[1]),
                      convert<TOutput>(v[2]));
}

inline Vector2f get_adjoint(const Vector2a &v) {
    return make_vector2(get_adjoint(v[0]),
                        get_adjoint(v[1]));
}

inline Vector3f get_adjoint(const Vector3a &v) {
    return make_vector3(get_adjoint(v[0]),
                        get_adjoint(v[1]),
                        get_adjoint(v[2]));
}

template <typename T>
inline auto sum(const TVector3<T> &v) {
    return v[0] + v[1] + v[2];
}

template <typename T>
inline auto max(const TVector3<T> &v) {
    return max(max(v[0], v[1]), v[2]);
}

template <typename T>
inline auto min(const TVector3<T> &v) {
    return min(min(v[0], v[1]), v[2]);
}

inline auto operator*(const Matrix3x3f &m0, const Matrix3x3f &m1) {
    Matrix3x3f ret;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ret(i, j) = 0.f;
            for (int k = 0; k < 3; k++) {
                ret(i, j) += m0(i, k) * m1(k, j);
            }
        }
    }
    return ret;
}

inline auto operator*(const Vector3f &v, const Matrix3x3f &m) {
    Vector3f ret;
    for (int i = 0; i < 3; i++) {
        ret[i] = 0.f;
        for (int j = 0; j < 3; j++) {
            ret[i] += v[j] * m(j, i);
        }
    }
    return ret;
}

inline auto operator*(const Matrix3x3f &m, const Vector3f &v) {
    Vector3f ret;
    for (int i = 0; i < 3; i++) {
        ret[i] = 0.f;
        for (int j = 0; j < 3; j++) {
            ret[i] += m(i, j) * v[j];
        }
    }
    return ret;
}

inline auto determinant(const Matrix3x3f &m) {
    return m(0, 0) * m(1, 1) * m(2, 2) +
           m(1, 0) * m(2, 1) * m(0, 2) +
           m(0, 1) * m(1, 2) * m(2, 0) -
           m(0, 2) * m(1, 1) * m(2, 0) -
           m(0, 1) * m(1, 0) * m(2, 2) -
           m(0, 0) * m(1, 2) * m(2, 1);
}

inline auto inverse(const Matrix3x3f &m) {
    // computes the inverse of a matrix m
    auto det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
               m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
               m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    auto invdet = 1 / det;

    auto m_inv = Matrix3x3f{};
    m_inv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
    m_inv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
    m_inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
    m_inv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
    m_inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
    m_inv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
    m_inv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
    m_inv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
    m_inv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;
    return m_inv;
}

inline auto transpose(const Matrix3x3f &m) {
    Matrix3x3f ret;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ret(i, j) = m(j, i);
        }
    }
    return ret;
}

inline auto intersect_jacobian(const Vector3f &dir,
                               const Vector3f &p,
                               const Vector3f &n) {
    auto d = -dot(p, n);
    auto dir_dot_n = dot(dir, n);
    if (fabs(dir_dot_n) < 1e-10f) {
        Matrix3x3f m;
        m(0, 0) = m(0, 1) = m(0, 2) = 0.f;
        m(1, 0) = m(1, 1) = m(1, 2) = 0.f;
        m(2, 0) = m(2, 1) = m(2, 2) = 0.f;
        return m;
    }
    auto inv_sq_dir_dot_n = 1.f / (dir_dot_n * dir_dot_n);
    auto t = -d / dir_dot_n;
    assert(!isnan(t) && !isnan(inv_sq_dir_dot_n));
    Matrix3x3f m;
    m(0, 0) = t + dir[0] * d * n[0] * inv_sq_dir_dot_n; // d p.x/d dir.x
    m(0, 1) =     dir[0] * d * n[1] * inv_sq_dir_dot_n; // d p.x/d dir.y
    m(0, 2) =     dir[0] * d * n[2] * inv_sq_dir_dot_n; // d p.x/d dir.z
    m(1, 0) =     dir[1] * d * n[0] * inv_sq_dir_dot_n; // d p.y/d dir.x
    m(1, 1) = t + dir[1] * d * n[1] * inv_sq_dir_dot_n; // d p.y/d dir.y
    m(1, 2) =     dir[1] * d * n[2] * inv_sq_dir_dot_n; // d p.y/d dir.z
    m(2, 0) =     dir[2] * d * n[0] * inv_sq_dir_dot_n; // d p.z/d dir.x
    m(2, 1) =     dir[2] * d * n[1] * inv_sq_dir_dot_n; // d p.z/d dir.y
    m(2, 2) = t + dir[2] * d * n[2] * inv_sq_dir_dot_n; // d p.z/d dir.z
    return m;
}

inline auto normalize_jacobian(const Vector3f &x) {
    auto d_sq = length_squared(x);
    auto d = sqrt(d_sq);
    auto inv_d_cu = 1.f / (d * d_sq);
    Matrix3x3f m;
    m(0, 0) = (d_sq - x[0] * x[0]) * inv_d_cu;
    m(0, 1) = (     - x[0] * x[1]) * inv_d_cu;
    m(0, 2) = (     - x[0] * x[2]) * inv_d_cu;
    m(1, 0) = (     - x[1] * x[0]) * inv_d_cu;
    m(1, 1) = (d_sq - x[1] * x[1]) * inv_d_cu;
    m(1, 2) = (     - x[1] * x[2]) * inv_d_cu;
    m(2, 0) = (     - x[2] * x[0]) * inv_d_cu;
    m(2, 1) = (     - x[2] * x[1]) * inv_d_cu;
    m(2, 2) = (d_sq - x[2] * x[2]) * inv_d_cu;
    return m;
}

template <typename T>
inline auto hypot2(float a, const T &b) {
    if (fabs(a) > fabs(b)) {
        auto ratio = b/a;
        return fabs(a) * sqrt(1.f + ratio * ratio);
    } else if (b != 0.f) {
        auto ratio = a/b;
        return fabs(b) * sqrt(1.f + ratio * ratio);
    }
    return T(0.f);
}

inline auto is_zero(const Vector3f& v) {
    return v[0] == 0.f && v[1] == 0.f && v[2] == 0.f;
}

template<typename T>
inline auto clamp(const T &val, const T &lower, const T &upper) {
    if (val <= lower) return lower;
    if (val >= upper) return upper;
    return val;
}

template<typename T>
inline auto operator+=(TMatrix4x4<T> &m0, const TMatrix4x4<T> &m1) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m0(i, j) += m1(i, j);
        }
    }
    return m0;
}

template <typename T>
auto abs_sum(const TVector3<T> &v) {
    return fabs(v[0]) + fabs(v[1]) + fabs(v[2]);
}

inline auto norm(const Matrix4x4f &m) {
    auto sum = 0.f;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            sum += m(i, j) * m(i, j);
        }
    }
    return sum;
}

template <typename T>
auto is_zero(const TVector3<T> &v) {
    return v[0] == 0.f && v[1] == 0.f && v[2] == 0.f;
}