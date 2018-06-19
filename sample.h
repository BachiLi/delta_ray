#pragma once

#include "vector.h"
#include "intersect.h"

template <typename T0, typename T1, typename T2>
inline auto sample_triangle(const TVector3<T0> &v0,
                            const TVector3<T1> &v1,
                            const TVector3<T2> &v2,
                            const Vector2f &tri_uv) {
    auto a = sqrt(tri_uv[0]);
    auto b1 = 1.f - a;
    auto b2 = a * tri_uv[1];
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto n = cross(e1, e2);
    // pdf = 1.f / (0.5f * length(convert<float>(n)));
    auto normalized_n = normalize(n);
    return make_surface_point(
        v0 + e1 * b1 + e2 * b2,
        normalized_n,
        make_frame(normalized_n), // TODO: give true light source normal
        convert<T0>(tri_uv)); // TODO: give true light source uv
}
