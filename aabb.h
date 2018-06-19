#pragma once

#include "vector.h"
#include "intersect.h"

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <array>

/**
 * Axis-aligned bounding box supporting cone intersection
 */
struct AABB {
    AABB();
    AABB(const Vector3f &v0, const Vector3f &v1);
    AABB(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2);

    int maximum_extent() const;
    Vector3f offset(const Vector3f &p) const;
    float surface_area() const;
    bool intersect(const Ray &ray) const;
    bool intersect(const Cone &cone) const;
    inline Vector3f center() const {
        return 0.5f * (p_min + p_max);
    }

    std::pair<Vector3f, Vector3f> get_centered_form() const;
    inline std::pair<Vector3f, float> bounding_sphere() const {
        auto center = (p_min + p_max) / 2.f;
        auto radius = distance(center, p_max);
        return {center, radius};
    }

    inline Vector3f corner(int i) const {
    	Vector3f ret;
    	ret[0] = ((i & 1) == 0) ? p_min[0] : p_max[0];
    	ret[1] = ((i & 2) == 0) ? p_min[1] : p_max[1];
    	ret[2] = ((i & 4) == 0) ? p_min[2] : p_max[2];
    	return ret;
    }
    bool below_plane(const Vector3f &position, const Vector3f &normal) const;

    bool inside(const Vector3f &p) const;

    Vector3f p_min, p_max;
};

AABB merge(const AABB &b, const Vector3f &p);
AABB merge(const AABB &b0, const AABB &b1);

inline std::ostream& operator<<(std::ostream &os, const AABB &b) {
    os << "p_min:" << b.p_min << std::endl;
    os << "p_max:" << b.p_max;
    return os;
}