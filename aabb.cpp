#include "aabb.h"

#include <iostream>

AABB::AABB() {
    p_min = make_vector3(std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::infinity() );
    p_max = make_vector3(-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity() );
}

AABB::AABB(const Vector3f &v0, const Vector3f &v1) {
    for (int i = 0; i < 3; i++) {
        p_min[i] = std::min(v0[i], v1[i]);
        p_max[i] = std::max(v0[i], v1[i]);
    }
}

AABB::AABB(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2) {
    for (int i = 0; i < 3; i++) {
        p_min[i] = std::min(std::min(v0[i], v1[i]), v2[i]);
        p_max[i] = std::max(std::max(v0[i], v1[i]), v2[i]);
    }
}

int AABB::maximum_extent() const {
    Vector3f d = p_max - p_min;
    if (d[0] > d[1] && d[0] > d[2]) {
        return 0;
    } else if (d[1] > d[2]) {
        return 1;
    } else {
        return 2;
    }
}

Vector3f AABB::offset(const Vector3f &p) const {
    Vector3f o = p - p_min;
    for (int i = 0; i < 3; i++) {
        if (p_max[i] > p_min[i]) {
            o[i] /= (p_max[i] - p_min[i]);
        }
    }
    return o;
}

float AABB::surface_area() const {
    Vector3f d = p_max - p_min;
    return 2.f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
}

std::pair<Vector3f, Vector3f> AABB::get_centered_form() const {
    auto center = 0.5f * (p_min + p_max);
    auto extent = 0.5f * (p_max - p_min);
    return {center, extent};
}

bool AABB::below_plane(const Vector3f &position, const Vector3f &normal) const {
    // Loop through 8 vertices, if anyone of them is above return false
    for (int i = 0; i < 8; i++) {
        if (dot(corner(i) - position, normal) > 1e-6f) {
            return false;
        }
    }
    return true;
}

inline auto gamma(int n) {
    auto eps = std::numeric_limits<float>::epsilon() * 0.5;
    return (n * eps) / (1 - n * eps);
}

bool AABB::intersect(const Ray &ray) const {
    auto t0 = 0.f;
    auto t1 = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        auto invRayDir = 1.f / ray.dir[i];
        auto tNear = (p_min[i] - ray.org[i]) * invRayDir;
        auto tFar = (p_max[i] - ray.org[i]) * invRayDir;

        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar) {
            std::swap(tNear, tFar);
        }

        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * gamma(3);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) return false;
    }
    return true;
}

// From GeometricTools (Boost license) 
// https://www.geometrictools.com/GTEngine/Include/Mathematics/GteIntrAlignedBox3Cone3.h
struct ConeIntersector {
    ConeIntersector() {
        // The vector z[] stores the box coordinates of the cone vertex, z[i] =
        // Dot(box.axis[i],cone.vertex-box.center).  Each mPolygon[] has a comment
        // with signs:  s[0] s[1] s[2].  If the sign is '-', z[i] < -e[i].  If the
        // sign is '+', z[i] > e[i].  If the sign is '0', |z[i]| <= e[i].
        polygons[0] = { 6,{ { 1, 5, 4, 6, 2, 3 } } };  // ---
        polygons[1] = { 6,{ { 0, 2, 3, 1, 5, 4 } } };  // 0--
        polygons[2] = { 6,{ { 0, 2, 3, 7, 5, 4 } } };  // +--
        polygons[3] = { 6,{ { 0, 4, 6, 2, 3, 1 } } };  // -0-
        polygons[4] = { 4,{ { 0, 2, 3, 1 } } };        // 00-
        polygons[5] = { 6,{ { 0, 2, 3, 7, 5, 1 } } };  // +0-
        polygons[6] = { 6,{ { 0, 4, 6, 7, 3, 1 } } };  // -+-
        polygons[7] = { 6,{ { 0, 2, 6, 7, 3, 1 } } };  // 0+-
        polygons[8] = { 6,{ { 0, 2, 6, 7, 5, 1 } } };  // ++-
        polygons[9] = { 6,{ { 0, 1, 5, 4, 6, 2 } } };  // --0
        polygons[10] = { 4,{ { 0, 1, 5, 4 } } };        // 0-0
        polygons[11] = { 6,{ { 0, 1, 3, 7, 5, 4 } } };  // +-0
        polygons[12] = { 4,{ { 0, 4, 6, 2 } } };        // -00
        polygons[13] = { 0,{ { 0 } } };                 // 000
        polygons[14] = { 4,{ { 1, 3, 7, 5 } } };        // +00
        polygons[15] = { 6,{ { 0, 4, 6, 7, 3, 2 } } };  // -+0
        polygons[16] = { 4,{ { 2, 6, 7, 3 } } };        // 0+0
        polygons[17] = { 6,{ { 1, 3, 2, 6, 7, 5 } } };  // ++0
        polygons[18] = { 6,{ { 0, 1, 5, 7, 6, 2 } } };  // --+
        polygons[19] = { 6,{ { 0, 1, 5, 7, 6, 4 } } };  // 0-+
        polygons[20] = { 6,{ { 0, 1, 3, 7, 6, 4 } } };  // +-+
        polygons[21] = { 6,{ { 0, 4, 5, 7, 6, 2 } } };  // -0+
        polygons[22] = { 4,{ { 4, 5, 7, 6 } } };        // 00+
        polygons[23] = { 6,{ { 1, 3, 7, 6, 4, 5 } } };  // +0+
        polygons[24] = { 6,{ { 0, 4, 5, 7, 3, 2 } } };  // -++
        polygons[25] = { 6,{ { 2, 6, 4, 5, 7, 3 } } };  // 0++
        polygons[26] = { 6,{ { 1, 3, 2, 6, 4, 5 } } };  // +++
    }

    bool operator()(const AABB &aabb, const Cone &cone) const {
        if (cone.cos_angle <= 0.f) {
            return true;
        }
        auto [center, extent] = aabb.get_centered_form();
        // Quick-rejection test for boxes below the supporting plane of the cone.
        Vector3f CmV = center - cone.ray.org;
        float DdCmV = dot(cone.ray.dir, CmV);  // interval center
        float radius =  // interval half-length
            extent[0] * std::abs(cone.ray.dir[0]) +
            extent[1] * std::abs(cone.ray.dir[1]) +
            extent[2] * std::abs(cone.ray.dir[2]);
        if (DdCmV + radius <= 0) {
            // The box is in the halfspace below the supporting plane of the cone.
            return false;
        }
        // Quick-rejection test for boxes outside the plane determined by the
        // height of the cone.
        // if (DdCmV - radius >= cone.ray.far) {
        //     // The box is outside the plane determined by the height of the
        //     // cone.
        //     return false;
        // }
        // Determine the box faces that are visible to the cone vertex.  The
        // box center has been translated (C-V) so that the cone vertex is at
        // the origin.  Compute the coordinates of the origin relative to the
        // translated box.
        int index[3] = {
            (CmV[0] < -extent[0] ? 2 : (CmV[0] > extent[0] ? 0 : 1)),
            (CmV[1] < -extent[1] ? 2 : (CmV[1] > extent[1] ? 0 : 1)),
            (CmV[2] < -extent[2] ? 2 : (CmV[2] > extent[2] ? 0 : 1))
        };
        int lookup = index[0] + 3 * index[1] + 9 * index[2];
        if (lookup == 13) {
            return true;
        }
        return intersect(extent,
                         cone.cos_angle * cone.cos_angle,
                         cone.ray.dir,
                         CmV,
                         DdCmV,
                         lookup);
    }

    bool intersect(const Vector3f &extent,
                   const float squared_cos_angle,
                   const Vector3f &dir,
                   const Vector3f &CmV,
                   const float DdCmV,
                   const int lookup) const {
        const Polygon &polygon = polygons[lookup];
        // Test polygon points.
        Vector3f X[8], PmV[8];
        float DdPmV[8], sqrDdPmV[8], sqrLenPmV[8], q;
        int iMax = -1, jMax = -1;
        for (int i = 0; i < polygon.numPoints; ++i) {
            int j = polygon.indices[i];
            X[j][0] = (j & 1 ? extent[0] : -extent[0]);
            X[j][1] = (j & 2 ? extent[1] : -extent[1]);
            X[j][2] = (j & 4 ? extent[2] : -extent[2]);
            DdPmV[j] = dot(dir, X[j]) + DdCmV;
            if (DdPmV[j] > 0) {
                PmV[j] = X[j] + CmV;
                sqrDdPmV[j] = DdPmV[j] * DdPmV[j];
                sqrLenPmV[j] = dot(PmV[j], PmV[j]);
                q = sqrDdPmV[j] - squared_cos_angle * sqrLenPmV[j];
                if (q > 0) {
                    return true;
                }
                // Keep track of the maximum in case we must process box edges.
                // This supports the gradient ascent search.
                if (iMax == -1 ||
                    sqrDdPmV[j] * sqrLenPmV[jMax] > sqrDdPmV[jMax] * sqrLenPmV[j]) {
                    iMax = i;
                    jMax = j;
                }
            }
        }
        // Theoretically, this function is called when the box has at least one corner
        // above the supporting plane, in which case DdPmV[j] > 0 for at least one
        // j and consequently iMax should not be -1.  But in case of numerical
        // rounding errors, return a no-intersection result if iMax is -1: the
        // box is below the supporting plane within numerical rounding errors.
        if (iMax == -1) {
            return false;
        }
        // Start the gradient ascent search at index jMax.
        float maxSqrLenPmV = sqrLenPmV[jMax];
        float maxDdPmV = DdPmV[jMax];
        Vector3f &maxX = X[jMax];
        Vector3f &maxPmV = PmV[jMax];
        int k0, k1, k2, jDiff;
        float s, fder, numer, denom, DdMmV, det;
        Vector3f MmV;
        // Search the counterclockwise edge <corner[jMax],corner[jNext]>.
        int iNext = (iMax < polygon.numPoints - 1 ? iMax + 1 : 0);
        int jNext = polygon.indices[iNext];
        jDiff = jNext - jMax;
        s = (jDiff > 0 ? 1 : -1);
        k0 = std::abs(jDiff) >> 1;
        fder = s * (dir[k0] * maxSqrLenPmV - maxDdPmV * maxPmV[k0]);
        if (fder > 0) {
            // The edge has an interior local maximum in F because
            // F(K[j0]) >= F(K[j1]) and the directional derivative of F at K0
            // is positive.  Compute the local maximum point.
            k1 = (k0 + 1) % 3;
            k2 = (k1 + 1) % 3;
            numer = maxPmV[k1] * maxPmV[k1] + maxPmV[k2] * maxPmV[k2];
            denom = dir[k1] * maxPmV[k1] + dir[k2] * maxPmV[k2];
            MmV[k0] = numer * dir[k0];
            MmV[k1] = denom * (maxX[k1] + CmV[k1]);
            MmV[k2] = denom * (maxX[k2] + CmV[k2]);
            // Theoretically, DdMmV > 0, so there is no need to test positivity.
            DdMmV = dot(dir, MmV);
            q = DdMmV * DdMmV - squared_cos_angle * dot(MmV, MmV);
            if (q > 0) {
                return true;
            }
            // Determine on which side of the spherical arc D lives on.  If the
            // polygon side, then the cone ray intersects the polygon and the cone
            // and box intersect.  Otherwise, the D is outside the polygon and the
            // cone and box do not intersect.
            det = s * (dir[k1] * maxPmV[k2] - dir[k2] * maxPmV[k1]);
            return det <= 0;
        }

        // Search the clockwise edge <corner[jMax],corner[jPrev]>.
        int iPrev = (iMax > 0 ? iMax - 1 : polygon.numPoints - 1);
        int jPrev = polygon.indices[iPrev];
        jDiff = jMax - jPrev;
        s = (jDiff > 0 ? 1 : -1);
        k0 = std::abs(jDiff) >> 1;
        fder = -s * (dir[k0] * maxSqrLenPmV - maxDdPmV * maxPmV[k0]);
        if (fder > 0) {
            // The edge has an interior local maximum in F because
            // F(K[j0]) >= F(K[j1]) and the directional derivative of F at K0
            // is positive.  Compute the local maximum point.
            k1 = (k0 + 1) % 3;
            k2 = (k1 + 1) % 3;
            numer = maxPmV[k1] * maxPmV[k1] + maxPmV[k2] * maxPmV[k2];
            denom = dir[k1] * maxPmV[k1] + dir[k2] * maxPmV[k2];
            MmV[k0] = numer * dir[k0];
            MmV[k1] = denom * (maxX[k1] + CmV[k1]);
            MmV[k2] = denom * (maxX[k2] + CmV[k2]);

            // Theoretically, DdMmV > 0, so there is no need to test positivity.
            DdMmV = dot(dir, MmV);
            q = DdMmV * DdMmV - squared_cos_angle * dot(MmV, MmV);
            if (q > 0) {
                return true;
            }

            // Determine on which side of the spherical arc D lives on.  If the
            // polygon side, then the cone ray intersects the polygon and the cone
            // and box intersect.  Otherwise, the D is outside the polygon and the
            // cone and box do not intersect.
            det = s * (dir[k1] * maxPmV[k2] - dir[k2] * maxPmV[k1]);
            return det <= 0;
        }

        return false;
    }
    // The spherical polygons have vertices stored in counterclockwise order
    // when viewed from outside the sphere.  The 'indices' are lookups into
    // {0..26}, where there are 27 possible spherical polygon configurations
    // based on the location of the cone vertex related to the box.
    struct Polygon {
        int numPoints;
        std::array<int, 6> indices;
    };
    std::array<Polygon, 27> polygons;
};

bool AABB::intersect(const Cone &cone) const {
    static ConeIntersector cone_intersector;
    return cone_intersector(*this, cone);
}

bool AABB::inside(const Vector3f &p) const {
    return p[0] >= p_min[0] && p[0] <= p_max[0] &&
           p[1] >= p_min[1] && p[1] <= p_max[1] &&
           p[2] >= p_min[2] && p[2] <= p_max[2];
}

AABB merge(const AABB &b, const Vector3f &p) {
    return AABB(make_vector3(std::min(b.p_min[0], p[0]),
                             std::min(b.p_min[1], p[1]),
                             std::min(b.p_min[2], p[2])),
                make_vector3(std::max(b.p_max[0], p[0]),
                             std::max(b.p_max[1], p[1]),
                             std::max(b.p_max[2], p[2])));
}

AABB merge(const AABB &b0, const AABB &b1) {
    return AABB(make_vector3(std::min(b0.p_min[0], b1.p_min[0]),
                             std::min(b0.p_min[1], b1.p_min[1]),
                             std::min(b0.p_min[2], b1.p_min[2])),
                make_vector3(std::max(b0.p_max[0], b1.p_max[0]),
                             std::max(b0.p_max[1], b1.p_max[1]),
                             std::max(b0.p_max[2], b1.p_max[2])));
}
