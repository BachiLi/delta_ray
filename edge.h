#pragma once

#include <random>

struct Shape;

struct Edge {
    const Shape *shape = nullptr;
    int v0, v1;
    int f0, f1;

    bool operator==(const Edge &other) const {
    	return shape == other.shape &&
    		v0 == other.v0 && v1 == other.v1 &&
    		f0 == other.f0 && f1 == other.f1;
    }
};

struct EdgeSample {
    float bsdf_component;
    float u;
};

inline auto make_edge_sample(std::mt19937 &rng) {
    std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
    return EdgeSample{uni_dist(rng), uni_dist(rng)};
}

inline auto get_v0(const Edge &edge) {
	return get_vertex(*edge.shape, edge.v0);
}

inline auto get_v1(const Edge &edge) {
	return get_vertex(*edge.shape, edge.v1);
}

inline auto get_non_shared_v0(const Edge &edge) {
    auto ind = get_indices(*edge.shape, edge.f0);
    for (int i = 0; i < 3; i++) {
        if (ind[i] != edge.v0 && ind[i] != edge.v1) {
            return get_vertex(*edge.shape, ind[i]);
        }
    }
    assert(false);
    return make_vector3(0.f, 0.f, 0.f);
}

inline auto get_non_shared_v1(const Edge &edge) {
    auto ind = get_indices(*edge.shape, edge.f1);
    for (int i = 0; i < 3; i++) {
        if (ind[i] != edge.v0 && ind[i] != edge.v1) {
            return get_vertex(*edge.shape, ind[i]);
        }
    }
    assert(false);
    return make_vector3(0.f, 0.f, 0.f);
}

inline bool is_silhouette(const Vector3f &p, const Edge &edge) {
    if (!has_shading_normals(*edge.shape)) {
        // If we are not using Phong normal, every edge is silhouette
        return true;
    }
    if (edge.f0 == -1 || edge.f1 == -1) {
        // Only adjacent to one face
        return true;
    }
    auto v0 = get_v0(edge);
    auto v1 = get_v1(edge);
    auto ns_v0 = get_non_shared_v0(edge);
    auto ns_v1 = get_non_shared_v1(edge);
    auto n0 = normalize(cross(v0 - ns_v0, v1 - ns_v0));
    auto n1 = normalize(cross(v1 - ns_v1, v0 - ns_v1));
    auto frontfacing0 =
        dot(p - v0, n0) > 0.f && dot(p - v1, n0) > 0.f && dot(p - ns_v0, n0) > 0.f;
    auto frontfacing1 =
        dot(p - v0, n1) > 0.f && dot(p - v1, n1) > 0.f && dot(p - ns_v1, n1) > 0.f;
    return (frontfacing0 && !frontfacing1) || (!frontfacing0 && frontfacing1);
}
