#pragma once

#include "aabb.h"
#include "edge.h"

#include <random>

struct Scene;
struct Camera;
struct Distribution1D;

struct LinearBVHNode {
    AABB spatial_bounds;
    Vector3f normal_cone_axis;
    float normal_cone_angle;
    union {
        int primitives_offset;    // leaf
        int second_child_offset;  // interior
    };
    int num_primitives;      // 0 -> interior node
    float total_length;
};

struct EdgeTree {
    EdgeTree() {}
    EdgeTree(const std::vector<LinearBVHNode> &nodes,
             const std::vector<Edge> &edges)
        : nodes(nodes), edges(edges) {}

    std::vector<LinearBVHNode> nodes;
    std::vector<Edge> edges;
};

struct EdgeAndWeight {
    Edge edge;
    float weight;
};

struct DoubleEdgeTree {
    std::vector<Edge> primary_edges;
    std::shared_ptr<Distribution1D> primary_edge_samplers;
    std::shared_ptr<EdgeTree> silhouette_tree;
    std::shared_ptr<EdgeTree> normal_tree;
};

DoubleEdgeTree build_double_edge_tree(const Camera &camera,
                                      const std::vector<Edge> &edges);
EdgeAndWeight sample_primary_edge(const DoubleEdgeTree &tree, float u);
void sample_shading_edge(const DoubleEdgeTree &tree,
                         const Vector3f &wi,
                         const Intersection &shading_isect,
                         const SurfacePoint &shading_point,
                         const Cone &cone,
                         std::vector<EdgeAndWeight> &edges,
                         std::mt19937 &rng);
