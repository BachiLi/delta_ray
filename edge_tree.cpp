#undef NDEBUG 

#include "edge_tree.h"
#include "scene.h"
#include "distribution.h"
#include "camera.h"
#include "material.h"
#include "line_clip.h"

struct BVHPrimitiveInfo {
    BVHPrimitiveInfo() {}
    BVHPrimitiveInfo(int primitive_number,
                     const AABB &spatial_bounds)
        : primitive_number(primitive_number),
          spatial_bounds(spatial_bounds),
          spatial_centroid(.5f * spatial_bounds.p_min + .5f * spatial_bounds.p_max) {}
    int primitive_number;
    Vector3f v0, v1;
    Vector3f n0, n1;
    AABB spatial_bounds;
    AABB scaled_directional_bounds;
    Vector3f spatial_centroid;
    Vector3f scaled_directional_centroid;
};

struct BVHBuildNode {
    void init_leaf(int first, int n) {
        first_prim_offset = first;
        num_primitives = n;
        children[0] = children[1] = nullptr;
    }
    void init_interior(int axis,
                       std::shared_ptr<BVHBuildNode> c0,
                       std::shared_ptr<BVHBuildNode> c1) {
        children[0] = c0;
        children[1] = c1;
        split_axis = axis;
        num_primitives = 0;
    }

    AABB spatial_bounds;
    Vector3f normal_cone_axis;
    float normal_cone_angle;
    AABB scaled_directional_bounds;
    std::shared_ptr<BVHBuildNode> children[2];
    int split_axis, first_prim_offset, num_primitives;
};

template <bool use_directional_bounds>
std::shared_ptr<BVHBuildNode> build_edge_tree(
        const std::vector<Edge> &edges,
        std::vector<BVHPrimitiveInfo> &primitive_info,
        int start, int end, int *total_nodes,
        std::vector<Edge> &ordered_prims) {
    std::shared_ptr<BVHBuildNode> node = std::make_shared<BVHBuildNode>();
    (*total_nodes)++;
    // Compute bounds of all primitives in BVH node
    AABB spatial_bounds;
    AABB scaled_directional_bounds;
    for (int i = start; i < end; ++i) {
        spatial_bounds = merge(spatial_bounds, primitive_info[i].spatial_bounds);
        // spatial_bounds = merge(spatial_bounds, primitive_info[i].v0);
        // spatial_bounds = merge(spatial_bounds, primitive_info[i].v1);
        if (use_directional_bounds) {
            scaled_directional_bounds =
                merge(scaled_directional_bounds,
                      primitive_info[i].scaled_directional_bounds);
        }
    }
    Vector3f normal_sum{0.f, 0.f, 0.f};
    for (int i = start; i < end; ++i) {
        normal_sum += primitive_info[i].n0;
        normal_sum += primitive_info[i].n1;
    }
    Vector3f cone_axis = normalize(normal_sum);
    float cone_cos_angle = 1.f;
    for (int i = start; i < end; ++i) {
        cone_cos_angle =
            std::min(cone_cos_angle, dot(primitive_info[i].n0, cone_axis));
        cone_cos_angle =
            std::min(cone_cos_angle, dot(primitive_info[i].n1, cone_axis));
    }
    node->spatial_bounds = spatial_bounds;
    node->normal_cone_axis = cone_axis;
    node->normal_cone_angle = acos(cone_cos_angle);
    node->scaled_directional_bounds = scaled_directional_bounds;
    int num_primitives = end - start;
    if (num_primitives == 1) {
        // Create leaf _BVHBuildNode_
        int first_prim_offset = ordered_prims.size();
        for (int i = start; i < end; ++i) {
            int prim_num = primitive_info[i].primitive_number;
            ordered_prims.push_back(edges[prim_num]);
        }
        node->init_leaf(first_prim_offset,
                        num_primitives);
    } else {
        // Compute bound of primitive centroids, choose split dimension _dim_
        AABB spatial_centroid_bounds;
        AABB directional_centroid_bounds;
        for (int i = start; i < end; ++i) {
            spatial_centroid_bounds =
                merge(spatial_centroid_bounds, primitive_info[i].spatial_centroid);
            if (use_directional_bounds) {
                directional_centroid_bounds =
                    merge(directional_centroid_bounds,
                          primitive_info[i].scaled_directional_centroid);
            }
        }
        int dim = -1;
        float max_extent = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < 3; i++) {
            float extent = spatial_centroid_bounds.p_max[i] -
                           spatial_centroid_bounds.p_min[i];
            if (extent > max_extent) {
                dim = i;
                max_extent = extent;
            }
        }
        if (use_directional_bounds) {
            for (int i = 0; i < 3; i++) {
                float extent = directional_centroid_bounds.p_max[i] -
                               directional_centroid_bounds.p_min[i];
                if (extent > max_extent) {
                    dim = i + 3;
                    max_extent = extent;
                }
            }
        }
        assert(dim >= 0);
        const AABB &used_centroid_bounds =
            dim < 3 ? spatial_centroid_bounds : directional_centroid_bounds;
        bool max_is_spatial = dim < 3;
        if (dim >= 3) {
            dim -= 3;
        }

        // Partition primitives into two sets and build children
        int mid = (start + end) / 2;
        if (used_centroid_bounds.p_max[dim] == used_centroid_bounds.p_min[dim]) {
            // Create leaf _BVHBuildNode_
            int first_prim_offset = ordered_prims.size();
            for (int i = start; i < end; ++i) {
                int prim_num = primitive_info[i].primitive_number;
                ordered_prims.push_back(edges[prim_num]);
            }
            node->init_leaf(first_prim_offset, num_primitives);
        } else {
            // Partition primitives into equally-sized subsets
            mid = (start + end) / 2;
            std::nth_element(&primitive_info[start], &primitive_info[mid],
                             &primitive_info[end - 1] + 1,
                             [dim, max_is_spatial](const BVHPrimitiveInfo &a,
                                   const BVHPrimitiveInfo &b) {
                                if (max_is_spatial) {
                                    return a.spatial_centroid[dim] <
                                           b.spatial_centroid[dim];
                                } else {
                                    return a.scaled_directional_centroid[dim] <
                                           b.scaled_directional_centroid[dim];
                                }
                             });
            node->init_interior(dim,
                build_edge_tree<use_directional_bounds>(
                    edges, primitive_info, start, mid,
                    total_nodes, ordered_prims),
                build_edge_tree<use_directional_bounds>(
                    edges, primitive_info, mid, end,
                    total_nodes, ordered_prims));
        }
    }
    return node;
}

int flatten_bvh(std::vector<LinearBVHNode> &nodes,
                const std::vector<Edge> &ordered_prims,
                const BVHBuildNode *build_node,
                int *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->spatial_bounds = build_node->spatial_bounds;
    linearNode->normal_cone_axis = build_node->normal_cone_axis;
    linearNode->normal_cone_angle = build_node->normal_cone_angle;
    int my_offset = (*offset)++;
    if (build_node->num_primitives > 0) {
        linearNode->primitives_offset = build_node->first_prim_offset;
        linearNode->num_primitives = build_node->num_primitives;
        linearNode->total_length = 0.f;
        for (int i = 0; i < linearNode->num_primitives; ++i) {
            const Edge &edge = ordered_prims[i + linearNode->primitives_offset];
            auto v0 = get_v0(edge);
            auto v1 = get_v1(edge);
            linearNode->total_length += distance(v0, v1);
        }
    } else {
        // Create interior flattened BVH node
        linearNode->num_primitives = 0;
        flatten_bvh(nodes, ordered_prims, build_node->children[0].get(), offset);
        linearNode->second_child_offset =
            flatten_bvh(nodes, ordered_prims, build_node->children[1].get(), offset);
        linearNode->total_length =
            nodes[my_offset + 1].total_length +
            nodes[linearNode->second_child_offset].total_length;
    }
    return my_offset;
}

template <bool use_directional_bounds>
std::shared_ptr<EdgeTree> build_edge_tree(const std::vector<Edge> &edges) {
    if (edges.size() == 0) {
        return std::make_shared<EdgeTree>();
    }
    std::vector<BVHPrimitiveInfo> primitive_info(edges.size());
    AABB world_bounds;
    for (int i = 0; i < (int)edges.size(); i++) {
        AABB spatial_bounds(get_v0(edges[i]), get_v1(edges[i]));
        primitive_info[i] = BVHPrimitiveInfo(i, spatial_bounds);
        world_bounds = merge(world_bounds, spatial_bounds);
    }
    if (use_directional_bounds) {
        // divide by 8.f comes from the multi-dimensional lightcuts paper
        float directional_factor =
            length(world_bounds.p_max - world_bounds.p_min) / 8.f;
        for (int i = 0; i < (int)edges.size(); i++) {
            assert(has_shading_normals(*edges[i].shape) &&
                   edges[i].f0 != -1 && edges[i].f1 != -1);
            // Sample shading normal at the center of the edge
            auto sn0 = get_shading_normal(*edges[i].shape, edges[i].v0);
            auto sn1 = get_shading_normal(*edges[i].shape, edges[i].v1);
            auto sn = normalize(sn0 + sn1);
            // Get geometric normals and flip them to the same direction of 
            // shading normal
            auto n0 = get_normal(*edges[i].shape, edges[i].f0);
            auto n1 = get_normal(*edges[i].shape, edges[i].f1);
            if (dot(n0, sn) < 0.f) {
                n0 = -n0;
            }
            if (dot(n1, sn) < 0.f) {
                n1 = -n1;
            }
            primitive_info[i].v0 = get_v0(edges[i]);
            primitive_info[i].v1 = get_v1(edges[i]);
            primitive_info[i].n0 = n0;
            primitive_info[i].n1 = n1;
            primitive_info[i].scaled_directional_bounds =
                AABB(n0 * directional_factor, n1 * directional_factor);
            primitive_info[i].scaled_directional_centroid =
                0.5f * (primitive_info[i].scaled_directional_bounds.p_min +
                        primitive_info[i].scaled_directional_bounds.p_max);
        }
    }
    std::vector<Edge> ordered_prims;
    auto total_nodes = 0;
    auto root = build_edge_tree<use_directional_bounds>(
        edges, primitive_info, 0, edges.size(), &total_nodes,
        ordered_prims);
    std::vector<LinearBVHNode> nodes(total_nodes);
    auto offset = 0;
    flatten_bvh(nodes, ordered_prims, root.get(), &offset);
    assert(ordered_prims.size() == edges.size());
    return std::make_shared<EdgeTree>(nodes, ordered_prims);
}

DoubleEdgeTree build_double_edge_tree(const Camera &camera,
                                      const std::vector<Edge> &edges) {
    // First gather all silhouette primary edges that are within the screen
    std::vector<Edge> primary_edges;
    std::vector<float> edges_length;
    for (const auto &edge : edges) {
        auto v0 = get_v0(edge);
        auto v1 = get_v1(edge);
        auto proj = project(camera, v0, v1);
        auto dist = 0.f;
        if (proj) {
            auto v0_ss = std::get<0>(*proj);
            auto v1_ss = std::get<1>(*proj);
            auto clipped = clip_line(v0_ss, v1_ss);
            if (clipped) {
                v0_ss = std::get<0>(*clipped);
                v1_ss = std::get<1>(*clipped);
                dist = distance(v0_ss, v1_ss);
            }
        }
        auto org = xfm_point(camera.cam_to_world, make_vector3(0.f, 0.f, 0.f));
        if (!is_silhouette(org, edge)) {
            // Reject non-silhouette edges
            dist = 0.f;
        }
        if (dist > 0.f) {
            primary_edges.push_back(edge);
            edges_length.push_back(dist);
        }
    }    
    auto primary_edge_sampler = std::make_shared<Distribution1D>(
        &edges_length[0], edges_length.size());

    std::vector<Edge> silhouette_edges;
    std::vector<Edge> normal_edges;
    for (const auto &edge : edges) {
        // If the edge only connects to one face or
        // the mesh doesn't have shading normals,
        // it's a "silhouette edge".
        if (edge.f0 == -1 || edge.f1 == -1 ||
                !has_shading_normals(*edge.shape)) {
            silhouette_edges.push_back(edge);
        } else {
            normal_edges.push_back(edge);
        }
    }
    // Don't use directional bounds for silhouette edges
    std::shared_ptr<EdgeTree> silhouette_tree =
        build_edge_tree<false>(silhouette_edges);
    std::shared_ptr<EdgeTree> normal_tree =
        build_edge_tree<true>(normal_edges);
    return DoubleEdgeTree{
        primary_edges, primary_edge_sampler, silhouette_tree, normal_tree};
}

EdgeAndWeight sample_primary_edge(const DoubleEdgeTree &tree,
                                  float u) {
    if (tree.primary_edges.size() == 0) {
        return EdgeAndWeight{Edge{}, 0.f};
    }
    float pmf = 0.f;
    auto id = tree.primary_edge_samplers->sample_discrete(u, &pmf);
    // assert(pmf > 0.f);
    float weight = pmf > 0.f ? 1.f / pmf : 0.f;
    return EdgeAndWeight{tree.primary_edges[id], weight};
}

auto min_square_bound(float min, float max) {
    if (min <= 0.f && max >= 0.f) {
        return 0.f;
    }
    if (min < 0.f && max < 0.f) {
        return max * max;
    }
    assert(min > 0.f && max > 0.f);
    return min * min;
}

auto max_square_bound(float min, float max) {
    return std::max(min * min, max * max);
}

float bsdf_bound(const LinearBVHNode &node,
                 const Vector3f &wi,
                 const Intersection &shading_isect,
                 const SurfacePoint &shading_point,
                 float diffuse_refl_lum,
                 float specular_refl_lum) {
    const auto &bounds = node.spatial_bounds;
    const auto &material = *shading_isect.shape->material;
    // Compute BSDF bound
    // We use a very conservative cosine bound since its faster
    auto cos_max = 0.f;
    for (int i = 0; i < 8; i++) {
        auto corner = bounds.corner(i);
        if (dot(shading_point.shading_frame[2],
                corner - shading_point.position) > 0.f) {
            cos_max = 1.f;
            break;
        }
    }
    // return cos_max;
    if (cos_max <= 0.f) {
        // Not true for refractive materials
        return 0.f;
    }
    // https://pdfs.semanticscholar.org/330e/59117d7da6c794750730a15f9a178391b9fe.pdf
    // // Project 8 corners of the original bounds into
    // // local space where shading normal = (0, 0, 1)
    // auto projected_bound = AABB();
    // for (int i = 0; i < 8; i++) {
    //     auto corner = bounds.corner(i);
    //     auto local_corner =
    //         to_local(shading_point.shading_frame, corner - shading_point.position);
    //     projected_bound = merge(projected_bound, local_corner);
    // }
    // // cosine bound:
    // auto max_z = projected_bound.p_max[2];
    // if (max_z <= 0.f) {
    //     // TODO: not true for refractive materials
    //     return 0.f;
    // }
    // auto min_x_sq =
    //     min_square_bound(projected_bound.p_min[0], projected_bound.p_max[0]);
    // auto min_y_sq =
    //     min_square_bound(projected_bound.p_min[1], projected_bound.p_max[1]);
    // auto cos_max = max_z /
    //     sqrt(min_x_sq + min_y_sq + max_z * max_z);
    auto max_phong_response = 0.f;
    // auto diffuse_refl = get_diffuse_reflectance(material, shading_point.uv);
    // auto specular_refl = get_specular_reflectance(material, shading_point.uv);
    if (specular_refl_lum > 0.f) {
        // Half vector cosine bound
        // Normal needs to be orthogonal to y
        auto hv_frame_y = normalize(cross(shading_point.shading_frame[2], wi));
        auto hv_frame = make_frame(
            normalize(cross(wi, hv_frame_y)),
            hv_frame_y,
            wi);
        // Again, projecct 8 corners to hv_frame
        auto hv_bound = AABB();
        for (int i = 0; i < 8; i++) {
            auto corner = bounds.corner(i);
            auto local_corner =
                to_local(hv_frame, corner - shading_point.position);
            hv_bound = merge(hv_bound, local_corner);
        }
        auto cos_phi_h_max = 0.f;
        auto hv_max_x = hv_bound.p_max[0];
        if (hv_max_x > 0.f) {
            auto hv_min_y_sq =
                min_square_bound(hv_bound.p_min[1], hv_bound.p_max[1]);
            cos_phi_h_max = hv_max_x /
                sqrt(hv_min_y_sq + hv_max_x * hv_max_x);
        } else if (hv_max_x == 0.f) {
            cos_phi_h_max = max(cos_phi_h_max, 0.f);
        } else {
            auto hv_max_y_sq =
                max_square_bound(hv_bound.p_min[1], hv_bound.p_max[1]);
            cos_phi_h_max = hv_max_x /
                sqrt(hv_max_y_sq + hv_max_x * hv_max_x);
        }
        auto cos_theta_o_min = 0.f;
        if (hv_bound.p_min[2] > 0.f) {
            auto max_x_sq = max_square_bound(hv_bound.p_min[0], hv_bound.p_max[0]);
            auto max_y_sq = max_square_bound(hv_bound.p_min[1], hv_bound.p_max[1]);
            cos_theta_o_min = hv_bound.p_min[2] /
                (max_x_sq + max_y_sq + hv_bound.p_min[2] * hv_bound.p_min[2]);
        } else {
            auto min_x_sq = min_square_bound(hv_bound.p_min[0], hv_bound.p_max[0]);
            auto min_y_sq = min_square_bound(hv_bound.p_min[1], hv_bound.p_max[1]);
            cos_theta_o_min = hv_bound.p_min[2] /
                (min_x_sq + min_y_sq + hv_bound.p_min[2] * hv_bound.p_min[2]);
        }
        auto cos_theta_o_max = 0.f;
        if (hv_bound.p_max[2] > 0.f) {
            auto min_x_sq = min_square_bound(hv_bound.p_min[0], hv_bound.p_max[0]);
            auto min_y_sq = min_square_bound(hv_bound.p_min[1], hv_bound.p_max[1]);
            cos_theta_o_max = hv_bound.p_max[2] /
                (min_x_sq + min_y_sq + hv_bound.p_max[2] * hv_bound.p_max[2]);
        } else {
            auto max_x_sq = max_square_bound(hv_bound.p_min[0], hv_bound.p_max[0]);
            auto max_y_sq = max_square_bound(hv_bound.p_min[1], hv_bound.p_max[1]);
            cos_theta_o_max = hv_bound.p_max[2] /
                (max_x_sq + max_y_sq + hv_bound.p_max[2] * hv_bound.p_max[2]);
        }
        auto cos_theta_h_min = sqrt((cos_theta_o_min + 1.f) / 2.f);
        auto cos_theta_h_max = sqrt((cos_theta_o_max + 1.f) / 2.f);
        auto cos_h_max = 0.f;
        auto local_n = to_local(hv_frame, shading_point.shading_frame[2]);
        assert(local_n[0] >= 0.f && local_n[2] >= 0.f);
        if (cos_phi_h_max >= 0.f) {
            cos_h_max = sqrt(local_n[2] * local_n[2] /
                (local_n[2] * local_n[2] + local_n[0] * local_n[0] *
                    cos_phi_h_max * cos_phi_h_max));
        } else {
            cos_h_max = 1.f;
        }
        if (cos_h_max > cos_theta_h_max) {
            cos_h_max = cos_theta_h_max;
        } else if (cos_h_max < cos_theta_h_min) {
            cos_h_max = cos_theta_h_min;
        }
        auto sin_h_max = sqrt(max(1.f - cos_h_max * cos_h_max, 0.f));
        auto h_dot_n_max = min(fabs(local_n[0] * sin_h_max * cos_phi_h_max) +
            local_n[2] * cos_h_max, 1.f);
        // Compute blinn-phong response based on cos_h_max
        auto phong_exponent =
            roughness_to_phong(get_roughness(material, shading_point.uv));
        max_phong_response = pow(h_dot_n_max, phong_exponent) *
            (phong_exponent + 2.f) / float(2 * M_PI);
        assert(isfinite(max_phong_response));
    }
    auto ret = cos_max * (diffuse_refl_lum / float(M_PI) +
        specular_refl_lum * max_phong_response);
    assert(isfinite(ret));
    return ret;
}

bool contains_silhouette(const LinearBVHNode &node,
                         const Vector3f &p) {
    auto [spatial_center, spatial_radius] = node.spatial_bounds.bounding_sphere();
    auto dir_to_node = spatial_center - p;
    auto len_dir_to_node = length(dir_to_node);
    auto view_tan_angle = spatial_radius / len_dir_to_node;
    if (view_tan_angle >= 1.f) {
        return true;
    }
    auto view_angle = atan(view_tan_angle);
    dir_to_node /= len_dir_to_node;
    auto angle = acos(dot(-dir_to_node, node.normal_cone_axis));
    auto angle_range = node.normal_cone_angle + view_angle;
    return angle - angle_range <= float(M_PI/2.f) &&
           angle + angle_range >= float(M_PI/2.f);
}

float node_importance(const LinearBVHNode &node,
                      float bsdf_bound,
                      const Vector3f &wi,
                      const Intersection &shading_isect,
                      const SurfacePoint &shading_point) {
    const auto &bounds = node.spatial_bounds;
    auto ret = bsdf_bound * node.total_length /
        distance_squared(bounds.center(), shading_point.position);
    assert(isfinite(ret));
    return ret;
}

template <bool use_directional_bounds, bool cone_mode>
void sample_edge(int depth,
                 int node_id,
                 float node_bsdf_bound,
                 const EdgeTree &tree,
                 const Vector3f &wi,
                 const Intersection &shading_isect,
                 const SurfacePoint &shading_point,
                 float diffuse_refl_lum,
                 float specular_refl_lum,
                 const Cone &cone,
                 std::vector<EdgeAndWeight> &edges,
                 float cone_pmf,
                 float all_pmf,
                 std::mt19937 &rng) {
    if (node_bsdf_bound <= 0.f) {
        return;
    }
    assert(node_id < (int)tree.nodes.size());
    const LinearBVHNode &node = tree.nodes[node_id];
    if (node.num_primitives > 0) {
        // Leaf node, uniformly sample the primitives
        auto weight = cone_mode ?
            cone_pmf / (cone_pmf * cone_pmf + all_pmf * all_pmf) :
            all_pmf / (cone_pmf * cone_pmf + all_pmf * all_pmf);
        for (int i = 0; i < node.num_primitives; i++) {
            auto index = node.primitives_offset + i;
            const Edge &edge = tree.edges[index];
            if (is_silhouette(shading_point.position, edge)) {
                edges.push_back(EdgeAndWeight{edge, weight});
            }
        }
        return;
    }

    // Decide whether or not to split the node
    // For now we just test whether the cone hits the bounds
    const LinearBVHNode &left_node = tree.nodes[node_id + 1];
    const LinearBVHNode &right_node = tree.nodes[node.second_child_offset];
    // split if the point is inside bound or if the bsdf bound is high
    bool all_split = node_bsdf_bound > 1.f || 
        node.spatial_bounds.inside(shading_point.position);
    bool left_contains_silhouette = true, right_contains_silhouette = true;
    if (use_directional_bounds) {
        left_contains_silhouette =
            contains_silhouette(left_node, shading_point.position);
        right_contains_silhouette =
            contains_silhouette(right_node, shading_point.position);
    }
    auto left_bsdf_bound = left_contains_silhouette ?
        bsdf_bound(left_node, wi, shading_isect, shading_point,
                   diffuse_refl_lum, specular_refl_lum) : 0.f;
    auto right_bsdf_bound = right_contains_silhouette ?
        bsdf_bound(right_node, wi, shading_isect, shading_point,
                   diffuse_refl_lum, specular_refl_lum) : 0.f;
    auto importance_left = left_contains_silhouette ?
        node_importance(left_node, left_bsdf_bound,
            wi, shading_isect, shading_point) : 0.f;
    auto importance_right = right_contains_silhouette ?
        node_importance(right_node, right_bsdf_bound,
            wi, shading_isect, shading_point) : 0.f;
    if (importance_left + importance_right <= 1e-10f) {
        return;
    }
    auto left_hit_cone = left_node.spatial_bounds.intersect(cone);
    auto right_hit_cone = right_node.spatial_bounds.intersect(cone);
    auto cone_split = all_split || (left_hit_cone && right_hit_cone &&
        left_contains_silhouette && right_contains_silhouette &&
        cone.cos_angle > 0.95f);
    auto importance_left_cone = importance_left;
    auto importance_right_cone = importance_right;
    if (!left_hit_cone) {
        importance_left_cone = 0.f;
    }
    if (!right_hit_cone) {
        importance_right_cone = 0.f;
    }
    if (cone_mode && importance_left_cone + importance_right_cone <= 1e-10f) {
        return;
    }
    auto pmf_left_cone = 0.f;
    auto pmf_right_cone = 0.f;
    auto pmf_left_all = 0.f;
    auto pmf_right_all = 0.f;
    if (cone_split) {
        pmf_left_cone = pmf_right_cone = 1.f;
    } else if (importance_left_cone + importance_right_cone > 1e-10f) {
        pmf_left_cone = importance_left_cone / (importance_left_cone + importance_right_cone);
        pmf_right_cone = 1.f - pmf_left_cone;        
    }
    if (all_split) {
        pmf_left_all = pmf_right_all = 1.f;
    } else {
        assert(importance_left + importance_right > 1e-10f);
        pmf_left_all = importance_left / (importance_left + importance_right);
        pmf_right_all = 1.f - pmf_left_all;        
    }

    auto split = cone_mode ? cone_split : all_split;
    if (split) {
        if (left_contains_silhouette) {
            sample_edge<use_directional_bounds, cone_mode>(
                depth + 1, node_id + 1, left_bsdf_bound, tree, wi,
                shading_isect, shading_point,
                diffuse_refl_lum, specular_refl_lum, 
                cone, edges,
                pmf_left_cone * cone_pmf,
                pmf_left_all * all_pmf, rng);
        }
        if (right_contains_silhouette) {
            sample_edge<use_directional_bounds, cone_mode>(
                depth + 1, node.second_child_offset, right_bsdf_bound,
                tree, wi, shading_isect, shading_point,
                diffuse_refl_lum, specular_refl_lum,
                cone, edges,
                pmf_right_cone * cone_pmf,
                pmf_right_all * all_pmf, rng);
        }
    } else {
        // Choose one node based on importance
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        float u = uni_dist(rng);
        bool sample_left = cone_mode ? u < pmf_left_cone : u < pmf_left_all;
        if (sample_left) {
            sample_edge<use_directional_bounds, cone_mode>(
                depth + 1, node_id + 1, left_bsdf_bound,
                tree, wi, shading_isect, shading_point,
                diffuse_refl_lum, specular_refl_lum, 
                cone, edges,
                pmf_left_cone * cone_pmf,
                pmf_left_all * all_pmf, rng);
        } else { // u >= pmf_left
            sample_edge<use_directional_bounds, cone_mode>(
                depth + 1, node.second_child_offset, right_bsdf_bound,
                tree, wi, shading_isect, shading_point,
                diffuse_refl_lum, specular_refl_lum, 
                cone, edges,
                pmf_right_cone * cone_pmf,
                pmf_right_all * all_pmf, rng);
        }
    }
}

template <bool use_directional_bounds>
void sample_shading_edge(const EdgeTree &tree,
                         const Vector3f &wi,
                         const Intersection &shading_isect,
                         const SurfacePoint &shading_point,
                         const Cone &cone,
                         float pmf,
                         std::vector<EdgeAndWeight> &edges,
                         std::mt19937 &rng) {
    if (tree.nodes.size() == 0) {
        return;
    }
    const LinearBVHNode &node = tree.nodes[0];
    const auto &material = *shading_isect.shape->material;
    auto diffuse_refl = get_diffuse_reflectance(material, shading_point.uv);
    auto specular_refl = get_specular_reflectance(material, shading_point.uv);
    auto diffuse_refl_lum = luminance(diffuse_refl);
    auto specular_refl_lum = luminance(specular_refl);
    auto b = bsdf_bound(node, wi, shading_isect, shading_point,
                        diffuse_refl_lum, specular_refl_lum);
    // Sample cone
    sample_edge<use_directional_bounds, true>(
        0, 0, b, tree, wi, shading_isect, shading_point,
        diffuse_refl_lum, specular_refl_lum, cone, edges,
        pmf, pmf, rng);
    // Sample all
    sample_edge<use_directional_bounds, false>(
       0, 0, b, tree, wi, shading_isect, shading_point,
       diffuse_refl_lum, specular_refl_lum, cone, edges, pmf, pmf, rng);
}

void sample_shading_edge(const DoubleEdgeTree &tree,
                         const Vector3f &wi,
                         const Intersection &shading_isect,
                         const SurfacePoint &shading_point,
                         const Cone &cone,
                         std::vector<EdgeAndWeight> &edges,
                         std::mt19937 &rng) {
    if (dot(wi, shading_point.shading_frame[2]) < 0.f) {
        // TODO: refraction
        return;
    }
    if (tree.normal_tree->nodes.size() == 0) {
        sample_shading_edge<false>(
            *tree.silhouette_tree, wi,
            shading_isect, shading_point, cone, 1.f, edges, rng);
        return;
    }
    if (tree.silhouette_tree->nodes.size() == 0) {
        sample_shading_edge<true>(
            *tree.normal_tree, wi,
            shading_isect, shading_point, cone, 1.f, edges, rng);
        return;
    }
    float nimp = tree.normal_tree->nodes[0].total_length;
    float simp = tree.silhouette_tree->nodes[0].total_length;
    float npmf = nimp / (nimp + simp);
    float spmf = simp / (nimp + simp);
    std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
    float u = uni_dist(rng);
    if (u < npmf) {
        // Sample normal edges
        sample_shading_edge<true>(
            *tree.normal_tree, wi,
            shading_isect, shading_point, cone, npmf, edges, rng);
    } else {
        // Sample silhouette edges
        sample_shading_edge<false>(
            *tree.silhouette_tree, wi,
            shading_isect, shading_point, cone, spmf, edges, rng);
    }
}
