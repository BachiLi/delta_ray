#undef NDEBUG

#include "vector.h"
#include "camera.h"
#include "intersect.h"
#include "distribution.h"
#include "shape.h"
#include "material.h"
#include "light.h"
#include "edge.h"
#include "sample.h"
#include "parallel.h"
#include "scene.h"
#include "progress_reporter.h"
#include "edge_tree.h"
#include "pathtrace.h"

#include <embree3/rtcore.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include <random>
#include <chrono>
#include <mutex>
#include <signal.h>

DECLARE_ADGRAPH();

namespace py = pybind11;

void kb_interrupt_handler(int sig) {
    // py::gil_scoped_acquire acquire;
    std::cerr << "\nInterrupted by Ctrl+C" << std::endl;
    Py_Exit(1);
}

struct ShadingEdgeSampleResult {
    Vector3f dp = make_vector3(0.f, 0.f, 0.f);
    Vector3f dv0 = make_vector3(0.f, 0.f, 0.f);
    Vector3f dv1 = make_vector3(0.f, 0.f, 0.f);
};

auto sample_shading_edge(const Scene &scene,
                         const EdgeSample &edge_sample,
                         const Edge &edge,
                         const LightSample &light_sample,
                         const std::vector<SecondarySample> &secondary_samples,
                         const Vector3f &wi,
                         const Intersection &shading_isect,
                         const SurfacePoint &shading_point,
                         const Vector3f &weight,
                         const Vector3f &d_color,
                         bool direct_only) {
    if (!shading_isect.valid()) {
        return ShadingEdgeSampleResult{};
    }
    if (!is_silhouette(shading_point.position, edge)) {
        return ShadingEdgeSampleResult{};
    }

    // Get the two vertices of the edge
    auto v0 = get_vertex(*edge.shape, edge.v0);
    auto v1 = get_vertex(*edge.shape, edge.v1);
    if (abs_sum(v1 - v0) == 1e-10f) {
        return ShadingEdgeSampleResult{};
    }

    // Importance sample the edge using linearly transformed cosine
    // First decide which component of BRDF to sample
    const Material &material = *shading_isect.shape->material;
    auto diffuse_reflectance =
        get_diffuse_reflectance(material, shading_point.uv);
    auto specular_reflectance =
        get_specular_reflectance(material, shading_point.uv);
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        // black BSDF?
        return ShadingEdgeSampleResult{};
    }
    auto diffuse_pmf = diffuse_weight / weight_sum;
    auto specular_pmf = specular_weight / weight_sum;
    auto m_pmf = 0.f;
    auto n = shading_point.shading_frame[2];
    auto frame_x = normalize(wi - n * dot(wi, n));
    auto frame_y = cross(n, frame_x);
    auto isotropic_frame = make_frame(frame_x, frame_y, n);
    auto m = Matrix3x3f{};
    auto m_inv = Matrix3x3f{};
    if (edge_sample.bsdf_component <= diffuse_pmf) {
        // M is shading frame * identity
        m_inv = make_matrix3x3(isotropic_frame);
        m = inverse(m_inv);
        m_pmf = diffuse_pmf;
    } else {
        m_inv = inverse(get_ltc_matrix(material, shading_point, wi)) *
                make_matrix3x3(isotropic_frame);
        m = inverse(m_inv);
        // m = inverse(make_matrix3x3(shading_point.shading_frame)) *
        //     get_ltc_matrix(material, shading_point, wi);
        // m_inv = inverse(m);
        m_pmf = specular_pmf;
    }
    auto v0o = m_inv * (v0 - shading_point.position);
    auto v1o = m_inv * (v1 - shading_point.position);
    if (v0o[2] <= 0.f && v1o[2] <= 0.f) {
        // The integral is 0 (edge below horizon)
        // The derivatives are 0.
        return ShadingEdgeSampleResult{};
    }
    // Clip to the horizon
    if (v0o[2] < 0.f) {
        v0o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
    }
    if (v1o[2] < 0.f) {
        v1o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
    }
    assert(v0o[2] >= 0.f && v1o[2] >= 0.f);
    auto u = edge_sample.u;
    auto vodir = v1o - v0o;
    if (abs_sum(vodir) < 1e-10f) {
        return ShadingEdgeSampleResult{};
    }
    auto wt = normalize(vodir);
    auto l0 = dot(v0o, wt);
    auto l1 = dot(v1o, wt);
    assert(!isnan(v0o));
    assert(!isnan(v1o));
    assert(!isnan(wt));
    assert(!isnan(l0));
    assert(!isnan(l1));
    auto vo = v0o - l0 * wt;
    auto d = length(vo);
    auto I = [&](float l) {
        return (l/(d*(d*d+l*l))+atan(l/d)/(d*d))*vo[2] +
               (l*l/(d*(d*d+l*l)))*wt[2];
    };
    auto Il0 = I(l0);
    auto Il1 = I(l1);
    auto normalization = Il1 - Il0;
    if (normalization <= 1e-10f) {
        return ShadingEdgeSampleResult{};
    }
    auto cdf = [&](float l) {
        return (I(l)-Il0)/normalization;
    };
    auto pdf = [&](float l) {
        auto dist_sq=d*d+l*l;
        return 2.f*d*(vo+l*wt)[2]/(normalization*dist_sq*dist_sq);
    };
    // Hybrid bisection & Newton iteration
    auto lb = l0;
    auto ub = l1;
    if (lb > ub) {
        std::swap(lb, ub);
    }
    auto l = 0.5f * (lb + ub);
    for (int it = 0; it < 20; it++) {
        if (!(l >= lb && l <= ub)) {
            l = 0.5f * (lb + ub);
        }
        auto value = cdf(l) - u;
        if (fabs(value) < 1e-5f || it == 19) {
            break;
        }
        // The derivative may not be entirely accurate,
        // but the bisection is going to handle this
        if (value > 0.f) {
            ub = l;
        } else {
            lb = l;
        }
        auto derivative = pdf(l);
        l -= value / derivative;
    }
    if (pdf(l) <= 0.f) {
        // Numerical issue
        return ShadingEdgeSampleResult{};
    }
    assert(!isnan(l));
    // Convert from l to position
    auto sample_p = m * (vo + l * wt);
    // shading_point.position, v0 and v1 forms a plane that split the space
    // into two parts
    auto split_plane_normal =
        normalize(cross(v0 - shading_point.position, v1 - shading_point.position));
    // Generate sample directions
    auto delta = 1e-3f * length(sample_p);
    auto sample_dir = normalize(sample_p);

    // Sample two rays on the two sides of the edge
    auto v_right_dir = normalize(sample_dir + delta * split_plane_normal);
    auto v_left_dir  = normalize(sample_dir - delta * split_plane_normal);
    // Intersect the world
    auto isect_right_result = intersect(scene,
        make_ray(shading_point.position, v_right_dir));
    auto isect_right = std::get<0>(isect_right_result);
    auto surface_point_right = std::get<1>(isect_right_result);
    auto isect_left_result  = intersect(scene,
        make_ray(shading_point.position, v_left_dir));
    auto isect_left = std::get<0>(isect_left_result);
    auto surface_point_left = std::get<1>(isect_left_result);
    // At least one of the intersections should be connected to the edge
    bool right_connected =
        isect_right.shape == edge.shape &&
        (isect_right.tri_id == edge.f0 || isect_right.tri_id == edge.f1);
    bool left_connected =
        isect_left.shape == edge.shape &&
        (isect_left.tri_id == edge.f0 || isect_left.tri_id == edge.f1);
    if ((!right_connected && !left_connected) ||
            (right_connected && left_connected) ||
            (isect_right.shape == shading_isect.shape &&
                isect_right.tri_id == shading_isect.tri_id) || // self intersection
            (isect_left.shape == shading_isect.shape &&
                isect_left.tri_id == shading_isect.tri_id)) {
        return ShadingEdgeSampleResult();
    }
    // Reject samples where the unconnected face is closer than the connected ones
    // The connected face is supposed to "block" the unconnected one
    // (probably some numerical error)
    // if (right_connected && isect_left.valid()) {
    //    auto left_dir = surface_point_left.position - shading_point.position;
    //    auto left_distance_squared = length_squared(left_dir);
    //    auto right_dir = surface_point_right.position - shading_point.position;
    //    auto right_distance_squared = length_squared(right_dir);
    //    if (left_distance_squared < right_distance_squared) {
    //        return ShadingEdgeSampleResult();
    //    }
    // }
    // if (left_connected && isect_right.valid()) {
    //    auto left_dir = surface_point_left.position - shading_point.position;
    //    auto left_distance_squared = length_squared(left_dir);
    //    auto right_dir = surface_point_right.position - shading_point.position;
    //    auto right_distance_squared = length_squared(right_dir);
    //    if (right_distance_squared < left_distance_squared) {
    //        return ShadingEdgeSampleResult();
    //    }
    // }

    auto eval_bsdf = bsdf(material, shading_point, wi, sample_dir);
    if (sum(eval_bsdf) < 1e-6f) {
        return ShadingEdgeSampleResult();
    }
    auto shade_right = make_vector3(0.f, 0.f, 0.f);
    auto shade_left = make_vector3(0.f, 0.f, 0.f);
    if (!direct_only) {
        shade_right = shade(scene,
                            -sample_dir,
                            isect_right,
                            surface_point_right,
                            light_sample,
                            secondary_samples,
                            false);
        shade_left = shade(scene,
                           -sample_dir,
                           isect_left,
                           surface_point_left,
                           light_sample,
                           secondary_samples,
                           false);
        if (isect_right.valid()) {
            auto dir = surface_point_right.position - shading_point.position;
            auto distance_squared = length_squared(dir);
            if (distance_squared > 0.f) {
                // XXX HACK: clamp the maximum contribution of global illumination
                shade_right /= std::max(distance_squared, 1e-2f);
                shade_right *= fabs(dot(normalize(dir), surface_point_right.geom_normal));
            } else {
                shade_right = make_vector3(0.f, 0.f, 0.f);
            }
        }
        if (isect_left.valid()) {
            auto dir = surface_point_left.position - shading_point.position;
            auto distance_squared = length_squared(dir);
            if (distance_squared > 0.f) {
                // XXX HACK: clamp the maximum contribution of global illumination
                shade_left /= std::max(distance_squared, 1e-2f);
                shade_left *= fabs(dot(normalize(dir), surface_point_left.geom_normal));
            } else {
                shade_left = make_vector3(0.f, 0.f, 0.f);
            }
        }
    }
    // Evaluate the contribution of the two rays
    auto eval_right = 
        eval(scene, wi, shading_isect, shading_point,
             isect_right, surface_point_right,
             false) + eval_bsdf * shade_right;
    auto eval_left =
        eval(scene, wi, shading_isect, shading_point,
             isect_left, surface_point_left,
             false) + eval_bsdf * shade_left;
    // XXX ad-hoc clamping XXX
    // Fix this at some point
    if (isect_right.valid()) {
        auto dir = surface_point_right.position - shading_point.position;
        auto distance = length(dir);
        auto cos_term = fabs(dot(normalize(dir), surface_point_right.geom_normal));
        if (distance * cos_term < 1e-3f) {
            return ShadingEdgeSampleResult();
        }
    }
    if (isect_left.valid()) {
        auto dir = surface_point_left.position - shading_point.position;
        auto distance = length(dir);
        auto cos_term = fabs(dot(normalize(dir), surface_point_left.geom_normal));
        if (distance * cos_term < 1e-3f) {
            return ShadingEdgeSampleResult();
        }
    }

    assert(!isnan(eval_right) && !isnan(eval_left));
    auto dcolor_dp = make_vector3(0.f, 0.f, 0.f);
    auto dcolor_dv0 = make_vector3(0.f, 0.f, 0.f);
    auto dcolor_dv1 = make_vector3(0.f, 0.f, 0.f);
    // Accumulate derivatives from two sides of the edge
    auto eval_edge = [&](const Intersection &light_surface_isect,
                         const SurfacePoint &light_surface_point,
                         const Vector3f &eval) {
        // Jacobian from l to p = wt
        // Jacobian from p to Mp = M
        // Jacobian from Mp to dir
        auto norm_jacobian = normalize_jacobian(sample_p);
        // Jacobian from dir to hit_pos
        auto isect_jacobian =
            intersect_jacobian(sample_dir,
                               light_surface_point.position,
                               light_surface_point.geom_normal);
        auto line_jacobian =
            length(isect_jacobian * norm_jacobian * m * wt) / (m_pmf * pdf(l));
        assert(!isnan(line_jacobian));

        auto p = shading_point.position;
        auto x = light_surface_point.position;
        auto d0 = v0 - p;
        auto d1 = v1 - p;
        auto dirac_jacobian = length(cross(d0, d1));
        assert(!isnan(dirac_jacobian));
        auto w = line_jacobian / dirac_jacobian;
        assert(!isnan(w));

        auto dp = w * (cross(d1, d0) +
                       cross(x - p, d1) +
                       cross(d0, x - p));
        auto dv0 = w * cross(d1, x - p);
        auto dv1 = w * cross(x - p, d0);
        assert(!isnan(dp) && !isnan(dv0) && !isnan(dv1));
        auto weighted_eval = dot(eval * weight, d_color);
        dcolor_dp += dp * weighted_eval;
        dcolor_dv0 += dv0 * weighted_eval;
        dcolor_dv1 += dv1 * weighted_eval;
    };
    if (isect_right.valid() && abs_sum(eval_right) > 0.f) {
        eval_edge(isect_right, surface_point_right, eval_right);
    }
    if (isect_left.valid() && abs_sum(eval_left) > 0.f) {
        eval_edge(isect_left, surface_point_left, -eval_left);
    }
    assert(!isnan(dcolor_dp) && !isnan(dcolor_dv0) && !isnan(dcolor_dv1));

    return ShadingEdgeSampleResult{dcolor_dp, dcolor_dv0, dcolor_dv1};
}

struct PrimaryEdgeSampleResult {
    float screen_x = -1.f, screen_y = -1.f;
    Edge edge = Edge{};
    Vector3f dv0 = make_vector3(0.f, 0.f, 0.f);
    Vector3f dv1 = make_vector3(0.f, 0.f, 0.f);
    Vector2f dv0_ss = make_vector2(0.f, 0.f);
    Vector2f dv1_ss = make_vector2(0.f, 0.f);
    Matrix4x4f d_cam_to_sample, d_world_to_cam;
};

auto sample_primary_edge(const Scene &scene,
                         const EdgeSample &edge_sample,
                         const Edge &edge,
                         const LightSample &light_sample,
                         const std::vector<SecondarySample> &secondary_samples,
                         const py::array_t<float> &d_image,
                         float weight) {
    // Project the edge onto screen space
    auto v0 = get_v0(edge);
    auto v1 = get_v1(edge);
    auto proj = project(scene.camera, v0, v1);
    if (!proj) {
        // Both points are behind the camera
        return PrimaryEdgeSampleResult{};
    }
    auto v0_ss = std::get<0>(*proj);
    auto v1_ss = std::get<1>(*proj);
    auto v0_dir = unproject(scene.camera, v0_ss);
    auto v1_dir = unproject(scene.camera, v1_ss);
    assert(!isnan(v0_dir) && !isnan(v1_dir));
    // Pick a point on the edge
    auto v_dir3 = v1_dir - v0_dir;
    auto t = edge_sample.u;
    auto v0_local = xfm_point(scene.camera.world_to_cam, v0);
    auto v1_local = xfm_point(scene.camera.world_to_cam, v1);
    auto edge_pt3 = v0_dir + t * v_dir3;
    auto edge_local = v0_local + t * v1_local;
    // Reject samples outside of image plane
    // TODO: a smarter sampling strategy is to not sample these in the first place
    auto edge_pt = project_local(scene.camera, edge_pt3);
    assert(!isnan(edge_pt));
    if (!in_screen(scene.camera, edge_pt)) {
        return PrimaryEdgeSampleResult{};
    }
    auto xi = int(edge_pt[0] * d_image.shape()[1]);
    auto yi = int(edge_pt[1] * d_image.shape()[0]);
    auto d_img_accessor = d_image.unchecked<3>();
    auto d_color = make_vector3(
        d_img_accessor(yi, xi, 0),
        d_img_accessor(yi, xi, 1),
        d_img_accessor(yi, xi, 2));

    // Compute boundary difference by shading twice
    auto delta = 1e-4f * length(edge_local);
    auto split_plane_normal = normalize(cross(v0_dir, v1_dir));
    // Sample two rays on the two sides of the edge
    auto right_dir = normalize(edge_pt3 + delta * split_plane_normal);
    auto right_pt = project_local(scene.camera, right_dir);
    auto right_ray = sample_primary(scene.camera, right_pt);
    auto right_isect_result = intersect(scene, right_ray);
    auto isect_right = std::get<0>(right_isect_result);
    auto left_dir = normalize(edge_pt3 - delta * split_plane_normal);
    auto left_pt = project_local(scene.camera, left_dir);
    auto left_ray = sample_primary(scene.camera, left_pt);
    auto left_isect_result = intersect(scene, left_ray);
    auto isect_left = std::get<0>(left_isect_result);
    // At least one of the intersections should be connected to the edge
    bool right_connected = isect_right.shape == edge.shape &&
        (isect_right.tri_id == edge.f0 || isect_right.tri_id == edge.f1);
    bool left_connected = isect_left.shape == edge.shape &&
        (isect_left.tri_id == edge.f0 || isect_left.tri_id == edge.f1);
    if (!right_connected && !left_connected) {
       return PrimaryEdgeSampleResult{};
    }
    auto right_color = shade(scene,
                             -right_ray.dir,
                             std::get<0>(right_isect_result),
                             std::get<1>(right_isect_result),
                             light_sample,
                             secondary_samples,
                             true);
    auto left_color = shade(scene,
                            -left_ray.dir,
                            std::get<0>(left_isect_result),
                            std::get<1>(left_isect_result),
                            light_sample,
                            secondary_samples,
                            true);
    auto diff = (right_color - left_color) * weight;
    auto xdiff = diff;
    if (left_pt[0] > right_pt[0]) {
        xdiff = -xdiff;
    }
    auto ydiff = diff;
    if (left_pt[1] > right_pt[1]) {
        ydiff = -ydiff;
    }
    auto edge_dir = unproject(scene.camera, edge_pt);
    auto d_v0_dir = d_unproject(scene.camera, v0_ss);
    auto d_v1_dir = d_unproject(scene.camera, v1_ss);
    auto d_edge_dir = d_unproject(scene.camera, edge_pt);
    // alpha(x, y) = dot(edge_dir(x, y), cross(v0_dir, v1_dir))
    // d alpha(x, y)/dx = dot(d/dx edge_dir(x, y),  cross(v0_dir, v1_dir))
    // d alpha(x, y)/d v0_ss_x = dot(cross(v1_dir, edge_dir), d_unproject(v0_ss).x)
    auto d_alpha_dx = dot(std::get<0>(d_edge_dir), cross(v0_dir, v1_dir));
    auto d_alpha_dy = dot(std::get<1>(d_edge_dir), cross(v0_dir, v1_dir));
    auto d_alpha_d_v0_ss_x = fabs(dot(cross(v1_dir, edge_dir), std::get<0>(d_v0_dir)));
    auto d_alpha_d_v0_ss_y = fabs(dot(cross(v1_dir, edge_dir), std::get<1>(d_v0_dir)));
    auto d_alpha_d_v1_ss_x = fabs(dot(cross(edge_dir, v0_dir), std::get<0>(d_v1_dir)));
    auto d_alpha_d_v1_ss_y = fabs(dot(cross(edge_dir, v0_dir), std::get<1>(d_v1_dir)));
    auto dirac_jacobian = 1.f / sqrt(d_alpha_dx * d_alpha_dx + d_alpha_dy * d_alpha_dy);
    auto edge_pt3_delta = v0_dir + (t + delta) * v_dir3;
    auto edge_pt_delta = project_local(scene.camera, edge_pt3_delta);
    auto line_jacobian = length((edge_pt_delta - edge_pt) / delta);
    auto jacobian = line_jacobian * dirac_jacobian;

    auto dv0_ss_x = dot(-xdiff * d_alpha_d_v0_ss_x * jacobian, d_color);
    auto dv0_ss_y = dot(-ydiff * d_alpha_d_v0_ss_y * jacobian, d_color);
    auto dv1_ss_x = dot(-xdiff * d_alpha_d_v1_ss_x * jacobian, d_color);
    auto dv1_ss_y = dot(-ydiff * d_alpha_d_v1_ss_y * jacobian, d_color);
    assert(!isnan(xdiff) && !isnan(ydiff));
    assert(!isnan(dv0_ss_x) && !isnan(dv0_ss_y) && !isnan(dv1_ss_x) && !isnan(dv1_ss_y));

    // Need to propagate from screen space to world space
    // Use finite difference to obtain the Jacobian
    // TODO: replace this with closed-form later
    auto fd_delta = 1e-3f;
    auto proj_v0x = project(scene.camera, v0 + make_vector3(fd_delta, 0.f, 0.f), v1);
    auto proj_v0y = project(scene.camera, v0 + make_vector3(0.f, fd_delta, 0.f), v1);
    auto proj_v0z = project(scene.camera, v0 + make_vector3(0.f, 0.f, fd_delta), v1);
    auto proj_v1x = project(scene.camera, v0, v1 + make_vector3(fd_delta, 0.f, 0.f));
    auto proj_v1y = project(scene.camera, v0, v1 + make_vector3(0.f, fd_delta, 0.f));
    auto proj_v1z = project(scene.camera, v0, v1 + make_vector3(0.f, 0.f, fd_delta));
    if (!proj_v0x || !proj_v0y || !proj_v0z || !proj_v1x || !proj_v1y || !proj_v1z) {
        // Numerical issue of finite difference, just abort operation.
        // This should be rare
        return PrimaryEdgeSampleResult{};
    }
    auto v0ss_delta_v0x = std::get<0>(*proj_v0x);
    auto v1ss_delta_v0x = std::get<1>(*proj_v0x);
    auto v0ss_delta_v0y = std::get<0>(*proj_v0y);
    auto v1ss_delta_v0y = std::get<1>(*proj_v0y);
    auto v0ss_delta_v0z = std::get<0>(*proj_v0z);
    auto v1ss_delta_v0z = std::get<1>(*proj_v0z);
    auto v0ss_delta_v1x = std::get<0>(*proj_v1x);
    auto v1ss_delta_v1x = std::get<1>(*proj_v1x);
    auto v0ss_delta_v1y = std::get<0>(*proj_v1y);
    auto v1ss_delta_v1y = std::get<1>(*proj_v1y);
    auto v0ss_delta_v1z = std::get<0>(*proj_v1z);
    auto v1ss_delta_v1z = std::get<1>(*proj_v1z);
    auto dv0ss_dv0x = (v0ss_delta_v0x - v0_ss) / fd_delta;
    auto dv0ss_dv0y = (v0ss_delta_v0y - v0_ss) / fd_delta;
    auto dv0ss_dv0z = (v0ss_delta_v0z - v0_ss) / fd_delta;
    auto dv0ss_dv1x = (v0ss_delta_v1x - v0_ss) / fd_delta;
    auto dv0ss_dv1y = (v0ss_delta_v1y - v0_ss) / fd_delta;
    auto dv0ss_dv1z = (v0ss_delta_v1z - v0_ss) / fd_delta;
    auto dv1ss_dv0x = (v1ss_delta_v0x - v1_ss) / fd_delta;
    auto dv1ss_dv0y = (v1ss_delta_v0y - v1_ss) / fd_delta;
    auto dv1ss_dv0z = (v1ss_delta_v0z - v1_ss) / fd_delta;
    auto dv1ss_dv1x = (v1ss_delta_v1x - v1_ss) / fd_delta;
    auto dv1ss_dv1y = (v1ss_delta_v1y - v1_ss) / fd_delta;
    auto dv1ss_dv1z = (v1ss_delta_v1z - v1_ss) / fd_delta;
    auto dv0 = make_vector3(
        dv0_ss_x * dv0ss_dv0x[0] + dv0_ss_y * dv0ss_dv0x[1] +
        dv1_ss_x * dv1ss_dv0x[0] + dv1_ss_y * dv1ss_dv0x[1],
        dv0_ss_x * dv0ss_dv0y[0] + dv0_ss_y * dv0ss_dv0y[1] +
        dv1_ss_x * dv1ss_dv0y[0] + dv1_ss_y * dv1ss_dv0y[1],
        dv0_ss_x * dv0ss_dv0z[0] + dv0_ss_y * dv0ss_dv0z[1] +
        dv1_ss_x * dv1ss_dv0z[0] + dv1_ss_y * dv1ss_dv0z[1]);
    auto dv1 = make_vector3(
        dv0_ss_x * dv0ss_dv1x[0] + dv0_ss_y * dv0ss_dv1x[1] +
        dv1_ss_x * dv1ss_dv1x[0] + dv1_ss_y * dv1ss_dv1x[1],
        dv0_ss_x * dv0ss_dv1y[0] + dv0_ss_y * dv0ss_dv1y[1] +
        dv1_ss_x * dv1ss_dv1y[0] + dv1_ss_y * dv1ss_dv1y[1],
        dv0_ss_x * dv0ss_dv1z[0] + dv0_ss_y * dv0ss_dv1z[1] +
        dv1_ss_x * dv1ss_dv1z[0] + dv1_ss_y * dv1ss_dv1z[1]);
    assert(!isnan(dv0) && !isnan(dv1));

    Matrix4x4f d_cam_to_sample, d_world_to_cam;
    d_project(scene.camera, v0, v1,
        dv0_ss_x, dv0_ss_y, dv1_ss_x, dv1_ss_y,
        d_cam_to_sample, d_world_to_cam);

    return PrimaryEdgeSampleResult{
        edge_pt[0], edge_pt[1],
        edge,
        dv0, dv1,
        make_vector2(dv0_ss_x, dv0_ss_y),
        make_vector2(dv1_ss_x, dv1_ss_y),
        d_cam_to_sample, d_world_to_cam};
}

struct CameraDerivative {
    Matrix4x4f d_cam_to_world, d_world_to_cam;
    Matrix4x4f d_sample_to_cam, d_cam_to_sample;
};

void accumulate(CameraDerivative &target, const CameraDerivative &source) {
    target.d_cam_to_world += source.d_cam_to_world;
    target.d_world_to_cam += source.d_world_to_cam;
    target.d_sample_to_cam += source.d_sample_to_cam;
    target.d_cam_to_sample += source.d_cam_to_sample;
}

struct ShapeDerivative {
    const Shape *shape;
    int tri_id;
    Vector3f dv0, dv1, dv2;
    Vector3f dn0, dn1, dn2;
};

template <typename T>
struct TextureDerivative {
    int xi, yi;
    T t00, t01, t10, t11;
};

struct MaterialDerivative {
    const Material *material;
    TextureDerivative<Vector3f> diffuse_reflectance;
    TextureDerivative<Vector3f> specular_reflectance;
    TextureDerivative<float> roughness;
    Vector2f diffuse_uv_scale;
    Vector2f specular_uv_scale;
    Vector2f roughness_uv_scale;
};

struct LightDerivative {
    const Light *light;
    Vector3f d_intensity;
};

struct EdgeDerivative {
    Edge edge;
    Vector3f dv0, dv1;
};

struct PositionDerivative {
    Vector3a p;
    Vector3f dp;
};

struct Sample {
    Vector3f color;
    Vector3f color_dx, color_dy;
    CameraDerivative camera_derivatives;
    std::vector<ShapeDerivative> shape_derivatives;
    std::vector<MaterialDerivative> material_derivatives;
    std::vector<LightDerivative> light_derivatives;
    std::vector<EdgeDerivative> edge_derivatives;
};

// Dynamic buffers used for rendering a sample
struct SampleBuffer {
    std::vector<IntersectParameter<AReal>> isect_params;
    std::vector<SecondarySample> secondary_samples;
    std::vector<SecondaryPoint<AReal>> secondary_points;
    std::vector<MaterialParameter<AReal>> material_params;
    std::vector<LightParameter<AReal>> light_params;
    std::vector<PositionDerivative> dps;
    std::vector<EdgeAndWeight> edges;

    void clear() {
        isect_params.clear();
        secondary_samples.clear();
        secondary_points.clear();
        material_params.clear();
        light_params.clear();
        dps.clear();
        edges.clear();
    }
};

auto render_sample_fast(const Scene &scene,
                        int img_width,
                        int img_height,
                        int xi,
                        int yi,
                        float sub_x,
                        float sub_y,
                        int num_samples,
                        int max_bounces,
                        std::mt19937 &rng,
                        SampleBuffer &sample_buffer) {
    auto sample_weight = 1.f / num_samples;
    Sample result;
    result.color = make_vector3(0.f, 0.f, 0.f);
    result.color_dx = make_vector3(0.f, 0.f, 0.f);
    result.color_dy = make_vector3(0.f, 0.f, 0.f);
    sample_buffer.clear();
    auto x = float((xi + sub_x) / img_width);
    auto y = float((yi + sub_y) / img_height);
    auto camera = TCamera<float>(scene.camera);
    auto screen_pos = make_vector2(x, y);
    auto primary_ray = sample_primary(camera, screen_pos);
    if (abs_sum(primary_ray.dir) == 0.f) {
        return result;
    }
    //auto& isect_params = sample_buffer.isect_params;
    auto isect_result = intersect(scene, primary_ray);
    auto primary_isect = std::get<0>(isect_result);
    if (!primary_isect.valid()) {
        return result;
    }
    auto shading_point = std::get<1>(isect_result);
    auto light_sample = make_light_sample(rng);
    auto& secondary_samples = sample_buffer.secondary_samples;
    for (int i = 0; i < max_bounces; i++) {
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        secondary_samples.push_back(
            SecondarySample{make_light_sample(rng),
                            make_bsdf_sample(rng),
                            uni_dist(rng)});
    }
    Vector3f color;
    color = shade(scene,
                  -primary_ray.dir,
                  primary_isect,
                  shading_point,
                  light_sample,
                  secondary_samples,
                  true) * sample_weight;
    result.color = convert<float>(color);
    assert(!isnan(result.color));
    return result;
}

auto render_sample(const Scene &scene,
                   int img_width,
                   int img_height,
                   int xi,
                   int yi,
                   float sub_x,
                   float sub_y,
                   int num_samples,
                   int max_bounces,
                   const Vector3f &d_color,
                   ADGraph &ad_graph,
                   std::mt19937 &rng,
                   SampleBuffer &sample_buffer,
                   bool compute_derivative) {
    if (!compute_derivative || is_zero(d_color)) {
        return render_sample_fast(scene,
                img_width, img_height, xi, yi,
                sub_x, sub_y, num_samples, max_bounces,
                rng, sample_buffer);
    }
    auto sample_weight = 1.f / num_samples;
    Sample result;
    result.color = make_vector3(0.f, 0.f, 0.f);
    result.color_dx = make_vector3(0.f, 0.f, 0.f);
    result.color_dy = make_vector3(0.f, 0.f, 0.f);
    ad_graph.clear();
    sample_buffer.clear();
    auto x = AReal((xi + sub_x) / img_width);
    auto y = AReal((yi + sub_y) / img_height);
    auto camera = TCamera<AReal>(scene.camera);
    auto screen_pos = make_vector2(x, y);
    auto primary_ray = sample_primary(camera, screen_pos);
    if (abs_sum(primary_ray.dir) == 0.f) {
        return result;
    }
    auto& isect_params = sample_buffer.isect_params;
    auto isect_result = intersect(scene, primary_ray, &isect_params);
    auto primary_isect = std::get<0>(isect_result);
    if (!primary_isect.valid()) {
        return result;
    }
    auto shading_point = std::get<1>(isect_result);
    auto light_sample = make_light_sample(rng);
    auto& secondary_samples = sample_buffer.secondary_samples;
    for (int i = 0; i < max_bounces; i++) {
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        secondary_samples.push_back(
            SecondarySample{make_light_sample(rng),
                            make_bsdf_sample(rng),
                            uni_dist(rng)});
    }
    auto& secondary_points = sample_buffer.secondary_points;
    auto& material_params = sample_buffer.material_params;
    auto& light_params = sample_buffer.light_params;
    Vector3a color;
    color = shade(scene,
                  -primary_ray.dir,
                  primary_isect,
                  shading_point,
                  light_sample,
                  secondary_samples,
                  true,
                  compute_derivative ? &secondary_points : nullptr,
                  compute_derivative ? &isect_params : nullptr,
                  compute_derivative ? &material_params : nullptr,
                  compute_derivative ? &light_params : nullptr) * sample_weight;
    // XXX Hack: don't perform autodiff if color is really really bright
    // usually this is due to bad importance sampling and can ruin the derivatives
    //if (luminance(color) > 5.f) {
        //color = convert<AReal>(convert<float>(color));
    //}
    result.color = convert<float>(color);
    assert(!isnan(result.color));
    if (!compute_derivative) {
        return result;
    }

    ad_graph.zero_adjoints();
    set_adjoint(color[0], d_color[0]);
    set_adjoint(color[1], d_color[1]);
    set_adjoint(color[2], d_color[2]);
    propagate_adjoint();
    // Camera derivatives
    auto set_matrix_adjoint = [&] (const TMatrix4x4<AReal> &mat, Matrix4x4f &buffer) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                buffer(i, j) += get_adjoint(mat(i, j));
            }
        }
    };
    set_matrix_adjoint(camera.cam_to_world, result.camera_derivatives.d_cam_to_world);
    set_matrix_adjoint(camera.world_to_cam, result.camera_derivatives.d_world_to_cam);
    set_matrix_adjoint(camera.sample_to_cam, result.camera_derivatives.d_sample_to_cam);
    set_matrix_adjoint(camera.cam_to_sample, result.camera_derivatives.d_cam_to_sample);
    // Shape derivatives
    for (int i = 0; i < (int)isect_params.size(); i++) {
        auto dv0 = get_adjoint(isect_params[i].v0);
        auto dv1 = get_adjoint(isect_params[i].v1);
        auto dv2 = get_adjoint(isect_params[i].v2);
        auto dn0 = make_vector3(0.f, 0.f, 0.f);
        auto dn1 = make_vector3(0.f, 0.f, 0.f);
        auto dn2 = make_vector3(0.f, 0.f, 0.f);
        if (has_shading_normals(*isect_params[i].shape)) {
            dn0 = get_adjoint(isect_params[i].n0);
            dn1 = get_adjoint(isect_params[i].n1);
            dn2 = get_adjoint(isect_params[i].n2);
        }
        result.shape_derivatives.push_back(
            ShapeDerivative{isect_params[i].shape, isect_params[i].tri_id,
                dv0, dv1, dv2, dn0, dn1, dn2});
    }

    // Material derivatives
    for (int i = 0; i < (int)material_params.size(); i++) {
        const auto &param = material_params[i];
        assert(param.material != nullptr);
        const auto &material = *param.material;
        TextureDerivative<Vector3f> d_diffuse;
        TextureDerivative<Vector3f> d_specular;
        TextureDerivative<float> d_roughness;
        if (has_texture(material.diffuse_reflectance)) {
            const auto &t = param.diffuse_reflectance;
            d_diffuse.xi = t.xi;
            d_diffuse.yi = t.yi;
            d_diffuse.t00 = get_adjoint(t.t00);
            d_diffuse.t01 = get_adjoint(t.t01);
            d_diffuse.t10 = get_adjoint(t.t10);
            d_diffuse.t11 = get_adjoint(t.t11);
        } else {
            const auto &t = param.diffuse_reflectance;
            d_diffuse.t00 = get_adjoint(t.t00);
        }
        if (has_texture(material.specular_reflectance)) {
            const auto &t = param.specular_reflectance;
            d_specular.xi = t.xi;
            d_specular.yi = t.yi;
            d_specular.t00 = get_adjoint(t.t00);
            d_specular.t01 = get_adjoint(t.t01);
            d_specular.t10 = get_adjoint(t.t10);
            d_specular.t11 = get_adjoint(t.t11);
        } else {
            const auto &t = param.specular_reflectance;
            d_specular.t00 = get_adjoint(t.t00);
        }
        if (has_texture(material.roughness)) {
            const auto &t = param.roughness;
            d_roughness.xi = t.xi;
            d_roughness.yi = t.yi;
            d_roughness.t00 = get_adjoint(t.t00);
            d_roughness.t01 = get_adjoint(t.t01);
            d_roughness.t10 = get_adjoint(t.t10);
            d_roughness.t11 = get_adjoint(t.t11);
        } else {
            const auto &t = param.roughness;
            auto d = get_adjoint(t.t00);
            if (isfinite(d)) {
                d_roughness.t00 = d;
            }
        }
        auto d_diffuse_uv_scale = get_adjoint(param.diffuse_uv_scale);
        auto d_specular_uv_scale = get_adjoint(param.specular_uv_scale);
        auto d_roughness_uv_scale = get_adjoint(param.roughness_uv_scale);
        result.material_derivatives.push_back(
            MaterialDerivative{&material,
                d_diffuse, d_specular, d_roughness,
                d_diffuse_uv_scale, d_specular_uv_scale, d_roughness_uv_scale});
    }
    for (int i = 0; i < (int)light_params.size(); i++) {
        const auto &param = light_params[i];
        assert(param.light != nullptr);
        result.light_derivatives.push_back(
            LightDerivative{param.light, get_adjoint(param.intensity)});
    }

    auto& dps = sample_buffer.dps;
    auto& edges = sample_buffer.edges;
    for (const auto &secondary_point : secondary_points) {
        const auto &bounce = secondary_point.bounce;
        const auto &wi = secondary_point.wi;
        const auto &isect = secondary_point.intersection;
        const auto &surface_point_ad = secondary_point.surface_point;
        const auto &surface_point = convert<float>(secondary_point.surface_point);
        const auto &throughput = secondary_point.throughput;
        auto edge_sample = make_edge_sample(rng);
        auto light_id = sample_light_id(scene, light_sample.light_sel);
        assert(light_id < (int)scene.lights.size());
        assert(scene.lights.size() == scene.light_bounds.size());
        const auto &light_bounds = scene.light_bounds[light_id];
        auto [b_center, b_radius] = light_bounds.bounding_sphere();
        auto dir_to_light = b_center -
            convert<float>(surface_point.position);
        auto tan_angle = b_radius / length(dir_to_light);
        auto cos_angle = tan_angle <= 1.f ?
            1.f / sqrt(tan_angle * tan_angle + 1.f) : 0.f;
        auto cone = Cone{make_ray(
            convert<float>(surface_point.position),
            normalize(dir_to_light)), cos_angle};
        // Sample a point on edges
        // Select a set of edges
        edges.clear();
        sample_shading_edge(scene.edge_sampler,
                            convert<float>(wi),
                            isect,
                            surface_point,
                            cone,
                            edges,
                            rng);
        for (const EdgeAndWeight &edge_and_weight : edges) {
            // Sum over edge contribution
            const Edge &edge = edge_and_weight.edge;
            auto edge_weight = edge_and_weight.weight;
            if (edge_weight == 0.f) {
                continue;
            }
            auto edge_light_sample = make_light_sample(rng);
            secondary_samples.clear();
            for (int i = 0; i < max_bounces - bounce - 1; i++) {
                std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
                secondary_samples.push_back(
                    SecondarySample{make_light_sample(rng),
                                    make_bsdf_sample(rng),
                                    uni_dist(rng)});
            }
            auto weight = throughput * (sample_weight * edge_weight);
            auto edge_sample_result =
                sample_shading_edge(scene,
                                    edge_sample,
                                    edge,
                                    edge_light_sample,
                                    secondary_samples,
                                    wi,
                                    isect,
                                    surface_point,
                                    weight,
                                    d_color,
                                    max_bounces == 1);

            auto dp = edge_sample_result.dp;
            // Compute duvw/dxy
            // if (bounce == -1) {
            //     // TODO: compute duvw/dxy also for bounces > -1
            //     auto delta = 1e-3f;
            //     auto dx_pos =
            //         make_vector2(convert<float>(x) + delta, convert<float>(y));
            //     auto dy_pos =
            //         make_vector2(convert<float>(x), convert<float>(y) + delta);
            //     auto dx_ray = sample_primary(scene.camera, dx_pos);
            //     auto dy_ray = sample_primary(scene.camera, dy_pos);
            //     auto dx_point = intersect(primary_isect, dx_ray);
            //     auto dy_point = intersect(primary_isect, dy_ray);
            //     auto dx_hit = dx_point.position -
            //                   convert<float>(shading_point.position);
            //     auto dy_hit = dy_point.position -
            //                   convert<float>(shading_point.position);
            //     auto dudx = dx_hit[0] / delta;
            //     auto dudy = dy_hit[0] / delta;
            //     auto dvdx = dx_hit[1] / delta;
            //     auto dvdy = dy_hit[1] / delta;
            //     auto dwdx = dx_hit[2] / delta;
            //     auto dwdy = dy_hit[2] / delta;
            //     assert(!isnan(du) && !isnan(dv) && !isnan(dw));
            //     auto dx = du * dudx + dv * dvdx + dw * dwdx;
            //     auto dy = du * dudy + dv * dvdy + dw * dwdy;
            //     assert(!isnan(dx) && !isnan(dy));
            //     result.color_dx += dx;
            //     result.color_dy += dy;
            // }

            if (!is_zero(dp)) {
                dps.push_back(PositionDerivative{
                    surface_point_ad.position, dp});
            }

            auto dv0 = edge_sample_result.dv0;
            auto dv1 = edge_sample_result.dv1;
            if (!is_zero(dv0) || !is_zero(dv1)) {
                result.edge_derivatives.push_back(
                    EdgeDerivative{edge, dv0, dv1});
            }
        }
    }
    // Propagate Dirac position derivatives to all dependent variables
    if (dps.size() > 0) {
        ad_graph.zero_adjoints();
        for (const auto &dp : dps) {
            const auto &p = dp.p;
            set_adjoint(p[0], dp.dp[0]);
            set_adjoint(p[1], dp.dp[1]);
            set_adjoint(p[2], dp.dp[2]);
        }
        propagate_adjoint();
        set_matrix_adjoint(camera.cam_to_world,
            result.camera_derivatives.d_cam_to_world);
        set_matrix_adjoint(camera.world_to_cam,
            result.camera_derivatives.d_world_to_cam);
        set_matrix_adjoint(camera.sample_to_cam,
            result.camera_derivatives.d_sample_to_cam);
        set_matrix_adjoint(camera.cam_to_sample,
            result.camera_derivatives.d_cam_to_sample);
        for (int i = 0; i < (int)isect_params.size(); i++) {
            auto dv0 = get_adjoint(isect_params[i].v0);
            auto dv1 = get_adjoint(isect_params[i].v1);
            auto dv2 = get_adjoint(isect_params[i].v2);
            result.shape_derivatives[i].dv0 += dv0;
            result.shape_derivatives[i].dv1 += dv1;
            result.shape_derivatives[i].dv2 += dv2;
        }
        // TODO: should propagate to materials here too (e.g. IOR)
    }

    return result;
}

struct RenderResult {
    py::array_t<float> image;
    py::array_t<float> dx_image;
    py::array_t<float> dy_image;
    DCamera d_camera;
    std::vector<DShape> d_shapes;
    std::vector<DMaterial> d_materials;
    std::vector<DLight> d_lights;
};

auto render(const Camera &camera,
            const std::vector<const Shape*> &shapes,
            const std::vector<const Material*> &materials,
            const std::vector<const Light*> &lights,
            const std::pair<int, int> &resolution,
            const py::array_t<float> &d_image,
            int num_samples,
            int max_bounces,
            int seed,
            bool print_progress) {
    // py::gil_scoped_release gil_release;
    auto prev_int_handler = signal(SIGINT, kb_interrupt_handler);
    auto compute_derivative = d_image.ndim() == 3;
    auto scene = Scene(camera, shapes, materials, lights);
    // Initialize the derivatives
    std::map<const Shape*, DShape> d_shapes_map;
    for (const Shape *shape : shapes) {
        assert(shape->vertices.ndim() == 2);
        assert(shape->vertices.shape()[1] == 3);
        auto d_vertices = py::array_t<float>(
            {shape->vertices.shape()[0], shape->vertices.shape()[1]});
        auto accessor = d_vertices.mutable_unchecked<2>();
        for (int i = 0; i < shape->vertices.shape()[0]; i++) {
            accessor(i, 0) = 0.f;
            accessor(i, 1) = 0.f;
            accessor(i, 2) = 0.f;
        }
        auto d_normals = py::array_t<float>();
        if (has_shading_normals(*shape)) {
            d_normals = py::array_t<float>(
                {shape->normals.shape()[0], shape->normals.shape()[1]});
            auto accessor = d_normals.mutable_unchecked<2>();
            for (int i = 0; i < shape->normals.shape()[0]; i++) {
                accessor(i, 0) = 0.f;
                accessor(i, 1) = 0.f;
                accessor(i, 2) = 0.f;
            }
        }
        d_shapes_map.insert(std::make_pair(shape, DShape{d_vertices, d_normals}));
    }
    std::map<const Material*, DMaterial> d_materials_map;
    for (const Material *material : materials) {
        auto d_diffuse = py::array_t<float>();
        if (material->diffuse_reflectance.ndim() == 1) {
            assert(material->diffuse_reflectance.shape()[0] == 3);
            d_diffuse = py::array_t<float>(3);
            auto accessor = d_diffuse.mutable_unchecked<1>();
            accessor(0) = 0.f;
            accessor(1) = 0.f;
            accessor(2) = 0.f;
        } else {
            assert(material->diffuse_reflectance.ndim() == 3);
            assert(material->diffuse_reflectance.shape()[2] == 3);
            d_diffuse = py::array_t<float>({
                material->diffuse_reflectance.shape()[0],
                material->diffuse_reflectance.shape()[1],
                material->diffuse_reflectance.shape()[2]});
            auto accessor = d_diffuse.mutable_unchecked<3>();
            for (int i = 0; i < material->diffuse_reflectance.shape()[0]; i++) {
                for (int j = 0; j < material->diffuse_reflectance.shape()[1]; j++) {
                    accessor(i, j, 0) = 0.f;
                    accessor(i, j, 1) = 0.f;
                    accessor(i, j, 2) = 0.f;
                }
            }
        }
        auto d_specular = py::array_t<float>();
        if (material->specular_reflectance.ndim() == 1) {
            assert(material->specular_reflectance.shape()[0] == 3);
            d_specular = py::array_t<float>(3);
            auto accessor = d_specular.mutable_unchecked<1>();
            accessor(0) = 0.f;
            accessor(1) = 0.f;
            accessor(2) = 0.f;
        } else {
            assert(material->specular_reflectance.ndim() == 3);
            assert(material->specular_reflectance.shape()[2] == 3);
            d_specular = py::array_t<float>({
                material->specular_reflectance.shape()[0],
                material->specular_reflectance.shape()[1],
                material->specular_reflectance.shape()[2]});
            auto accessor = d_specular.mutable_unchecked<3>();
            for (int i = 0; i < material->specular_reflectance.shape()[0]; i++) {
                for (int j = 0; j < material->specular_reflectance.shape()[1]; j++) {
                    accessor(i, j, 0) = 0.f;
                    accessor(i, j, 1) = 0.f;
                    accessor(i, j, 2) = 0.f;
                }
            }
        }
        auto d_roughness = py::array_t<float>();
        if (material->roughness.ndim() == 1) {
            assert(material->roughness.shape()[0] == 1);
            d_roughness = py::array_t<float>(1);
            auto accessor = d_roughness.mutable_unchecked<1>();
            accessor(0) = 0.f;
        } else {
            assert(material->roughness.ndim() == 3);
            assert(material->roughness.shape()[2] == 1);
            d_roughness = py::array_t<float>({
                material->roughness.shape()[0],
                material->roughness.shape()[1],
                material->roughness.shape()[2]});
            auto accessor = d_roughness.mutable_unchecked<3>();
            for (int i = 0; i < material->roughness.shape()[0]; i++) {
                for (int j = 0; j < material->roughness.shape()[1]; j++) {
                    accessor(i, j, 0) = 0.f;
                }
            }
        }
        d_materials_map.insert(std::make_pair(material,
            DMaterial{d_diffuse, d_specular, d_roughness}));
    }
    std::map<const Light*, DLight> d_lights_map;
    for (const Light *light : lights) {
        assert(light->intensity.ndim() == 1);
        assert(light->intensity.shape()[0] == 3);
        auto d_intensity = py::array_t<float>(3);
        auto accessor = d_intensity.mutable_unchecked<1>();
        accessor(0) = 0.f;
        accessor(1) = 0.f;
        accessor(2) = 0.f;
        d_lights_map.insert(std::make_pair(light, DLight{d_intensity}));
    }

    DCamera d_camera;
    d_camera.d_cam_to_world = py::array_t<float>({4, 4});
    d_camera.d_world_to_cam = py::array_t<float>({4, 4});
    d_camera.d_sample_to_cam = py::array_t<float>({4, 4});
    d_camera.d_cam_to_sample = py::array_t<float>({4, 4});
    auto zero_matrix = [](py::array_t<float> &mat) {
        auto accessor = mat.mutable_unchecked<2>();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                accessor(i, j) = 0.f;
            }
        }
    };
    auto accumulate_matrix =
            [](py::array_t<float> &py_mat, const auto &mat) {
        auto accessor = py_mat.mutable_unchecked<2>();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                accessor(i, j) += mat(i, j);
            }
        }
    };
    zero_matrix(d_camera.d_cam_to_world);
    zero_matrix(d_camera.d_world_to_cam);
    zero_matrix(d_camera.d_sample_to_cam);
    zero_matrix(d_camera.d_cam_to_sample);

    // Rendering
    std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
    auto width = (int)std::get<0>(resolution);
    auto height = (int)std::get<1>(resolution);
    auto img = py::array_t<float>({height, width, 3});
    auto dx_img = py::array_t<float>({height, width, 3});
    auto dy_img = py::array_t<float>({height, width, 3});
    auto img_accessor = img.mutable_unchecked<3>();
    auto dx_img_accessor = dx_img.mutable_unchecked<3>();
    auto dy_img_accessor = dy_img.mutable_unchecked<3>();
    auto tile_size = 8;
    auto num_xtiles = (width  + tile_size - 1) / tile_size;
    auto num_ytiles = (height + tile_size - 1) / tile_size;
    auto num_threads = num_xtiles * num_ytiles;
    std::mt19937 rng(seed);
    std::vector<int> seeds;
    for (int i = 0; i < num_threads + num_system_cores(); i++) {
        seeds.push_back(rng());
    }
    std::mutex vertex_buffer_mutex, material_buffer_mutex, light_buffer_mutex;
    ProgressReporter reporter(num_xtiles * num_ytiles, print_progress);
    parallel_for([&](Vector2i tile) {
        std::mt19937 rng(seeds[tile[1] * num_xtiles + tile[0]]);
        auto x0 = tile[0] * tile_size;
        auto x1 = std::min(x0 + tile_size, width);
        auto y0 = tile[1] * tile_size;
        auto y1 = std::min(y0 + tile_size, height);

        CameraDerivative thread_d_cam_buffer;
        struct VertexDerivative {
            int xi, yi;
            const Shape *shape;
            int vid;
            Vector3f dv, dn;
        };
        std::vector<VertexDerivative> vertex_derivative_buffer;
        std::vector<MaterialDerivative> material_derivative_buffer;
        std::vector<LightDerivative> light_derivative_buffer;
        auto max_buffer_size = 65536;
        ADGraph ad_graph;
        for (int yi = y0; yi < y1; yi++) {
            for (int xi = x0; xi < x1; xi++) {
                img_accessor(yi, xi, 0) = 0.0f;
                img_accessor(yi, xi, 1) = 0.0f;
                img_accessor(yi, xi, 2) = 0.0f;
                dx_img_accessor(yi, xi, 0) = 0.0f;
                dx_img_accessor(yi, xi, 1) = 0.0f;
                dx_img_accessor(yi, xi, 2) = 0.0f;
                dy_img_accessor(yi, xi, 0) = 0.0f;
                dy_img_accessor(yi, xi, 1) = 0.0f;
                dy_img_accessor(yi, xi, 2) = 0.0f;
                auto color_sum = make_vector3(0.f, 0.f, 0.f);
                // auto sqrt_num_samples = (int)sqrt(num_samples);
                // assert(sqrt_num_samples * sqrt_num_samples == num_samples);
                auto d_color = make_vector3(0.f, 0.f, 0.f);
                if (compute_derivative) {
                    auto d_img_accessor = d_image.unchecked<3>();
                    d_color = make_vector3(
                        d_img_accessor(yi, xi, 0),
                        d_img_accessor(yi, xi, 1),
                        d_img_accessor(yi, xi, 2));
                }
                auto dx_color_sum = make_vector3(0.f, 0.f, 0.f);
                auto dy_color_sum = make_vector3(0.f, 0.f, 0.f);
                auto sample_buffer = SampleBuffer{};
                for (int sample_id = 0; sample_id < num_samples; sample_id++) {
                    // Box filter
                    auto sub_x = uni_dist(rng) / num_samples;
                    auto sub_y = uni_dist(rng) / num_samples;
                    // Importance sampled tent filter
                    // auto r0 = 2.f * uni_dist(rng);
                    // auto r1 = 2.f * uni_dist(rng);
                    // auto dx = r0 < 1.f ? sqrt(r0) - 1.f : 1.f - sqrt(2.f - r0);
                    // auto dy = r1 < 1.f ? sqrt(r1) - 1.f : 1.f - sqrt(2.f - r1);
                    // auto sub_x = (sample_x + 0.5f + dx) / sqrt_num_samples;
                    // auto sub_y = (sample_y + 0.5f + dy) / sqrt_num_samples;
                    auto sample = render_sample(
                        scene, width, height,
                        xi, yi, sub_x, sub_y, num_samples, max_bounces,
                        d_color, ad_graph, rng, sample_buffer,
                        compute_derivative);
                    color_sum += sample.color;
                    if (compute_derivative) {
                        dx_color_sum[0] += sample_buffer.edges.size() / 1000.f;
                        dx_color_sum[1] += sample_buffer.edges.size() / 1000.f;
                        dx_color_sum[2] += sample_buffer.edges.size() / 1000.f;
                        accumulate(thread_d_cam_buffer, sample.camera_derivatives);
                        for (const auto &it : sample.shape_derivatives) {
                            assert(it.shape != nullptr);
                            assert(d_shapes_map.find(it.shape) != 
                                    d_shapes_map.end());
                            auto ind = get_indices(*it.shape, it.tri_id);
                            vertex_derivative_buffer.push_back(
                                    VertexDerivative{
                                    xi, yi, it.shape, ind[0], it.dv0, it.dn0});
                            vertex_derivative_buffer.push_back(
                                    VertexDerivative{
                                    xi, yi, it.shape, ind[1], it.dv1, it.dn0});
                            vertex_derivative_buffer.push_back(
                                    VertexDerivative{
                                    xi, yi, it.shape, ind[2], it.dv2, it.dn0});
                            // if (it.shape->vertices.shape()[0] > 100) {
                            //     dx_color_sum[0] += (it.dv0[0] + it.dv1[0] + it.dv2[0]) / 3.f;
                            //     dx_color_sum[1] += (it.dv0[0] + it.dv1[0] + it.dv2[0]) / 3.f;
                            //     dx_color_sum[2] += (it.dv0[0] + it.dv1[0] + it.dv2[0]) / 3.f;
                            //     dy_color_sum[0] += (it.dv0[1] + it.dv1[1] + it.dv2[1]) / 3.f;
                            //     dy_color_sum[1] += (it.dv0[1] + it.dv1[1] + it.dv2[1]) / 3.f;
                            //     dy_color_sum[2] += (it.dv0[1] + it.dv1[1] + it.dv2[1]) / 3.f;
                            // }
                        }
                        for (const auto &it : sample.edge_derivatives) {
                            assert(it.edge.shape != nullptr);
                            vertex_derivative_buffer.push_back(
                                    VertexDerivative{
                                    xi, yi,
                                    it.edge.shape, it.edge.v0, it.dv0,
                                    make_vector3(0.f, 0.f, 0.f)});
                            vertex_derivative_buffer.push_back(
                                    VertexDerivative{
                                    xi, yi,
                                    it.edge.shape, it.edge.v1, it.dv1,
                                    make_vector3(0.f, 0.f, 0.f)});
                            // if (it.edge.shape->vertices.shape()[0] > 100) {
                            // // if (false) {
                            //     dx_color_sum[0] += (it.dv0[0] + it.dv1[0]) / 3.f;
                            //     dx_color_sum[1] += (it.dv0[0] + it.dv1[0]) / 3.f;
                            //     dx_color_sum[2] += (it.dv0[0] + it.dv1[0]) / 3.f;
                            //     dy_color_sum[0] += (it.dv0[1] + it.dv1[1]) / 3.f;
                            //     dy_color_sum[1] += (it.dv0[1] + it.dv1[1]) / 3.f;
                            //     dy_color_sum[2] += (it.dv0[1] + it.dv1[1]) / 3.f;
                            // }
                        }
                        material_derivative_buffer.insert(
                                material_derivative_buffer.end(),
                                sample.material_derivatives.begin(),
                                sample.material_derivatives.end());
                        // if (false) {
                        //     for (const auto &m : sample.material_derivatives) {
                        //         if (m.material == materials[7]) {
                        //             dx_color_sum[0] += m.diffuse_reflectance.t00[0];
                        //             dx_color_sum[1] += m.diffuse_reflectance.t00[1];
                        //             dx_color_sum[2] += m.diffuse_reflectance.t00[2];
                        //         }
                        //     }
                        // }
                        light_derivative_buffer.insert(
                                light_derivative_buffer.end(),
                                sample.light_derivatives.begin(),
                                sample.light_derivatives.end());
                        // if (false) {
                        //     for (const auto &l : sample.light_derivatives) {
                        //         dx_color_sum += l.d_intensity;
                        //     }
                        // }
                        }
                }
                img_accessor(yi, xi, 0) += color_sum[0];
                img_accessor(yi, xi, 1) += color_sum[1];
                img_accessor(yi, xi, 2) += color_sum[2];
                dx_img_accessor(yi, xi, 0) += dx_color_sum[0];
                dx_img_accessor(yi, xi, 1) += dx_color_sum[1];
                dx_img_accessor(yi, xi, 2) += dx_color_sum[2];
                dy_img_accessor(yi, xi, 0) += dy_color_sum[0];
                dy_img_accessor(yi, xi, 1) += dy_color_sum[1];
                dy_img_accessor(yi, xi, 2) += dy_color_sum[2];
                if (compute_derivative) {
                    // Write from local buffer to global buffer
                    if ((int)vertex_derivative_buffer.size() > max_buffer_size ||
                        (yi == y1 - 1 && xi == x1 - 1)) {
                        std::unique_lock<std::mutex> guard(vertex_buffer_mutex);
                        accumulate_matrix(d_camera.d_cam_to_world, thread_d_cam_buffer.d_cam_to_world);
                        accumulate_matrix(d_camera.d_world_to_cam, thread_d_cam_buffer.d_world_to_cam);
                        accumulate_matrix(d_camera.d_sample_to_cam, thread_d_cam_buffer.d_sample_to_cam);
                        accumulate_matrix(d_camera.d_cam_to_sample, thread_d_cam_buffer.d_cam_to_sample);
                        thread_d_cam_buffer = CameraDerivative{};
                        for (const auto &sample : vertex_derivative_buffer) {
                            assert(sample.shape != nullptr);
                            assert(d_shapes_map.find(sample.shape) != 
                                   d_shapes_map.end());
                            DShape &d_shape = d_shapes_map[sample.shape];
                            accumulate_derivative(d_shape, sample.vid, sample.dv, sample.dn);
                        }
                        vertex_derivative_buffer.clear();
                    }
                    if ((int)material_derivative_buffer.size() > max_buffer_size ||
                        (yi == y1 - 1 && xi == x1 - 1)) {
                        std::unique_lock<std::mutex> guard(material_buffer_mutex);
                        for (const auto &sample : material_derivative_buffer) {
                            assert(sample.material != nullptr);
                            assert(d_materials_map.find(sample.material) != 
                                   d_materials_map.end());
                            const auto &material = *sample.material;
                            DMaterial &d_material = d_materials_map[sample.material];
                            if (has_texture(material.diffuse_reflectance)) {
                                const auto &t = sample.diffuse_reflectance;
                                auto width = d_material.diffuse_reflectance.shape()[1];
                                auto height = d_material.diffuse_reflectance.shape()[0];
                                auto xf = t.xi;
                                auto yf = t.yi;
                                auto xc = modulo(xf + 1, width);
                                auto yc = modulo(yf + 1, height);
                                auto accessor =
                                    d_material.diffuse_reflectance.mutable_unchecked<3>();
                                accessor(yc, xc, 0) += t.t00[0];
                                accessor(yc, xc, 1) += t.t00[1];
                                accessor(yc, xc, 2) += t.t00[2];
                                accessor(yc, xf, 0) += t.t01[0];
                                accessor(yc, xf, 1) += t.t01[1];
                                accessor(yc, xf, 2) += t.t01[2];
                                accessor(yf, xc, 0) += t.t10[0];
                                accessor(yf, xc, 1) += t.t10[1];
                                accessor(yf, xc, 2) += t.t10[2];
                                accessor(yf, xf, 0) += t.t11[0];
                                accessor(yf, xf, 1) += t.t11[1];
                                accessor(yf, xf, 2) += t.t11[2];
                            } else {
                                const auto &t = sample.diffuse_reflectance;
                                auto accessor =
                                    d_material.diffuse_reflectance.mutable_unchecked<1>();
                                accessor(0) += t.t00[0];
                                accessor(1) += t.t00[1];
                                accessor(2) += t.t00[2];
                            }
                            if (has_texture(material.specular_reflectance)) {
                                const auto &t = sample.specular_reflectance;
                                auto width = d_material.specular_reflectance.shape()[1];
                                auto height = d_material.specular_reflectance.shape()[0];
                                auto xf = t.xi;
                                auto yf = t.yi;
                                auto xc = modulo(xf + 1, width);
                                auto yc = modulo(yf + 1, height);
                                auto accessor =
                                    d_material.specular_reflectance.mutable_unchecked<3>();
                                accessor(yc, xc, 0) += t.t00[0];
                                accessor(yc, xc, 1) += t.t00[1];
                                accessor(yc, xc, 2) += t.t00[2];
                                accessor(yc, xf, 0) += t.t01[0];
                                accessor(yc, xf, 1) += t.t01[1];
                                accessor(yc, xf, 2) += t.t01[2];
                                accessor(yf, xc, 0) += t.t10[0];
                                accessor(yf, xc, 1) += t.t10[1];
                                accessor(yf, xc, 2) += t.t10[2];
                                accessor(yf, xf, 0) += t.t11[0];
                                accessor(yf, xf, 1) += t.t11[1];
                                accessor(yf, xf, 2) += t.t11[2];
                            } else {
                                const auto &t = sample.specular_reflectance;
                                auto accessor =
                                    d_material.specular_reflectance.mutable_unchecked<1>();
                                accessor(0) += t.t00[0];
                                accessor(1) += t.t00[1];
                                accessor(2) += t.t00[2];
                            }
                            if (has_texture(material.roughness)) {
                                const auto &t = sample.roughness;
                                auto width = d_material.roughness.shape()[1];
                                auto height = d_material.roughness.shape()[0];
                                auto xf = t.xi;
                                auto yf = t.yi;
                                auto xc = modulo(xf + 1, width);
                                auto yc = modulo(yf + 1, height);
                                auto accessor =
                                    d_material.roughness.mutable_unchecked<3>();
                                accessor(yc, xc, 0) += t.t00;
                                accessor(yc, xf, 0) += t.t01;
                                accessor(yf, xc, 0) += t.t10;
                                accessor(yf, xf, 0) += t.t11;
                            } else {
                                const auto &t = sample.roughness;
                                auto accessor =
                                    d_material.roughness.mutable_unchecked<1>();
                                accessor(0) += t.t00;
                            }
                        }
                        material_derivative_buffer.clear();
                    }
                    if ((int)light_derivative_buffer.size() > max_buffer_size ||
                            (yi == y1 - 1 && xi == x1 - 1)) {
                        std::unique_lock<std::mutex> guard(light_buffer_mutex);
                        for (const auto &sample : light_derivative_buffer) {
                            assert(sample.light != nullptr);
                            assert(d_lights_map.find(sample.light) != 
                                   d_lights_map.end());
                            DLight &d_light = d_lights_map[sample.light];
                            auto accessor = d_light.intensity.mutable_unchecked<1>();
                            accessor(0) += sample.d_intensity[0];
                            accessor(1) += sample.d_intensity[1];
                            accessor(2) += sample.d_intensity[2];
                        }
                        light_derivative_buffer.clear();
                    }
                }
            }
        }
        reporter.update(1);
    }, make_vector2(num_xtiles, num_ytiles));
    terminate_worker_threads();
    reporter.done();

    if (compute_derivative) {
        ProgressReporter edge_reporter((long long)num_samples * (width * height),
                                       print_progress);
        // Sample the edges and project onto the camera
        parallel_for([&](int thread_index) {
            std::mt19937 rng(seeds[num_threads + thread_index]);
            int samples_per_thread =
                (long long)(num_samples * (width * height)) / num_system_cores();
            std::vector<PrimaryEdgeSampleResult> derivative_buffer;
            auto max_buffer_size = 65536;
            for (int sample_id = 0; sample_id < samples_per_thread; sample_id++) {
                auto edge_and_weight =
                    sample_primary_edge(scene.edge_sampler, uni_dist(rng));
                if (edge_and_weight.weight <= 0.f) {
                    // Numerical issue or no primary edges
                    continue;
                }
                auto edge_sample = make_edge_sample(rng);
                auto light_sample = make_light_sample(rng);
                auto secondary_samples = std::vector<SecondarySample>{};
                for (int i = 0; i < max_bounces; i++) {
                    std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
                    secondary_samples.push_back(
                        SecondarySample{make_light_sample(rng),
                                        make_bsdf_sample(rng),
                                        uni_dist(rng)});
                }
                auto primary_edge_sample_result =
                    sample_primary_edge(scene,
                                        edge_sample,
                                        edge_and_weight.edge,
                                        light_sample,
                                        secondary_samples,
                                        d_image,
                                        edge_and_weight.weight / num_samples);
                derivative_buffer.push_back(primary_edge_sample_result);
                if ((int)derivative_buffer.size() >= max_buffer_size ||
                        sample_id == samples_per_thread - 1) {
                    // Write local buffer to global buffer
                    std::unique_lock<std::mutex> guard(vertex_buffer_mutex);
                    for (const auto &sample : derivative_buffer) {
                        if (sample.screen_x < 0.f || sample.screen_y < 0.f ||
                            sample.screen_x > 1.f || sample.screen_y > 1.f) {
                            continue;
                        }
                        assert(sample.edge.shape != nullptr);
                        DShape &d_shape = d_shapes_map[sample.edge.shape];
                        auto dv0 = sample.dv0;
                        auto dv1 = sample.dv1;
                        accumulate_derivative(d_shape, sample.edge.v0, dv0);
                        accumulate_derivative(d_shape, sample.edge.v1, dv1);
                        accumulate_matrix(
                            d_camera.d_cam_to_sample, sample.d_cam_to_sample);
                        accumulate_matrix(
                            d_camera.d_world_to_cam, sample.d_world_to_cam);
                        // int xi = sample.screen_x * width;
                        // int yi = sample.screen_y * height;
                        // if (sample.edge.shape->vertices.shape()[0] > 100) {
                        // // if (true) {
                        //     dx_img_accessor(yi, xi, 0) +=
                        //         (sample.dv0[0] + sample.dv1[0]) / 3.f;
                        //     dx_img_accessor(yi, xi, 1) +=
                        //         (sample.dv0[0] + sample.dv1[0]) / 3.f;
                        //     dx_img_accessor(yi, xi, 2) +=
                        //         (sample.dv0[0] + sample.dv1[0]) / 3.f;
                        //     dy_img_accessor(yi, xi, 0) +=
                        //         (sample.dv0[1] + sample.dv1[1]) / 3.f;
                        //     dy_img_accessor(yi, xi, 1) +=
                        //         (sample.dv0[1] + sample.dv1[1]) / 3.f;
                        //     dy_img_accessor(yi, xi, 2) +=
                        //         (sample.dv0[1] + sample.dv1[1]) / 3.f;
                        // }
                    }
                    derivative_buffer.clear();
                }
                if (sample_id > 0 && sample_id % 65536 == 0) {
                    edge_reporter.update(65536);
                }
            }
        }, num_system_cores());
        terminate_worker_threads();
        edge_reporter.done();
    }

    std::vector<DShape> d_shapes;
    for (const Shape *shape : shapes) {
        d_shapes.push_back(d_shapes_map[shape]);
    }

    std::vector<DMaterial> d_materials;
    for (const Material *material : materials) {
        d_materials.push_back(d_materials_map[material]);
    }

    std::vector<DLight> d_lights;
    for (const Light *light : lights) {
        d_lights.push_back(d_lights_map[light]);
    }

    signal(SIGINT, prev_int_handler);
    return RenderResult{img, dx_img, dy_img, d_camera,
        d_shapes, d_materials, d_lights};
}

void test_line_integral() {
    auto p1 = Vector3f{-10.f, 3.f, 0.f};
    auto p2 = Vector3f{ 10.f, 3.f, 0.f};
    auto y = 3.f;
    // Assume a constant function at y
    // Integrate over p2 - p1
    std::mt19937 rng;
    // Uniform distribution on line
    auto num_samples = 65536;
    auto uniform_line_mean = 0.f;
    auto uniform_line_m2 = 0.f;
    for (int i = 0; i < num_samples; i++) {
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        auto u = uni_dist(rng);
        auto p = p1 + u * (p2 - p1);
        auto len_p = length(p);
        auto dir = p / len_p;
        // intersect dir with y = 5
        auto t = y / dir[1];
        auto hit_pos = dir * t;
        auto norm_jacobian = normalize_jacobian(p);
        auto isect_jacobian =
            intersect_jacobian(dir, hit_pos, Vector3f{0.f, -1.f, 0.f});
        auto jacobian = length(isect_jacobian * norm_jacobian * (p2 - p1));
        auto brdf = dir[1] / float(M_PI); // cosine
        // auto geom = 2.f * length(cross(normalize(hit_pos), normalize(p2-p1))) / length_squared(hit_pos);
        auto geom = dot(Vector3f{0.f,1.f,0.f}, dir) / length_squared(hit_pos);
        auto val = brdf * geom * jacobian;
        {
            auto delta = val - uniform_line_mean;
            uniform_line_mean += (delta / float(i + 1));
            auto delta_2 = val - uniform_line_mean;
            uniform_line_m2 += delta * delta_2;
        }
    }
    auto uniform_line_variance = uniform_line_m2 / float(num_samples);
    std::cout << "Uniform line:" << uniform_line_mean << std::endl;
    std::cout << "Uniform variance:" << uniform_line_variance << std::endl;
    auto cos_line_mean = 0.f;
    auto cos_line_m2 = 0.f;
    for (int i = 0; i < num_samples; i++) {
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        auto u = uni_dist(rng);
        auto wt = normalize(p2 - p1);
        auto l1 = dot(p1, wt);
        auto l2 = dot(p2, wt);
        auto po = p1 - l1 * wt;
        auto d = length(po);
        auto normal = Vector3f{0.f, 1.f, 0.f};
        auto I = [&](float l) {
            return (l/(d*(d*d+l*l))+atan(l/d)/(d*d))*dot(po,normal) +
                   (l*l/(d*(d*d+l*l)))*dot(wt,normal);
        };
        auto Il1 = I(l1);
        auto Il2 = I(l2);
        auto normalization = Il2 - Il1;
        auto cdf = [&](float l) {
            return (I(l)-I(l1))/normalization;
        };
        auto pdf = [&](float l) {
            auto dist_sq = d * d + l * l;
            return 2.f*d*dot(po+l*wt,normal)/(normalization*dist_sq*dist_sq);
        };
        // Hybrid bisection & Newton iteration
        auto l = 0.f;
        auto lb = l1;
        auto ub = l2;
        if (lb > ub) {
            std::swap(lb, ub);
        }
        for (int it = 0; it < 20; it++) {
            if (!(l >= lb && l <= ub)) {
                l = 0.5f * (lb + ub);
            }
            auto value = cdf(l) - u;
            auto derivative = pdf(l);
            if (fabs(value) < 1e-5f) {
                break;
            }
            if (value > 0.f) {
                ub = l;
            } else {
                lb = l;
            }
            l -= value / derivative;
        }
        auto p = po + l * wt;
        auto dir = normalize(p);
        // intersect dir with y = 5
        auto t = y / dir[1];
        auto hit_pos = dir * t;
        auto norm_jacobian = normalize_jacobian(p);
        auto isect_jacobian =
            intersect_jacobian(dir, hit_pos, Vector3f{0.f, -1.f, 0.f});
        auto jacobian = length(isect_jacobian * norm_jacobian * wt) / pdf(l);
        // auto geom = 2.f * length(cross(normalize(hit_pos), normalize(p2-p1))) / length_squared(hit_pos);
        auto geom = dot(Vector3f{0.f,1.f,0.f}, dir) / length_squared(hit_pos);
        auto brdf = dir[1] / float(M_PI); // cosine
        auto val = brdf * geom * jacobian;
        {
            auto delta = val - cos_line_mean;
            cos_line_mean += (delta / float(i + 1));
            auto delta_2 = val - cos_line_mean;
            cos_line_m2 += delta * delta_2;
        }
        // auto integral = normalization / float(M_PI);
        // std::cerr << "integral:" << integral << std::endl;
        // return;
    }
    auto cosine_line_variance = cos_line_m2 / float(num_samples);
    std::cout << "Cosine line:" << cos_line_mean << std::endl;
    std::cout << "Cosine variance:" << cosine_line_variance << std::endl;
    auto uniform_line_glossy_mean = 0.f;
    auto uniform_line_glossy_m2 = 0.f;
    auto frame = make_frame(Vector3f{0.f, 1.f, 0.f});
    auto Minv = Matrix3x3f{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Minv(i, j) = 0.f;
        }
    }
    Minv(0, 0) = 3.f;
    Minv(1, 1) = 3.f;
    Minv(2, 2) = 1.f;
    std::cerr << "[before] Minv:" << std::endl;
    std::cerr << Minv << std::endl;
    std::cerr << "frame:" << std::endl;
    std::cerr << make_matrix3x3(frame) << std::endl;
    Minv = Minv * make_matrix3x3(frame);
    std::cerr << "[after] Minv:" << std::endl;
    std::cerr << Minv << std::endl;
    auto M = Matrix3x3f{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M(i, j) = 0.f;
        }
    }
    M(0, 0) = 1.f/3.f;
    M(1, 1) = 1.f/3.f;
    M(2, 2) = 1.f/1.f;
    M = transpose(make_matrix3x3(frame)) * M;
    auto det = fabs(determinant(Minv));
    for (int i = 0; i < num_samples; i++) {
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        auto u = uni_dist(rng);
        auto p = p1 + u * (p2 - p1);
        auto dir = normalize(p);
        // intersect dir with y = 5
        auto t = y / dir[1];
        auto hit_pos = dir * t;
        auto norm_jacobian = normalize_jacobian(p);
        auto isect_jacobian =
            intersect_jacobian(dir, hit_pos, Vector3f{0.f, -1.f, 0.f});
        auto jacobian = length(isect_jacobian * norm_jacobian * (p2 - p1));
        auto transformed_dir = Minv * dir;
        auto transformed_dir_length = length(transformed_dir);
        auto normalized_transformed_dir = transformed_dir / transformed_dir_length;
        auto D0 = normalized_transformed_dir[2] / float(M_PI);
        auto Mjacobian = det / pow(transformed_dir_length, 3.f);
        auto brdf = D0 * Mjacobian;
        auto geom = dot(Vector3f{0.f,1.f,0.f}, dir) / length_squared(hit_pos);
        auto val = brdf * geom * jacobian;
        {
            auto delta = val - uniform_line_glossy_mean;
            uniform_line_glossy_mean += (delta / float(i + 1));
            auto delta_2 = val - uniform_line_glossy_mean;
            uniform_line_glossy_m2 += delta * delta_2;
        }
    }
    auto uniform_line_glossy_variance = uniform_line_glossy_m2 / float(num_samples);
    std::cout << "Uniform line glossy:" << uniform_line_glossy_mean << std::endl;
    std::cout << "Uniform line glossy variance:" << uniform_line_glossy_variance << std::endl;
    auto cos_line_glossy_mean = 0.f;
    auto cos_line_glossy_m2 = 0.f;
    for (int i = 0; i < num_samples; i++) {
        std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
        auto p1o = Minv * p1;
        auto p2o = Minv * p2;
        auto u = uni_dist(rng);
        auto wt = normalize(p2o - p1o);
        auto l1 = dot(p1o, wt);
        auto l2 = dot(p2o, wt);
        auto po = p1o - l1 * wt;
        auto d = length(po);
        // auto normal = Vector3f{0.f, 1.f, 0.f};
        auto I = [&](float l) {
            return (l/(d*(d*d+l*l))+atan(l/d)/(d*d))*po[2] +
                   (l*l/(d*(d*d+l*l)))*wt[2];
        };
        auto Il1 = I(l1);
        auto Il2 = I(l2);
        auto normalization = Il2 - Il1;
        auto cdf = [&](float l) {
            return (I(l)-Il1)/normalization;
        };
        auto pdf = [&](float l) {
            auto dist_sq = d * d + l * l;
            return 2.f*d*(po+l*wt)[2]/(normalization*dist_sq*dist_sq);
        };
        // Hybrid bisection & Newton iteration
        auto l = 0.f;
        auto lb = l1;
        auto ub = l2;
        if (lb > ub) {
            std::swap(lb, ub);
        }
        for (int it = 0; it < 20; it++) {
            if (!(l >= lb && l <= ub)) {
                l = 0.5f * (lb + ub);
            }
            auto value = cdf(l) - u;
            auto derivative = pdf(l);
            if (fabs(value) < 1e-5f) {
                break;
            }
            if (value > 0.f) {
                ub = l;
            } else {
                lb = l;
            }
            l -= value / derivative;
        }
        // Convert from l to p
        auto p = po + l * wt;
        // Convert the position back to world space
        auto Mp = M * p;
        // Convert the direction back to original space
        auto dir = normalize(Mp);
        // intersect dir with y = 5
        auto t = y / dir[1];
        auto hit_pos = dir * t;

        // Jacobian from l to p = wt
        // Jacobian from p to Mp = M
        // Jacobian from Mp to dir
        auto norm_jacobian = normalize_jacobian(Mp);
        // Jacobian from dir to hit_pos
        auto isect_jacobian =
            intersect_jacobian(dir, hit_pos, Vector3f{0.f, -1.f, 0.f});
        auto transformed_dir = Minv * dir;
        auto transformed_dir_length = length(transformed_dir);
        auto Mjacobian = det / pow(transformed_dir_length, 3.f);
        auto jacobian = length(isect_jacobian * norm_jacobian * M * wt) / pdf(l);
        auto normalized_transformed_dir = transformed_dir / transformed_dir_length;
        auto D0 = normalized_transformed_dir[2] / float(M_PI);
        auto brdf = D0 * Mjacobian;
        auto geom = dot(Vector3f{0.f,1.f,0.f}, dir) / length_squared(hit_pos);
        auto val = brdf * geom * jacobian;
        {
            auto delta = val - cos_line_glossy_mean;
            cos_line_glossy_mean += (delta / float(i + 1));
            auto delta_2 = val - cos_line_glossy_mean;
            cos_line_glossy_m2 += delta * delta_2;
        }
        // // Closed-form:
        // auto MTw = length(M * normalize(cross(p1, p2)));
        // auto integral = normalization / (float(M_PI) * MTw);
        // std::cerr << "integral:" << integral << std::endl;
        // return;
    }
    auto cos_line_glossy_variance = cos_line_glossy_m2 / float(num_samples);
    std::cout << "Cosine line glossy:" << cos_line_glossy_mean << std::endl;
    std::cout << "Cosine line glossy variance:" <<
        cos_line_glossy_variance << std::endl;
}

PYBIND11_MODULE(delta_ray, m) {
    m.doc() = "Delta Ray"; // optional module docstring

    py::class_<Material>(m, "Material")
        .def(py::init<py::array_t<float>, // diffuse reflectance
                      py::array_t<float>, // specular reflectance
                      py::array_t<float>, // roughness
                      py::array_t<float>, // diffuse uv scale
                      py::array_t<float>, // specular uv scale
                      py::array_t<float>, // roughness uv scale
                      bool>());

    py::class_<DMaterial>(m, "DMaterial")
        .def(py::init<py::array_t<float>, // diffuse reflectance
                      py::array_t<float>, // specular reflectance
                      py::array_t<float>, // roughness
                      py::array_t<float>, // diffuse uv scale
                      py::array_t<float>, // specular uv scale
                      py::array_t<float>> // roughness uv scale
                      ()) 
        .def_readwrite("diffuse_reflectance", &DMaterial::diffuse_reflectance)
        .def_readwrite("specular_reflectance", &DMaterial::specular_reflectance)
        .def_readwrite("roughness", &DMaterial::roughness)
        .def_readwrite("diffuse_uv_scale", &DMaterial::diffuse_uv_scale)
        .def_readwrite("specular_uv_scale", &DMaterial::specular_uv_scale)
        .def_readwrite("roughness_uv_scale", &DMaterial::roughness_uv_scale);

    py::class_<Shape>(m, "Shape")
        .def(py::init<py::array_t<float>,
                      py::array_t<int>,
                      py::array_t<float>,
                      py::array_t<float>,
                      const Material*,
                      const Light*>())
        .def_readwrite("light", &Shape::light);

    py::class_<DShape>(m, "DShape")
        .def(py::init<py::array_t<float>,
                      py::array_t<float>>())
        .def_readwrite("vertices", &DShape::vertices)
        .def_readwrite("normals", &DShape::normals);

    py::class_<Light>(m, "Light")
        .def(py::init<const Shape*,
                      py::array_t<float>>());

    py::class_<DLight>(m, "DLight")
        .def(py::init<py::array_t<float>>())
        .def_readwrite("intensity", &DLight::intensity);

    py::class_<Camera>(m, "Camera")
        .def(py::init<py::array_t<float>,
                      py::array_t<float>,
                      py::array_t<float>,
                      py::array_t<float>,
                      float,
                      float,
                      float,
                      bool>());

    py::class_<DCamera>(m, "DCamera")
        .def(py::init<py::array_t<float>,
                      py::array_t<float>,
                      py::array_t<float>,
                      py::array_t<float>>())
        .def_readwrite("d_cam_to_world", &DCamera::d_cam_to_world)
        .def_readwrite("d_world_to_cam", &DCamera::d_world_to_cam)
        .def_readwrite("d_sample_to_cam", &DCamera::d_sample_to_cam)
        .def_readwrite("d_cam_to_sample", &DCamera::d_cam_to_sample);

    py::class_<RenderResult>(m, "RenderResult")
        .def(py::init<>())
        .def_readwrite("image", &RenderResult::image)
        .def_readwrite("dx_image", &RenderResult::dx_image)
        .def_readwrite("dy_image", &RenderResult::dy_image)
        .def_readwrite("d_camera", &RenderResult::d_camera)
        .def_readwrite("d_shapes", &RenderResult::d_shapes)
        .def_readwrite("d_materials", &RenderResult::d_materials)
        .def_readwrite("d_lights", &RenderResult::d_lights);

    m.def("test_line_integral", &test_line_integral, "");
    m.def("render", &render, "render a scene and return an image and derivatives");
}
