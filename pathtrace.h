#pragma once

#include "scene.h"
#include "material.h"
#include "light.h"

template <typename T>
auto eval(const Scene            &scene,
          const TVector3<T>      &wi,
          const Intersection     &shading_isect,
          const TSurfacePoint<T> &shading_point,
          const Intersection     &light_isect,
          const TSurfacePoint<T> &light_point,
          bool test_visibility = true,
          std::vector<MaterialParameter<T>> *material_params = nullptr,
          std::vector<LightParameter<T>> *light_params = nullptr) {
    if (!shading_isect.valid() || !light_isect.valid()) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }
    const Shape &shape = *shading_isect.shape;
    const Material &material = *shape.material;
    const Shape &light_shape = *light_isect.shape;
    if (light_shape.light == nullptr) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }
    const Light &light = *light_shape.light;
    // Direction
    auto dir_to_light = light_point.position - shading_point.position;
    auto dist_to_light = length(dir_to_light);
    if (dist_to_light == 0.f) {
        // self intersection?
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }
    auto n_dir_to_light = dir_to_light / dist_to_light;

    // Shadow
    if (occluded(scene,
                 make_ray(convert<float>(shading_point.position),
                          convert<float>(n_dir_to_light)),
                 convert<float>(dist_to_light))) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }

    // Compute BRDF & light contribution
    auto bsdf_contrib =
        bsdf(material, shading_point, wi, n_dir_to_light, material_params);
    assert(!isnan(convert<float>(bsdf_contrib)));
    auto light_cos = dot(light_point.shading_frame[2], -n_dir_to_light);
    if (light_cos <= 0.f) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }
    auto dist_squared = square(dist_to_light);
    auto light_contrib = get_intensity(light, light_params) *
                         (light_cos / dist_squared);
    assert(!isnan(convert<float>(light_contrib)));
    return bsdf_contrib * light_contrib;
}

template <typename T>
auto next_event_estimation(const Scene &scene,
                           const LightSample &light_sample,
                           const TVector3<T> &wi,
                           const Intersection &shading_isect,
                           const TSurfacePoint<T> &shading_point,
                           std::vector<IntersectParameter<T>> *isect_params = nullptr,
                           std::vector<MaterialParameter<T>> *material_params = nullptr,
                           std::vector<LightParameter<T>> *light_params = nullptr) {
    if (!shading_isect.valid()) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }

    // Sample a point on light sources
    auto [light_isect, light_point] =
      sample_point_on_light(scene, light_sample, isect_params);
    auto light_dir = normalize(light_point.position - shading_point.position);
    auto light_cos = fabs(dot(light_dir, light_point.geom_normal));
    auto sampled_light_pdf = light_pdf(scene, light_isect, isect_params);
    // Don't do autodiff for MIS (is this correct?)
    // TODO: find a cleaner way to do this
    auto sampled_bsdf_pdf =
        bsdf_pdf(*shading_isect.shape->material,
            convert<float>(shading_point),
            convert<float>(wi),
            convert<float>(light_dir)) *
        convert<float>(light_cos) /
        length_squared(
          convert<float>(light_point.position) -
          convert<float>(shading_point.position));
    auto mis_weight =
        convert<float>(square(sampled_light_pdf) /
        (square(sampled_bsdf_pdf) + square(sampled_light_pdf)));

    return eval<T>(scene,
                wi,
                shading_isect,
                shading_point,
                light_isect,
                light_point,
                true,
                material_params,
                light_params) * (mis_weight / sampled_light_pdf);
}

template <typename T>
auto sample_bsdf(const Scene &scene,
                 const BSDFSample &bsdf_sample,
                 const TVector3<T> &wi,
                 const Intersection &shading_isect,
                 const TSurfacePoint<T> &shading_point,
                 std::vector<IntersectParameter<T>> *isect_params = nullptr,
                 std::vector<MaterialParameter<T>> *material_params = nullptr) {
    if (!shading_isect.valid()) {
        return std::make_tuple(Intersection{}, TSurfacePoint<T>{}, false);
    }
    const Material &material = *shading_isect.shape->material;
    if (material.two_sided) {
        if (dot(wi, shading_point.shading_frame[2]) < 0.f) {
            auto flipped_shading_point = shading_point;
            flipped_shading_point.shading_frame[0] =
                -flipped_shading_point.shading_frame[0];
            flipped_shading_point.shading_frame[1] =
                -flipped_shading_point.shading_frame[1];
            flipped_shading_point.shading_frame[2] =
                -flipped_shading_point.shading_frame[2];
            flipped_shading_point.geom_normal = -flipped_shading_point.geom_normal;
            return sample_bsdf(scene,
                               bsdf_sample,
                               wi,
                               shading_isect,
                               flipped_shading_point,
                               isect_params,
                               material_params);
        }
    }

    auto sample_result = sample_bsdf(*shading_isect.shape->material,
                                     shading_point,
                                     wi,
                                     bsdf_sample,
                                     material_params);
    auto dir = std::get<0>(sample_result);
    auto is_specular = std::get<1>(sample_result);
    if (length(dir) <= 0.f) {
        // black bsdf?
        return std::tuple(Intersection{}, TSurfacePoint<T>{}, false);
    }
    auto ray = make_ray(shading_point.position, dir);
    std::vector<IntersectParameter<T>> *isect_params_ = isect_params;
    if (!is_specular) {
        // XXX HACK: stop propagating gradient if sampled a diffuse BRDF
        // Diffuse indirect illumination causes a huge amount of noise to autodiff gradient
        isect_params_ = nullptr;
    }
    auto isect_result = intersect(scene, ray, isect_params_);
    if (!is_specular) {
        // XXX same as above
        std::get<1>(isect_result) =
            convert<T>(convert<float>(std::get<1>(isect_result)));
    }
    return std::make_tuple(
            std::get<0>(isect_result), std::get<1>(isect_result), is_specular);
}

struct SecondarySample {
    LightSample light_sample;
    BSDFSample bsdf_sample;
    float rr_sample;
};

template <typename T>
struct SecondaryPoint {
    int bounce;
    Vector3f wi;
    Intersection intersection;
    TSurfacePoint<T> surface_point;
    Vector3f throughput;
};

/**
 *  Path trace from some surface point
 */
template <typename T>
auto shade(const Scene &scene,
           const TVector3<T> &wi,
           const Intersection &primary_isect,
           const TSurfacePoint<T> &shading_point,
           const LightSample &light_sample,
           const std::vector<SecondarySample> &secondary_samples,
           bool include_primary_light = true,
           std::vector<SecondaryPoint<T>> *secondary_points = nullptr,
           std::vector<IntersectParameter<T>> *isect_params = nullptr,
           std::vector<MaterialParameter<T>> *material_params = nullptr,
           std::vector<LightParameter<T>> *light_params = nullptr) {
    auto color = make_vector3(T(0.f), T(0.f), T(0.f));
    if (!primary_isect.valid()) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }
    if (include_primary_light && primary_isect.shape->light != nullptr) {
        if (dot(shading_point.shading_frame[2], wi) > 0.f) {
            // Don't optimize direct visible light source
            // color += get_intensity<T>(*primary_isect.shape->light);
            color += get_intensity(*primary_isect.shape->light, light_params);
        }
    }

    //std::vector<MaterialParameter<T>> *nm = nullptr;
    auto throughput = make_vector3(T(1.f), T(1.f), T(1.f));
    auto current_wi = wi;
    auto current_isect = primary_isect;
    auto current_shading_point = shading_point;
    auto hit_diffuse = false;
    for (int bounce = 0; bounce < (int)secondary_samples.size(); bounce++) {
        color += throughput * next_event_estimation(
            scene, light_sample, wi,
            current_isect, current_shading_point,
            isect_params, material_params, light_params);
        assert(!isnan(convert<float>(color)));
        if (secondary_points != nullptr) {
            if (!hit_diffuse) {
                secondary_points->push_back(
                    SecondaryPoint<T>{bounce,
                                      convert<float>(current_wi),
                                      current_isect,
                                      current_shading_point,
                                      convert<float>(throughput)});
            }
        }
        const auto &secondary_sample = secondary_samples[bounce];
        auto sample_bsdf_result = sample_bsdf(
            scene,
            secondary_sample.bsdf_sample,
            current_wi,
            current_isect,
            current_shading_point,
            isect_params,
            material_params);
        hit_diffuse = hit_diffuse || (!std::get<2>(sample_bsdf_result));
        auto secondary_isect = std::get<0>(sample_bsdf_result);
        if (!secondary_isect.valid()) {
            break;
        }
        auto secondary_shading_point = std::get<1>(sample_bsdf_result);
        auto wo = normalize(secondary_shading_point.position -
                            current_shading_point.position);
        if (hit_diffuse) {
            wo = convert<T>(convert<float>(wo));
        }
        auto sampled_bsdf_pdf = bsdf_pdf(
            *current_isect.shape->material,
            current_shading_point,
            current_wi,
            wo,
            material_params);
        if (sampled_bsdf_pdf <= 0.f) {
            break;
        }
        auto bsdf_eval = bsdf(*current_isect.shape->material,
                              current_shading_point,
                              current_wi,
                              wo,
                              material_params);
        if (sum(bsdf_eval) <= 0.f) {
            break;
        }
        if (secondary_isect.shape->light != nullptr) {
            if (dot(secondary_shading_point.shading_frame[2], -wo) > 0.f) {
                auto contrib = bsdf_eval * get_intensity(
                    *secondary_isect.shape->light, light_params);
                // Don't do autodiff for MIS (is this correct?)
                auto sampled_light_pdf =
                    light_pdf<float>(scene, secondary_isect) *
                    length_squared(
                      convert<float>(secondary_shading_point.position) -
                      convert<float>(current_shading_point.position)) /
                    fabs(dot(convert<float>(wo),
                             convert<float>(secondary_shading_point.geom_normal)));
                auto mis_weight = convert<float>(
                    square(sampled_bsdf_pdf) /
                    (square(sampled_light_pdf) + square(sampled_bsdf_pdf)));
                color += throughput * contrib * (mis_weight / sampled_bsdf_pdf);
                assert(!isnan(convert<float>(color)));
            }
        }
        throughput *= (bsdf_eval / sampled_bsdf_pdf);
        if (bounce >= 5) {
            auto q = min(max(convert<float>(throughput)), 0.95f);
            if (secondary_sample.rr_sample >= q) {
                break;
            }
            throughput /= q;
        }
        current_wi = -wo;
        current_isect = secondary_isect;
        current_shading_point = secondary_shading_point;
    }
    return color;
}

