#pragma once

#include "vector.h"
#include "intersect.h"
#include "distribution.h"
#include "edge.h"
#include "edge_tree.h"
#include "aabb.h"
#include "light.h"

#include <embree3/rtcore.h>

#include <vector>
#include <memory>
#include <unordered_map>

struct Camera;
struct Shape;
struct Material;

struct Scene {
    Scene(const Camera &camera,
          const std::vector<const Shape*> &shapes,
          const std::vector<const Material*> &materials,
          const std::vector<const Light*> &lights);
    virtual ~Scene();

    const Camera                       &camera;
    const std::vector<const Shape*>    &shapes;
    const std::vector<const Material*> &materials;
    const std::vector<const Light*>    &lights;

    std::vector<std::unique_ptr<Distribution1D>> light_triangle_samplers;
    std::unique_ptr<Distribution1D> light_sampler;
    std::vector<AABB> light_bounds;
    float total_light_intensity;
    std::unordered_map<const Light*, float> light_area;

    DoubleEdgeTree edge_sampler;
    // std::vector<Edge> edges;

    // Embree states
    RTCDevice rtc_device;
    RTCScene rtc_scene;
};

int sample_light_id(const Scene &scene, float light_sel);

template <typename T>
inline std::pair<Intersection, TSurfacePoint<T>> sample_point_on_light(
        const Scene &scene,
        const LightSample &light_sample,
        std::vector<IntersectParameter<T>> *isect_params) {
    // Pick a light
    assert(scene.light_sampler.get() != nullptr);
    auto light_id =
        scene.light_sampler->sample_discrete(light_sample.light_sel);
    const Light *light = scene.lights[light_id];
    // Pick a triangle
    auto tri_id = 
        scene.light_triangle_samplers[light_id]->sample_discrete(
            light_sample.tri_sel);
    const Shape &shape = *light->shape;
    Intersection light_isect{&shape, tri_id};
    // Pick a point on the triangle
    auto ind = get_indices(shape, tri_id);
    auto v0 = convert<T>(get_vertex(shape, std::get<0>(ind)));
    auto v1 = convert<T>(get_vertex(shape, std::get<1>(ind)));
    auto v2 = convert<T>(get_vertex(shape, std::get<2>(ind)));
    if (isect_params != nullptr) {
        isect_params->push_back(IntersectParameter<T>{&shape, tri_id, v0, v1, v2});
    }
    auto sample = sample_triangle(v0, v1, v2, light_sample.light_uv);
    return std::make_pair(light_isect, sample);
}

template <typename T>
inline auto light_pdf(const Scene &scene,
                      const Intersection &isect,
                      std::vector<IntersectParameter<T>> *isect_params = nullptr) {
    if (isect.shape == nullptr) {
        return T(0);
    }
    const Light *light = isect.shape->light;
    if (light == nullptr) {
        return T(0);
    }

    // TODO: derivative of light_pmf is not zero?
    auto light_pmf = luminance(get_intensity<T>(*light)) /
        scene.total_light_intensity;
    const Shape &shape = *isect.shape;
    auto tri_id = isect.tri_id;
    auto ind = get_indices(shape, tri_id);
    auto v0 = convert<T>(get_vertex(shape, std::get<0>(ind)));
    auto v1 = convert<T>(get_vertex(shape, std::get<1>(ind)));
    auto v2 = convert<T>(get_vertex(shape, std::get<2>(ind)));
    if (isect_params != nullptr) {
        isect_params->push_back(IntersectParameter<T>{&shape, tri_id, v0, v1, v2});
    }
    auto area = 0.5f * length(cross(v1 - v0, v2 - v0));
    // In theory we also want to differentiate tri_pdf
    // but v0, v1, v2 also affect the total area of tri_pdf
    // as an hack we ignore the derivatives of tri_pdf
    // this should still be an unbiased estimator of the gradient?
    auto tri_pmf = convert<float>(area) / scene.light_area.find(light)->second;
    return light_pmf * tri_pmf / area;
}

bool occluded(const Scene &scene,
              const Ray &ray,
              float max_t);

bool occluded(const Scene &scene,
              const Vector3f &p0,
              const Vector3f &p1);

Intersection nearest_hit(const Scene &scene,
                         const Ray &ray);

template <typename T>
inline auto intersect(const Scene &scene,
                      const TRay<T> &ray,
                      std::vector<IntersectParameter<T>> *params = nullptr) {
    auto isect = nearest_hit(scene, convert<float>(ray));
    if (!isect.valid()) {
        return std::make_pair(
            isect,
            make_surface_point<T>());
    }
    // Redo primary ray intersection (for autodiff)
    auto hit_point = intersect(isect, ray, params);
    return std::make_pair(isect, hit_point);
}
