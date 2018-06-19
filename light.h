#pragma once

#include "distribution.h"
#include "sample.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <random>

namespace py = pybind11;

struct Shape;

struct Light {
    const Shape *shape;
    py::array_t<float> intensity;
};

struct DLight {
    py::array_t<float> intensity;
};

struct LightSample {
    Vector2f light_uv;
    float light_sel;
    float tri_sel;
};

template <typename T>
struct LightParameter {
    const Light *light;
    TVector3<T> intensity;
};

template <typename T>
inline auto get_intensity(const Light &light,
                          std::vector<LightParameter<T>> *param = nullptr) {
    auto ret = convert<T>(make_vector3(light.intensity));
    if (param != nullptr) {
        param->push_back(LightParameter<T>{&light, ret});
    }
    return ret;
}

inline auto make_light_sample(std::mt19937 &rng) {
    std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
    return LightSample{
        make_vector2(uni_dist(rng), uni_dist(rng)),
        uni_dist(rng), uni_dist(rng)};
}
