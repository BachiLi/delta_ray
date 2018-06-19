#pragma once

#include "ltc.inc"
#include "vector.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <random>
#include <memory>

namespace py = pybind11;

struct Material {
    py::array_t<float> diffuse_reflectance;
    py::array_t<float> specular_reflectance;
    py::array_t<float> roughness;
    py::array_t<float> diffuse_uv_scale;
    py::array_t<float> specular_uv_scale;
    py::array_t<float> roughness_uv_scale;
    bool two_sided;
};

struct DMaterial {
    py::array_t<float> diffuse_reflectance;
    py::array_t<float> specular_reflectance;
    py::array_t<float> roughness;
    py::array_t<float> diffuse_uv_scale;
    py::array_t<float> specular_uv_scale;
    py::array_t<float> roughness_uv_scale;
};

struct BSDFSample {
    Vector2f uv;
    float w;
};

template <typename T>
struct TextureParameter {
    int xi, yi;
    T t00, t01, t10, t11;
};

template <typename T>
struct MaterialParameter {
    const Material *material = nullptr;
    TextureParameter<TVector3<T>> diffuse_reflectance;
    TextureParameter<TVector3<T>> specular_reflectance;
    TextureParameter<T> roughness;
    TVector2<T> diffuse_uv_scale;
    TVector2<T> specular_uv_scale;
    TVector2<T> roughness_uv_scale;
};

// Always-positive modulo function (assumes b > 0)
inline auto modulo(int a, int b) {
    auto r = a % b;
    return (r < 0) ? r+b : r;
}

template <typename T>
inline auto get_diffuse_uv_scale(const Material &material) {
    auto accessor = material.diffuse_uv_scale.unchecked<1>();
    return TVector2<T>{accessor(0), accessor(1)};
}

template <typename T>
inline auto get_specular_uv_scale(const Material &material) {
    auto accessor = material.specular_uv_scale.unchecked<1>();
    return TVector2<T>{accessor(0), accessor(1)};
}

template <typename T>
inline auto get_roughness_uv_scale(const Material &material) {
    auto accessor = material.roughness_uv_scale.unchecked<1>();
    return TVector2<T>{accessor(0), accessor(1)};
}

inline auto has_texture(const py::array_t<float> &value) {
    return value.ndim() == 3;
}

// TODO: backprop to textures
template <typename T>
inline auto get_constant_value_1d(const py::array_t<float> &value,
                                  TextureParameter<T> *param = nullptr) {
    auto accessor = value.unchecked<1>();
    auto val = convert<T>(accessor(0));
    if (param != nullptr) {
        param->xi = param->yi = -1;
        param->t00 = val;
    }
    return val;
}

template <typename T>
inline auto get_constant_value_3d(const py::array_t<float> &value,
                                  TextureParameter<TVector3<T>> *param = nullptr) {
    auto accessor = value.unchecked<1>();
    auto val = make_vector3(T(accessor(0)), T(accessor(1)), T(accessor(2)));
    if (param != nullptr) {
        param->xi = param->yi = -1;
        param->t00 = val;
    }
    return val;
}

template <typename T>
inline auto get_value_1d(const py::array_t<float> &value,
                         const TVector2<T> &uv,
                         TextureParameter<T> *param = nullptr) {
    if (!has_texture(value)) {
        return get_constant_value_1d<T>(value, param);
    }
    assert(value.shape()[2] == 1);
    auto width = int(value.shape()[1]);
    auto height = int(value.shape()[0]);
    auto x_max = width - 1;
    auto y_max = height - 1;
    assert(x_max >= 0 && y_max >= 0);
    auto xf = (int)floor(convert<float>(uv[0]) * width);
    auto yf = (int)floor(convert<float>(uv[1]) * height);
    auto xc = xf + 1;
    auto yc = yf + 1;
    auto u = (uv[0] * width) - xf;
    auto v = (uv[1] * height) - yf;
    auto xfi = modulo(xf, width);
    auto yfi = modulo(yf, height);
    auto xci = modulo(xc, width);
    auto yci = modulo(yc, height);
    auto texture_accessor = value.unchecked<3>();
    auto value_ff = convert<T>(texture_accessor(yfi, xfi, 0));
    auto value_fc = convert<T>(texture_accessor(yci, xfi, 0));
    auto value_cf = convert<T>(texture_accessor(yfi, xci, 0));
    auto value_cc = convert<T>(texture_accessor(yci, xci, 0));
    if (param != nullptr) {
        param->xi = xfi;
        param->yi = yfi;
        param->t00 = value_ff;
        param->t01 = value_fc;
        param->t10 = value_cf;
        param->t11 = value_cc;
    }
    return value_ff * (1.f - u) * (1.f - v) +
           value_fc * (1.f - u) *        v  +
           value_cf *        u  * (1.f - v) +
           value_cc *        u  *        v;
}

template <typename T>
inline auto get_value_3d(const py::array_t<float> &value,
                         const TVector2<T> &uv,
                         TextureParameter<TVector3<T>> *param = nullptr) {
    if (!has_texture(value)) {
        return convert<T>(get_constant_value_3d<T>(value, param));
    }
    assert(value.shape()[2] == 3);
    auto width = int(value.shape()[1]);
    auto height = int(value.shape()[0]);
    auto x_max = width - 1;
    auto y_max = height - 1;
    assert(x_max >= 0 && y_max >= 0);
    auto xf = (int)floor(convert<float>(uv[0]) * width);
    auto yf = (int)floor(convert<float>(uv[1]) * height);
    auto xc = xf + 1;
    auto yc = yf + 1;
    auto u = (uv[0] * width) - xf;
    auto v = (uv[1] * height) - yf;
    auto xfi = modulo(xf, width);
    auto yfi = modulo(yf, height);
    auto xci = modulo(xc, width);
    auto yci = modulo(yc, height);
    auto texture_accessor = value.unchecked<3>();
    auto color_ff =
        make_vector3(T(texture_accessor(yfi, xfi, 0)),
                     T(texture_accessor(yfi, xfi, 1)),
                     T(texture_accessor(yfi, xfi, 2)));
    auto color_fc =
        make_vector3(T(texture_accessor(yci, xfi, 0)),
                     T(texture_accessor(yci, xfi, 1)),
                     T(texture_accessor(yci, xfi, 2)));
    auto color_cf =
        make_vector3(T(texture_accessor(yfi, xci, 0)),
                     T(texture_accessor(yfi, xci, 1)),
                     T(texture_accessor(yfi, xci, 2)));
    auto color_cc =
        make_vector3(T(texture_accessor(yci, xci, 0)),
                     T(texture_accessor(yci, xci, 1)),
                     T(texture_accessor(yci, xci, 2)));
    auto color = color_ff * (1.f - u) * (1.f - v) +
                 color_fc * (1.f - u) *        v  +
                 color_cf *        u  * (1.f - v) +
                 color_cc *        u  *        v;
    if (param != nullptr) {
        param->xi = xfi;
        param->yi = yfi;
        param->t00 = color_ff;
        param->t01 = color_fc;
        param->t10 = color_cf;
        param->t11 = color_cc;
    }
    return color;
}

template <typename T>
inline auto get_diffuse_reflectance(const Material &material,
                                    const TVector2<T> &uv,
                                    MaterialParameter<T> *param = nullptr) {
    auto uv_scale = get_diffuse_uv_scale<T>(material);
    if (param != nullptr) {
        param->diffuse_uv_scale = uv_scale;
    }
    return get_value_3d(material.diffuse_reflectance, uv * uv_scale,
        param != nullptr ? &param->diffuse_reflectance : nullptr);
}

template <typename T>
inline auto get_specular_reflectance(const Material &material,
                                     const TVector2<T> &uv,
                                     MaterialParameter<T> *param = nullptr) {
    auto uv_scale = get_specular_uv_scale<T>(material);
    if (param != nullptr) {
        param->specular_uv_scale = uv_scale;
    }
    return get_value_3d(material.specular_reflectance, uv * uv_scale,
        param != nullptr ? &param->specular_reflectance : nullptr);
}

template <typename T>
inline auto get_roughness(const Material &material,
                          const TVector2<T> &uv,
                          MaterialParameter<T> *param = nullptr) {
    auto uv_scale = get_roughness_uv_scale<T>(material);
    if (param != nullptr) {
        param->roughness_uv_scale = uv_scale;
    }
    return get_value_1d(material.roughness, uv * uv_scale,
        param != nullptr ? &param->roughness : nullptr);
}

template <typename T>
inline auto roughness_to_phong(const T &roughness) {
    return max(2.f / roughness, T(0));
}

inline auto make_bsdf_sample(std::mt19937 &rng) {
    std::uniform_real_distribution<float> uni_dist(0.f, 1.f);
    return BSDFSample{make_vector2(uni_dist(rng), uni_dist(rng)),
                      uni_dist(rng)};
}

template <typename T>
auto schlick_fresnel(const T &cos_theta) {
    return pow(1 - cos_theta, 5.f);
}

template <typename T>
auto bsdf(const Material &material,
          const TSurfacePoint<T> &surface_point,
          const TVector3<T> &wi,
          const TVector3<T> &wo,
          std::vector<MaterialParameter<T>> *params = nullptr) {
    auto n = surface_point.shading_frame[2];
    if (material.two_sided) {
        if (dot(wi, n) < 0.f) {
            n = -n;
        }
    }
    //auto cos_wi = dot(n, wi);
    auto bsdf_cos = dot(n, wo);
    if (bsdf_cos <= 5e-3f) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }
    if (params != nullptr) {
        params->push_back(MaterialParameter<T>{});
        params->back().material = &material;
    }
    auto diffuse_reflectance =
        get_diffuse_reflectance(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    auto specular_reflectance =
        get_specular_reflectance(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    // TODO: anisotropic material
    // roughness = alpha^2 in ggx
    auto roughness =
        get_roughness(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    assert(convert<float>(roughness) > 0.f);
    // In theory we should multiply (1-\int specular) to account for
    // energy loss at specular surface but who cares.
    auto diffuse_contrib = diffuse_reflectance * bsdf_cos / float(M_PI);
    assert(!isnan(convert<float>(diffuse_contrib)));
    auto specular_contrib = make_vector3(T(0.f), T(0.f), T(0.f));
    if (sum(specular_reflectance) > 0.f) {
        // blinn-phong
        if (dot(wi, n) > 0.f) {
            // half-vector
            auto m = normalize(wi + wo);
            auto m_local = to_local(surface_point.shading_frame, m);
            if (m_local[2] > 0.f) {
                auto phong_exponent = roughness_to_phong(roughness);
                auto D = pow(max(m_local[2], 0.f), phong_exponent) *
                    (phong_exponent + 2.f) / float(2 * M_PI);
                auto project_roughness = [&](const auto &v) {
                    auto v_local = to_local(surface_point.shading_frame, v);
                    auto cos_theta = v_local[2];
                    auto sin_theta_sq = 1.f - cos_theta * cos_theta;
                    if (sin_theta_sq <= 0.f) {
                        return sqrt(roughness);
                    }
                    auto inv_sin_theta_sq = 1.f / sin_theta_sq;
                    auto cos_phi_2 = v_local[0] * v_local[0] * inv_sin_theta_sq;
                    auto sin_phi_2 = v_local[1] * v_local[1] * inv_sin_theta_sq;
                    return sqrt(cos_phi_2 * roughness + sin_phi_2 * roughness);
                };
                auto smithG1 = [&](const auto &v) {
                    auto cos_theta = dot(v, n);
                    if (dot(v, m) * cos_theta <= 0) {
                        return T(0);
                    }
                    // tan^2 + 1 = 1/cos^2
                    auto tan_theta = sqrt(max(1.f / (cos_theta * cos_theta) - 1.f, 0.f));
                    if (tan_theta == 0.0f) {
                        return T(1);
                    }
                    auto alpha = project_roughness(v);
                    auto a = 1.f / (alpha * tan_theta);
                    if (a >= 1.6f) {
                        return T(1);
                    }
                    auto a_sqr = a*a;
                    return (3.535f * a + 2.181f * a_sqr)
                         / (1.0f + 2.276f * a + 2.577f * a_sqr);
                };
                auto G = smithG1(wi) * smithG1(wo);
                auto cos_theta_d = dot(m, wo);
                // Schlick's approximation
                auto F = specular_reflectance +
                    (1.f - specular_reflectance) *
                    pow(max(1.f - cos_theta_d, 0.f), 5.f);
                specular_contrib = F * D * G / (4.f * dot(wi, n));
                assert(!isnan(convert<float>(G)));
                assert(!isnan(convert<float>(D)));
                assert(!isnan(convert<float>(F)));
                assert(!isnan(convert<float>(specular_contrib)));
            }
        }
    }
    return diffuse_contrib + specular_contrib;
}

template <typename T>
auto sample_bsdf(const Material &material,
                 const TSurfacePoint<T> &surface_point,
                 const TVector3<T> &wi,
                 const BSDFSample &bsdf_sample,
                 std::vector<MaterialParameter<T>> *params = nullptr) {
    if (params != nullptr) {
        params->push_back(MaterialParameter<T>{});
        params->back().material = &material;
    }
    auto diffuse_reflectance =
        get_diffuse_reflectance(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    auto specular_reflectance =
        get_specular_reflectance(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        return std::make_pair(make_vector3(T(0.f), T(0.f), T(0.f)), false);
    }
    /*auto n = surface_point.shading_frame[2];
    auto cos_wi = dot(convert<float>(n), convert<float>(wi));
    if (cos_wi <= 0.f) {
        return make_vector3(T(0.f), T(0.f), T(0.f));
    }*/
    auto diffuse_pmf = diffuse_weight / weight_sum;
    // auto specular_pmf = specular_weight / weight_sum;
    if (bsdf_sample.w <= diffuse_pmf) {
        // Lambertian
        if (diffuse_pmf <= 0.f) {
            return std::make_pair(make_vector3(T(0.f), T(0.f), T(0.f)), false);
        }
        auto local_dir = cos_hemisphere(bsdf_sample.uv);
        return std::make_pair(to_world(surface_point.shading_frame, local_dir), false);
    } else {
        if (specular_weight <= 0.f) {
            return std::make_pair(make_vector3(T(0.f), T(0.f), T(0.f)), false);
        }
        // Blinn-phong
        auto roughness = get_roughness(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
        auto phong_exponent = roughness_to_phong(roughness);
        // Sample phi
        auto phi = 2.f * float(M_PI) * bsdf_sample.uv[1];
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        
        // Sample theta
        auto cos_theta = pow(bsdf_sample.uv[0], 1.0f / (phong_exponent + 2.0f));
        auto sin_theta = sqrt(max(1.f - cos_theta*cos_theta, 0.f));
        // local microfacet normal
        auto m_local = make_vector3(
            sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        auto m = to_world(surface_point.shading_frame, m_local);
        auto dir = 2.f * dot(wi, m) * m - wi;
        return std::make_pair(dir, roughness <= 0.01f);
    }
}

template <typename T>
auto bsdf_pdf(const Material &material,
              const TSurfacePoint<T> &surface_point,
              const TVector3<T> &wi,
              const TVector3<T> &wo,
              std::vector<MaterialParameter<T>> *params = nullptr) {
    if (params != nullptr) {
        params->push_back(MaterialParameter<T>{});
        params->back().material = &material;
    }
    auto diffuse_reflectance =
        get_diffuse_reflectance(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    auto specular_reflectance =
        get_specular_reflectance(material, surface_point.uv,
            params != nullptr ? &params->back() : nullptr);
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        return T(0);
    }
    auto diffuse_pmf = diffuse_weight / weight_sum;
    auto specular_pmf = specular_weight / weight_sum;
    auto diffuse_pdf = T(0);
    if (diffuse_pmf > 0.f) {
        auto bsdf_cos = dot(surface_point.shading_frame[2], wo);
        if (!material.two_sided) {
            bsdf_cos = max(bsdf_cos, 0.f);
        } else {
            bsdf_cos = fabs(bsdf_cos);
        }
        diffuse_pdf = diffuse_pmf * bsdf_cos / float(M_PI);
    }
    auto specular_pdf = T(0);
    if (specular_pmf > 0.f) {
        auto m = normalize(wi + wo);
        auto m_local = to_local(surface_point.shading_frame, m);
        if (m_local[2] > 0.f) {
            auto roughness = get_roughness(material, surface_point.uv,
                params != nullptr ? &params->back() : nullptr);
            auto phong_exponent = roughness_to_phong(roughness);
            auto D = pow(m_local[2], phong_exponent) *
                (phong_exponent + 2.f) / float(2 * M_PI);
            specular_pdf = specular_pmf * D / (4.f * fabs(dot(wo, m)));
        }
    }
    return diffuse_pdf + specular_pdf;
}

inline auto get_ltc_matrix(const Material &material,
                           const SurfacePoint &surface_point,
                           const Vector3f &wi) {
    auto roughness = get_roughness(material, surface_point.uv);
    auto cos_theta = dot(wi, surface_point.shading_frame[2]);
    auto theta = acos(cos_theta);
    // search lookup table
    auto rid = int(roughness * (ltc::size - 1));
    auto tid = int((theta / (M_PI / 2.f)) * (ltc::size - 1));
    assert(rid >= 0 && rid < ltc::size && tid >= 0 && tid < ltc::size);
    // TODO: linear interpolation?
    return make_matrix3x3(ltc::tabM[rid+tid*ltc::size]);
}

inline auto get_ltc_matrix(const Material &material,
                           const SurfacePoint &surface_point,
                           float cos_theta) {
    auto roughness = get_roughness(material, surface_point.uv);
    auto theta = acos(cos_theta);
    // search lookup table
    auto rid = int(roughness * (ltc::size - 1));
    auto tid = int((theta / (M_PI / 2.f)) * (ltc::size - 1));
    assert(rid >= 0 && rid < ltc::size && tid >= 0 && tid < ltc::size);
    // TODO: linear interpolation?
    return make_matrix3x3(ltc::tabM[rid+tid*ltc::size]);
}
