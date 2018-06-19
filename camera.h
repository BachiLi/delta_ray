#pragma once

#include "vector.h"
#include "transform.h"
#include "intersect.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <optional>

struct Camera {
    // 4x4 matrices
    pybind11::array_t<float> cam_to_world;
    pybind11::array_t<float> world_to_cam;
    pybind11::array_t<float> sample_to_cam;
    pybind11::array_t<float> cam_to_sample;
    float fov_factor; // tan(fov / 2)
    float aspect_ratio;
    float clip_near;
    bool fisheye;
};

struct DCamera {
    // 4x4 matrices
    pybind11::array_t<float> d_cam_to_world;
    pybind11::array_t<float> d_world_to_cam;
    pybind11::array_t<float> d_sample_to_cam;
    pybind11::array_t<float> d_cam_to_sample;
};

template <typename T>
struct TCamera {
    TCamera(const Camera &cam) :
        cam_to_world(cam.cam_to_world), world_to_cam(cam.world_to_cam),
        sample_to_cam(cam.sample_to_cam), cam_to_sample(cam.cam_to_sample),
        fov_factor(cam.fov_factor), aspect_ratio(cam.aspect_ratio),
        fisheye(cam.fisheye) {}
    TMatrix4x4<T> cam_to_world;
    TMatrix4x4<T> world_to_cam;
    TMatrix4x4<T> sample_to_cam;
    TMatrix4x4<T> cam_to_sample;
    T fov_factor;
    T aspect_ratio;
    bool fisheye;
};

template <typename Cam, typename T>
inline auto sample_primary(const Cam &camera,
                           const TVector2<T> &screen_pos) {
    if (camera.fisheye) {
        // Equi-angular projection
        auto org = xfm_point(camera.cam_to_world, make_vector3(T(0), T(0), T(0)));
        // x, y to polar coordinate
        auto x = 2.f * (screen_pos[0] - 0.5f);
        auto y = 2.f * (screen_pos[1] - 0.5f);
        if (x * x + y * y > 1.f) {
            return make_ray(org, make_vector3(T(0), T(0), T(0)));
        }
        auto r = sqrt(x*x + y*y);
        auto phi = atan2(y, x);
        // polar coordinate to spherical, map r to angle through polynomial
        auto theta = r * float(M_PI) / 2.f;
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        auto sin_theta = sin(theta);
        auto cos_theta = cos(theta);
        auto dir = make_vector3(-cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta);
        auto n_dir = normalize(dir);
        auto world_dir = xfm_vector(camera.cam_to_world, n_dir);
        return make_ray(org, world_dir);
    } else {
        // Linear projection
        auto org = xfm_point(camera.cam_to_world, make_vector3(T(0), T(0), T(0)));
        // [0, 1] x [0, 1] -> [1, -1] -> [1, 1-]/aspect_ratio
        auto ndc = make_vector2((screen_pos[0] - 0.5f) * -2.f,
                                (screen_pos[1] - 0.5f) * -2.f / camera.aspect_ratio);
        // Assume film at z=1, thus w=tan(fov), h=tan(fov) / aspect_ratio
        auto dir = make_vector3(camera.fov_factor * ndc[0], camera.fov_factor * ndc[1], T(1));
        auto n_dir = normalize(dir);
        auto world_dir = xfm_vector(camera.cam_to_world, n_dir);
        return make_ray(org, world_dir);
    }
}

template <typename T>
inline auto unproject(const Camera &camera,
                      const TVector2<T> &screen_pos) {
    if (camera.fisheye) {
        // x, y to polar coordinate
        auto x = 2.f * (screen_pos[0] - 0.5f);
        auto y = 2.f * (screen_pos[1] - 0.5f);
        auto r = sqrt(x*x + y*y);
        auto phi = atan2(y, x);
        // polar coordinate to spherical, map r linearly on angle
        auto theta = r * float(M_PI) / 2.f;
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        auto sin_theta = sin(theta);
        auto cos_theta = cos(theta);
        auto dir = make_vector3(-cos_phi * sin_theta, -sin_phi * sin_theta, cos_theta);
        return dir;
    } else {
        // Linear projection
        // [0, 1] x [0, 1] -> [1, -1] -> [1, 1-]/aspect_ratio
        auto ndc = make_vector2((screen_pos[0] - 0.5f) * -2.f,
                                (screen_pos[1] - 0.5f) * -2.f / camera.aspect_ratio);
        // Assume film at z=1, thus w=tan(fov), h=tan(fov) / aspect_ratio
        auto dir = make_vector3(camera.fov_factor * ndc[0], camera.fov_factor * ndc[1], T(1));
        return dir;
    }
}

template <typename T>
inline auto d_unproject(const Camera &camera,
                        const TVector2<T> &screen_pos) {
    if (camera.fisheye) {
        // x, y to polar coordinate
        auto x = 2.f * (screen_pos[0] - 0.5f);
        auto y = 2.f * (screen_pos[1] - 0.5f);
        auto r = sqrt(x*x + y*y);
        auto phi = atan2(y, x);
        // polar coordinate to spherical, map r linearly on angle
        auto theta = r * float(M_PI) / 2.f;
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        auto sin_theta = sin(theta);
        auto cos_theta = cos(theta);
 
        // d dir d screen_pos:
        auto d_dir_x_d_phi = sin_phi * sin_theta;
        auto d_dir_x_d_theta = -cos_phi * cos_theta;
        auto d_dir_y_d_phi = -cos_phi * sin_theta;
        auto d_dir_y_d_theta = -sin_phi * cos_theta;
        auto d_dir_z_d_theta = -sin_theta;
        auto d_phi_d_x = -y / (r*r);
        auto d_phi_d_y = x / (r*r);
        auto d_theta_d_x = (float(M_PI) / 2.f) * x / r;
        auto d_theta_d_y = (float(M_PI) / 2.f) * y / r;

        return std::make_pair(
            2.f * make_vector3(d_dir_x_d_phi * d_phi_d_x + d_dir_x_d_theta * d_theta_d_x,
                               d_dir_y_d_phi * d_phi_d_x + d_dir_y_d_theta * d_theta_d_x,
                               d_dir_z_d_theta * d_theta_d_x),
            2.f * make_vector3(d_dir_x_d_phi * d_phi_d_y + d_dir_x_d_theta * d_theta_d_y,
                               d_dir_y_d_phi * d_phi_d_y + d_dir_y_d_theta * d_theta_d_y,
                               d_dir_z_d_theta * d_theta_d_y));
    } else {
        return std::make_pair(
            make_vector3(-2.f * camera.fov_factor, T(0), T(0)),
            make_vector3(T(0), -2.f * camera.fov_factor / camera.aspect_ratio, T(0)));
    }
}

template <typename T>
auto project_local(const Camera &camera,
                   const TVector3<T> &local) {
    if (camera.fisheye) {
        // Equi-angular projection
        auto dir = normalize(local);
        auto cos_theta = dir[2];
        auto phi = atan2(dir[1], dir[0]);
        auto theta = acos(cos_theta);
        auto r = theta * 2.f / float(M_PI);
        auto x = 0.5f * (-r * cos(phi) + 1.f);
        auto y = 0.5f * (-r * sin(phi) + 1.f);
        return make_vector2(x, y);
    } else {
        // Linear projection
        auto x = (-local[0] / (local[2] * camera.fov_factor) + 1.f) * 0.5f;
        auto y = (-local[1] / (local[2] * camera.fov_factor * camera.aspect_ratio) + 1.f) * 0.5f;
        // auto projected = xfm_point(camera.cam_to_sample, local);
        return make_vector2(x, y);
    }
}

inline auto d_project_local(const Camera &camera,
                            const Vector3f &local,
                            float dx, float dy) {
    // Backprop dx, dy to local
    if (camera.fisheye) {
        auto dir = normalize(local);
        auto phi = atan2(dir[1], dir[0]);
        auto theta = acos(dir[2]);
        auto r = theta * 2.f / float(M_PI);
        // Backprop x = 0.5f * (-r * cos(phi) + 1.f);
        //          y = 0.5f * (-r * sin(phi) + 1.f);
        auto dr = -0.5f * (cos(phi) * dx + sin(phi) * dy);
        auto dphi = 0.5f * r * sin(phi) * dx -
                    0.5f * r * cos(phi) * dy;
        // Backprop r = theta * 2.f / float(M_PI);
        auto dtheta = dr * (2.f / float(M_PI));
        // Backprop theta = acos(cos_theta);
        auto d_cos_theta = -dtheta / sqrt(1.f - dir[2]*dir[2]);
        // Backprop phi = atan2(dir[1], dir[0]);
        auto atan2_tmp = dir[0] * dir[0] + dir[1] * dir[1];
        auto ddir0 = -dphi * dir[1] / atan2_tmp;
        auto ddir1 =  dphi * dir[0] / atan2_tmp;
        // Backprop cos_theta = dir[2];
        auto ddir2 = d_cos_theta;
        // Backprop dir = normalize(local);
        auto ddir = make_vector3(ddir0, ddir1, ddir2);
        return d_normalize(local, ddir);
    } else {
        auto zfov = local[2] * camera.fov_factor;
        auto dxdlocal0 = -1.f / zfov;
        auto dxdlocal2 = camera.fov_factor * local[0] / (zfov * zfov);
        auto zfovy = zfov * camera.aspect_ratio;
        auto dydlocal1 = -1.f / zfovy;
        auto dydlocal2 = camera.fov_factor * camera.aspect_ratio * local[1] / (zfovy * zfovy);
        return make_vector3(
            dx * dxdlocal0, dy * dydlocal1, dx * dxdlocal2 + dy * dydlocal2);
    }
}

template <typename T>
std::optional<std::pair<TVector2<T>, TVector2<T>>> 
    project(const Camera &camera,
            const TVector3<T> p0,
            const TVector3<T> p1) {
    auto p0_local = xfm_point(camera.world_to_cam, p0);
    auto p1_local = xfm_point(camera.world_to_cam, p1);
    if (p0_local[2] < camera.clip_near &&
           p1_local[2] < camera.clip_near) {
        return {};
    }
    // clip against z = clip_near
    if (p0_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p0_local - p1_local;
        auto t = -(p1_local[2] - camera.clip_near) / dir[2];
        p0_local = p1_local + t * dir;
    } else if (p1_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p1_local - p0_local;
        auto t = -(p0_local[2] - camera.clip_near) / dir[2];
        p1_local = p0_local + t * dir;
    }
    // project to 2d screen
    return std::make_pair(project_local(camera, p0_local),
                          project_local(camera, p1_local));
}

inline void d_project(const Camera &camera,
                      const Vector3f &p0,
                      const Vector3f &p1,
                      float dp0x, float dp0y,
                      float dp1x, float dp1y,
                      Matrix4x4f &d_cam_to_sample,
                      Matrix4x4f &d_world_to_cam) {
    auto p0_local = xfm_point(camera.world_to_cam, p0);
    auto p1_local = xfm_point(camera.world_to_cam, p1);
    auto clipped_p0_local = p0_local;
    auto clipped_p1_local = p1_local;
    // clip against z = clip_near
    if (p0_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p0_local - p1_local;
        auto t = -(p1_local[2] + camera.clip_near) / dir[2];
        clipped_p0_local = p1_local + t * dir;
    } else if (p1_local[2] < camera.clip_near) {
        // a ray from p1 to p0
        auto dir = p1_local - p0_local;
        auto t = -(p0_local[2] + camera.clip_near) / dir[2];
        clipped_p1_local = p0_local + t * dir;
    }
    auto dclipped_p0_local =
        d_project_local(camera, clipped_p0_local, dp0x, dp0y);
    auto dclipped_p1_local =
        d_project_local(camera, clipped_p1_local, dp1x, dp1y);
    // backprop buffer
    auto dp0_local = make_vector3(0.f, 0.f, 0.f);
    auto dp1_local = make_vector3(0.f, 0.f, 0.f);
    // differentiate through clipping
    if (p0_local[2] < camera.clip_near) {
        // forward variables
        auto dir = p0_local - p1_local;
        auto t = -(p1_local[2] + camera.clip_near) / dir[2];
        // backprop clipped_p0_local = p1_local + t * dir;
        dp1_local += dclipped_p0_local;
        auto dt = dot(dir, dclipped_p0_local);
        auto ddir = t * dclipped_p0_local;
        // backprop t = -p1_local[2] / dir[2];
        dp1_local[2] += (-dt / dir[2]);
        ddir[2] += dt * (p1_local[2] / (dir[2] * dir[2]));
        // backprop dir = p0_local - p1_local;
        dp0_local += ddir;
        dp1_local -= ddir;
        dp1_local += dclipped_p1_local;
    } else if (p1_local[2] < camera.clip_near) {
        // forward variables
        auto dir = p1_local - p0_local;
        auto t = -(p0_local[2] + camera.clip_near) / dir[2];
        // backprop clipped_p1_local = p0_local + t * dir;
        dp0_local += dclipped_p1_local;
        auto dt = dot(dir, dclipped_p1_local);
        auto ddir = t * dclipped_p1_local;
        // backprop t = -p0_local[2] / dir[2];
        dp0_local[2] += (-dt / dir[2]);
        ddir[2] += dt * (p0_local[2] / (dir[2] * dir[2]));
        // backprop dir = p1_local - p0_local;
        dp1_local += ddir;
        dp0_local -= ddir;
        dp0_local += dclipped_p0_local;
    } else {
        dp0_local += dclipped_p0_local;
        dp1_local += dclipped_p1_local;
    }

    // now backprop dp0_local & dp1_local to p0 & p1
    d_xfm_point(camera.world_to_cam, p0, dp0_local, d_world_to_cam);
    d_xfm_point(camera.world_to_cam, p1, dp1_local, d_world_to_cam);
}

inline auto in_screen(const Camera &cam, const Vector2f &pt) {
    if (!cam.fisheye) {
        return pt[0] >= 0.f && pt[0] < 1.f &&
               pt[1] >= 0.f && pt[1] < 1.f;
    } else {
        auto dist_sq =
            (pt[0] - 0.5f) * (pt[0] - 0.5f) + (pt[1] - 0.5f) * (pt[1] - 0.5f);
        return dist_sq < 0.25f;
    }
}
