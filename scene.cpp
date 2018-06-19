#undef NDEBUG

#include "scene.h"
#include "camera.h"
#include "shape.h"
#include "material.h"
#include "light.h"
#include "distribution.h"
#include "intersect.h"

#include <embree3/rtcore_ray.h>

#include <map>
#include <memory>

struct Vertex4f { float x,y,z,r;  };
struct Triangle { int v0, v1, v2; };

Scene::Scene(const Camera &camera,
             const std::vector<const Shape*> &shapes,
             const std::vector<const Material*> &materials,
             const std::vector<const Light*> &lights)
    : camera(camera), shapes(shapes), materials(materials), lights(lights) {
    // Preprocessing
    // Light sampler
    total_light_intensity = 0.f;
    std::vector<float> light_importance;
    for (const Light *light : lights) {
        light_importance.push_back(luminance(get_intensity<float>(*light)));
        total_light_intensity += luminance(get_intensity<float>(*light));
        const Shape &shape = *light->shape;
        std::vector<float> tri_areas;
        AABB bounds;
        float area_sum = 0.f;
        for (int tri_id = 0; tri_id < shape.indices.shape()[0]; tri_id++) {
            auto ind = get_indices(shape, tri_id);
            auto v0 = get_vertex(shape, std::get<0>(ind));
            auto v1 = get_vertex(shape, std::get<1>(ind));
            auto v2 = get_vertex(shape, std::get<2>(ind));
            auto area = 0.5f * length(cross(v1 - v0, v2 - v0));
            area_sum += area;
            tri_areas.push_back(area);
            bounds = merge(bounds, AABB(v0, v1, v2));
        }
        light_triangle_samplers.push_back(
            std::make_unique<Distribution1D>(&tri_areas[0], tri_areas.size()));
        light_area[light] = area_sum;
        light_bounds.push_back(bounds);
    }

    light_sampler = std::make_unique<Distribution1D>(
        &light_importance[0], light_importance.size());

    // Edge sampler
    std::map<std::tuple<const Shape*, int, int>, Edge> edges_map;
    for (const Shape *shape : shapes) {
        auto indices_accessor = shape->indices.unchecked<2>();
        for (int tri_id = 0; tri_id < shape->indices.shape()[0]; tri_id++) {
            int v0 = indices_accessor(tri_id, 0);
            int v1 = indices_accessor(tri_id, 1);
            int v2 = indices_accessor(tri_id, 2);
            auto add_edge = [&](int v0, int v1) {
                if (v0 > v1) {
                    std::swap(v0, v1);
                }
                auto key = std::make_tuple(shape, v0, v1);
                if (edges_map.find(key) == edges_map.end()) {
                    Edge edge{shape, v0, v1, tri_id, -1};
                    edges_map[key] = edge;
                } else {
                    assert(edges_map[key].f1 == -1);
                    edges_map[key].f1 = tri_id;
                }
            };
            add_edge(v0, v1);
            add_edge(v1, v2);
            add_edge(v2, v0);
        }
    }
    std::vector<Edge> edges;
    for (const auto &it : edges_map) {
        auto edge = it.second;
        assert(edge.f0 != -1);
        if (abs_sum(get_v0(edge) - get_v1(edge)) < 1e-10f) {
            continue;
        }
        if (edge.f1 != -1) {
            // Don't include the edge if the angle is zero or 180 degree
            auto n0 = get_normal(*edge.shape, edge.f0);
            auto n1 = get_normal(*edge.shape, edge.f1);
            auto angle = fabs(dot(n0, n1));
            if (angle == 1 || angle == -1) {
                continue;
            }
        }
        edges.push_back(it.second);
    }
    edge_sampler = build_double_edge_tree(camera, edges);

    // Embree setup
    rtc_device = rtcNewDevice(nullptr);
    rtc_scene = rtcNewScene(rtc_device);
    rtcSetSceneBuildQuality(rtc_scene, RTC_BUILD_QUALITY_HIGH);
    rtcSetSceneFlags(rtc_scene, RTC_SCENE_FLAG_ROBUST);
    auto shape_id = 0;
    for (const Shape *shape : shapes) {
        RTCGeometry mesh = rtcNewGeometry(rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        auto nv = num_vertices(*shape);
        auto nt = num_triangles(*shape);
        Vertex4f* vertices =
            (Vertex4f*) rtcSetNewGeometryBuffer(
                mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                sizeof(Vertex4f), nv);
        for (auto i = 0; i < nv; i++) {
            auto vertex = get_vertex(*shape, i);
            vertices[i] = Vertex4f{std::get<0>(vertex), std::get<1>(vertex), std::get<2>(vertex), 0.f};
        }
        Triangle* triangles =
            (Triangle*) rtcSetNewGeometryBuffer(
                mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                sizeof(Triangle), nt);
        for (auto i = 0; i < nt; i++) {
            auto indices = get_indices(*shape, i);
            triangles[i] = Triangle{std::get<0>(indices),
                                    std::get<1>(indices),
                                    std::get<2>(indices)};
        }
        rtcSetGeometryVertexAttributeCount(mesh, 1);
        rtcCommitGeometry(mesh);
        auto geom_id = rtcAttachGeometry(rtc_scene, mesh);
        assert((int)geom_id == shape_id);
        shape_id++;
        rtcReleaseGeometry(mesh);
    }
    rtcCommitScene(rtc_scene);
}

Scene::~Scene() {
    rtcReleaseScene(rtc_scene);
    rtcReleaseDevice(rtc_device);
}

int sample_light_id(const Scene &scene, float light_sel) {
    return scene.light_sampler->sample_discrete(light_sel);
}

bool occluded(const Scene &scene,
              const Ray &ray,
              float max_t) {
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRay rtc_ray;
    rtc_ray.org_x = std::get<0>(ray.org);
    rtc_ray.org_y = std::get<1>(ray.org);
    rtc_ray.org_z = std::get<2>(ray.org);
    rtc_ray.dir_x = std::get<0>(ray.dir);
    rtc_ray.dir_y = std::get<1>(ray.dir);
    rtc_ray.dir_z = std::get<2>(ray.dir);
    rtc_ray.tnear = max_t * 1e-3f;
    rtc_ray.tfar = max_t * (1.f - 1e-3f);
    rtc_ray.mask = -1;
    rtc_ray.time = 0.f;
    rtc_ray.flags = 0;
    rtcOccluded1(scene.rtc_scene, &rtc_context, &rtc_ray);
    return rtc_ray.tfar < 0.f;
}

bool occluded(const Scene &scene,
              const Vector3f &p0,
              const Vector3f &p1) {
    auto dir = p1 - p0;
    auto distance = length(dir);
    auto ray = Ray{p0, dir / distance};
    return occluded(scene, ray, distance);
}

Intersection nearest_hit(const Scene &scene,
                         const Ray &ray) {
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = std::get<0>(ray.org);
    rtc_ray_hit.ray.org_y = std::get<1>(ray.org);
    rtc_ray_hit.ray.org_z = std::get<2>(ray.org);
    rtc_ray_hit.ray.dir_x = std::get<0>(ray.dir);
    rtc_ray_hit.ray.dir_y = std::get<1>(ray.dir);
    rtc_ray_hit.ray.dir_z = std::get<2>(ray.dir);
    rtc_ray_hit.ray.tnear = 1e-3f;
    rtc_ray_hit.ray.tfar = std::numeric_limits<float>::infinity();
    rtc_ray_hit.ray.mask = -1;
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(scene.rtc_scene, &rtc_context, &rtc_ray_hit);
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return Intersection{nullptr, -1};
    }
    return Intersection{scene.shapes[rtc_ray_hit.hit.geomID],
                        (int)rtc_ray_hit.hit.primID};
}
