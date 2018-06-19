#ifndef _BRDF_BLINN_PHONG_
#define _BRDF_BLINN_PHONG_

#include "brdf.h"

class BrdfBlinnPhong : public Brdf
{
public:

	virtual float eval(const vec3& V, const vec3& L, const float alpha, float& pdf) const
	{
		if(V.z <= 0)
		{
			pdf = 0;
			return 0;
		}

		const vec3 H = normalize(V+L);

        auto project_roughness = [&](const auto &v) {
            auto cos_theta = v.z;
            auto sin_theta_sq = 1.f - cos_theta * cos_theta;
            if (sin_theta_sq <= 0.f) {
                return alpha;
            }
            auto inv_sin_theta_sq = 1.f / sin_theta_sq;
            auto cos_phi_2 = v.x * v.x * inv_sin_theta_sq;
            auto sin_phi_2 = v.y * v.y * inv_sin_theta_sq;
            return sqrt(cos_phi_2 + sin_phi_2) * alpha;
        };
        auto smithG1 = [&](const auto &v) {
            auto cos_theta = v.z;
            if (dot(v, H) * cos_theta <= 0) {
                return 0.f;
            }
            // tan^2 + 1 = 1/cos^2
            auto tan_theta = sqrt(1.f / (cos_theta * cos_theta) - 1.f);
            if (tan_theta == 0.0f) {
                return 1.f;
            }
            auto root = project_roughness(v) * tan_theta;
            return 2.0f / (1.0f + hypot2(1.0f, root));
        };
        auto G = smithG1(V) * smithG1(L);

		// D
        auto phong_exponent = std::max(2.f / (alpha * alpha), 0.f);
        auto D = pow(H.z, phong_exponent) *
            (phong_exponent + 2.f) / float(2 * M_PI);

		pdf = fabsf(D * H.z / 4.0f / dot(V,H));
		float res = D * G / 4.0f / V.z;

		return res;
	}

	virtual vec3 sample(const vec3& V, const float alpha, const float U1, const float U2) const
	{
		const float phi = 2.0f*3.14159f * U1;
		const float phong_exponent = std::max(2.f / (alpha * alpha), 0.f);
		const float cos_theta = pow(U2, 1.f / (phong_exponent + 2.f));
		const float r = sqrt(1.f / (cos_theta * cos_theta) - 1.f);
		const vec3 N = normalize(vec3(r*cosf(phi), r*sinf(phi), 1.0f));
		const vec3 L = -V + 2.0f * N * dot(N, V);
		return L;
	}

};

#endif
