#pragma once

#include "vector.h"
#include <vector>

// From https://github.com/mmp/pbrt-v3/blob/master/src/core/sampling.h
struct Distribution1D {
    Distribution1D(const float *f, int n) {
        count = n;
        func = new float[n];
        memcpy(func, f, n * sizeof(float));
        cdf = new float[n + 1];
        cdf[0] = 0.;
        for (int i = 1; i < count + 1; ++i)
            cdf[i] = cdf[i - 1] + func[i - 1] / n;

        func_int = cdf[count];
        if (func_int == 0.f) {
            for (int i = 1; i < n + 1; ++i)
                cdf[i] = float(i) / float(n);
        } else {
            for (int i = 1; i < n + 1; ++i)
                cdf[i] /= func_int;
        }
    }
    ~Distribution1D() {
        delete[] func;
        delete[] cdf;
    }
    float sample_continuous(float u, float *pdf, int *off = nullptr) const {
        float *ptr = std::upper_bound(cdf, cdf + count + 1, u);
        int offset = clamp(int(ptr - cdf - 1), 0, count - 1);
        if (off) {
            *off = offset;
        }

        float du = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);

        if (pdf) {
            *pdf = func[offset] / func_int;
        }

        return (offset + du) / count;
    }
    int sample_discrete(float u, float *pdf = nullptr) const {
        float *ptr = std::upper_bound(cdf, cdf + count + 1, u);
        int offset = clamp(int(ptr - cdf - 1), 0, count - 1);
        if (pdf != nullptr) {
            *pdf = func[offset] / (func_int * count);
        }
        return offset;
    }

    float pmf(int offset) const {
        return func[offset] / (func_int * count);
    }

    float *func, *cdf;
    float func_int;
    int count;
};
