#ifndef AUTODIFF_H__
#define AUTODIFF_H__

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI std::acos(-1)
#endif

namespace autodiff {

// Change the following line if you want to use single precision floats
typedef float Real; 
typedef int VertexId;

struct ADGraph;
struct AReal;

extern thread_local ADGraph* g_ADGraph;
// Declare this in your .cpp source
#define DECLARE_ADGRAPH() namespace autodiff { thread_local ADGraph* g_ADGraph = 0; }

AReal NewAReal(const Real val);

struct AReal {
    AReal() : varId(-1) {}

    AReal(const Real val) {
        *this = NewAReal(val);
    }

    AReal(const Real val, const VertexId varId) : 
        val(val), varId(varId) {}

    Real val;
    VertexId varId;
};

struct ADEdge {
    ADEdge() {}
    ADEdge(VertexId from, VertexId to, Real w) : 
        from(from), to(to), w(w) {}

    VertexId from, to;
    Real w;
};

struct ADGraph {
    ADGraph() {
        g_ADGraph = this;
    }

    inline void clear() {
        num_vertices = 0;
        edges.clear();
    }

    inline void zero_adjoints() {
        adjoints.resize(num_vertices);
        std::fill(adjoints.begin(), adjoints.end(), 0.f);
    }

    int num_vertices;
    std::vector<float> adjoints;
    std::vector<ADEdge> edges;
};

inline AReal NewAReal(const Real val) {
    VertexId newId = g_ADGraph->num_vertices;
    g_ADGraph->num_vertices++;
    return AReal(val, newId);
}

inline void AddEdge(const AReal &c, const AReal &p, 
                    const Real w) {
    if (w != 0.f) {
        g_ADGraph->edges.emplace_back(c.varId, p.varId, w);
    }
}

////////////////////// Addition ///////////////////////////
inline AReal operator+(const AReal &l, const AReal &r) {
    AReal ret = NewAReal(l.val + r.val);
    AddEdge(ret, l, Real(1.0));
    AddEdge(ret, r, Real(1.0));
    return ret;
}
inline AReal operator+(const AReal &l, const Real r) {
    AReal ret = NewAReal(l.val + r);
    AddEdge(ret, l, Real(1.0));
    return ret;
}
inline AReal operator+(const Real l, const AReal &r) {
    return r + l;
}
inline AReal& operator+=(AReal &l, const AReal &r) {
    return (l = l + r);
}
inline AReal& operator+=(AReal &l, const Real r) {
    return (l = l + r);
}
///////////////////////////////////////////////////////////

////////////////// Subtraction ////////////////////////////
inline AReal operator-(const AReal &l, const AReal &r) {
    AReal ret = NewAReal(l.val - r.val);
    AddEdge(ret, l, Real(1.0));
    AddEdge(ret, r, Real(-1.0));
    return ret;
}
inline AReal operator-(const AReal &l, const Real r) {
    AReal ret = NewAReal(l.val - r);
    AddEdge(ret, l, Real(1.0));
    return ret;
}
inline AReal operator-(const Real l, const AReal &r) {
    AReal ret = NewAReal(l - r.val);
    AddEdge(ret, r, Real(-1.0));
    return ret;
}
inline AReal& operator-=(AReal &l, const AReal &r) {
    return (l = l - r);
}
inline AReal& operator-=(AReal &l, const Real r) {
    return (l = l - r);
}
inline AReal operator-(const AReal &x) {
    AReal ret = NewAReal(-x.val);
    AddEdge(ret, x, Real(-1.0));
    return ret;
}
///////////////////////////////////////////////////////////

////////////////// Multiplication /////////////////////////
inline AReal operator*(const AReal &l, const AReal &r) {
    AReal ret = NewAReal(l.val * r.val);
    AddEdge(ret, l, r.val);
    AddEdge(ret, r, l.val);
    return ret;
}
inline AReal operator*(const AReal &l, const Real r) {
    AReal ret = NewAReal(l.val * r);
    AddEdge(ret, l, r);
    return ret;
}
inline AReal operator*(const Real l, const AReal &r) {
    return r * l;
}
inline AReal& operator*=(AReal &l, const AReal &r) {
    return (l = l * r);
}
inline AReal& operator*=(AReal &l, const Real r) {
    return (l = l * r);
}
///////////////////////////////////////////////////////////

////////////////// Division ///////////////////////////////
inline AReal operator/(const AReal &l, const AReal &r) {
    // d/dx x / y = 1 / y
    // d/dy x / y = -x / y^2
    Real invR = Real(1.0) / r.val;
    Real invRSq = invR * invR;
    AReal ret = NewAReal(l.val * invR);
    AddEdge(ret, l, invR);
    AddEdge(ret, r, -l.val * invRSq);
    return ret;
}
inline AReal operator/(const AReal &l, const Real r) {
    return l * (Real(1.0) / r);
}
inline AReal operator/(const Real l, const AReal &r) {
    Real invR = Real(1.0) / r.val;
    Real invRSq = invR * invR;
    AReal ret = NewAReal(l * invR);
    AddEdge(ret, r, -l * invRSq);
    return ret;
}
inline AReal& operator/=(AReal &l, const AReal &r) {
    return (l = l / r);
}
inline AReal& operator/=(AReal &l, const Real r) {
    return (l = l / r);
}
///////////////////////////////////////////////////////////

////////////////// Comparisons ////////////////////////////
inline bool operator<(const AReal &l, const AReal &r) {
    return l.val < r.val;
}
inline bool operator<=(const AReal &l, const AReal &r) {
    return l.val <= r.val;
}
inline bool operator>(const AReal &l, const AReal &r) {
    return l.val > r.val;
}
inline bool operator>=(const AReal &l, const AReal &r) {
    return l.val >= r.val;
}
inline bool operator==(const AReal &l, const AReal &r) {
    return l.val == r.val;
}
inline bool operator!=(const AReal &l, const AReal &r) {
    return l.val != r.val;
}
///////////////////////////////////////////////////////////

//////////////// Misc functions ///////////////////////////
inline Real square(const Real x) {
    return x * x;
}
inline AReal square(const AReal &x) {
    Real sqX = x.val * x.val;
    AReal ret = NewAReal(sqX);
    AddEdge(ret, x, Real(2.0) * x.val);
    return ret;
}
inline AReal sqrt(const AReal &x) {
    Real sqrtX = std::sqrt(x.val);
    Real invSqrtX = Real(1.0) / sqrtX;
    AReal ret = NewAReal(sqrtX);
    AddEdge(ret, x, Real(0.5) * invSqrtX);
    return ret;
}
inline AReal pow(const AReal &x, const Real a) {
    Real powX = std::pow(x.val, a);
    AReal ret = NewAReal(powX);
    AddEdge(ret, x, a * std::pow(x.val, a - Real(1.0)));
    return ret;
}
inline AReal pow(const AReal &x, const AReal &a) {
    Real powX = std::pow(x.val, a.val);
    AReal ret = NewAReal(powX);
    AddEdge(ret, x, a.val * std::pow(x.val, a.val - Real(1.0)));
    AddEdge(ret, a, powX * log(std::max(x.val, 1e-6f)));
    return ret;
}
inline AReal exp(const AReal &x) {
    Real expX = std::exp(x.val);
    AReal ret = NewAReal(expX);
    AddEdge(ret, x, expX);
    return ret;
}
inline AReal log(const AReal &x) {
    Real logX = std::log(x.val);
    AReal ret = NewAReal(logX);
    Real invX = Real(1.0) / x.val;
    AddEdge(ret, x, invX);
    return ret;
}
inline AReal sin(const AReal &x) {
    Real sinX = std::sin(x.val);
    AReal ret = NewAReal(sinX);
    AddEdge(ret, x, std::cos(x.val));
    return ret;
}
inline AReal cos(const AReal &x) {
    AReal ret = NewAReal(std::cos(x.val));
    AddEdge(ret, x, -std::sin(x.val));
    return ret;
}
inline AReal tan(const AReal &x) {
    Real tanX = std::tan(x.val);
    Real secX = Real(1.0) / std::cos(x.val);
    Real sec2X = secX * secX;
    AReal ret = NewAReal(tanX);
    AddEdge(ret, x, sec2X);
    return ret;
}
inline AReal asin(const AReal &x) {
    Real asinX = std::asin(x.val);
    AReal ret = NewAReal(asinX);
    Real tmp = Real(1.0) / (Real(1.0) - x.val * x.val);
    Real sqrtTmp = std::sqrt(tmp);
    AddEdge(ret, x, sqrtTmp);
    return ret;
}
inline AReal acos(const AReal &x) {
    Real acosX = std::acos(x.val);
    AReal ret = NewAReal(acosX);
    Real tmp = Real(1.0) / (Real(1.0) - x.val * x.val);
    Real negSqrtTmp = -std::sqrt(tmp);
    AddEdge(ret, x, negSqrtTmp);
    return ret;
}
inline AReal atan2(const AReal &y, const AReal &x) {
    Real atan2xy = std::atan2(y.val, x.val);
    AReal ret = NewAReal(atan2xy);
    Real tmp = x.val * x.val + y.val * y.val;
    Real dx = -y.val / tmp;
    Real dy = x.val / tmp;
    AddEdge(ret, y, dy);
    AddEdge(ret, x, dx);
    return ret;
}
inline AReal fabs(const AReal &x) {
    return x.val >= 0.f ? x : -x;
}
inline AReal max(const AReal &x, const AReal &y) {
    return x.val > y.val ? x : y;
}
inline AReal max(const AReal &x, float y) {
    return x.val >= y ? x : AReal(y);
}
inline AReal max(float x, const AReal &y) {
    return x > y.val ? AReal(x) : y;
}
inline float max(float x, float y) {
    return x > y ? x : y;
}
inline AReal min(const AReal &x, const AReal &y) {
    return x.val < y.val ? x : y;
}
inline AReal min(const AReal &x, float y) {
    return x.val <= y ? x : AReal(y);
}
inline AReal min(float x, const AReal &y) {
    return x < y.val ? AReal(x) : y;
}
inline float min(float x, float y) {
    return x < y ? x : y;
}
///////////////////////////////////////////////////////////

template <typename TOut, typename TIn>
inline TOut convert(const TIn &v) {
    return TOut(v);
}

template <>
inline Real convert(const AReal &v) {
    return v.val;
}

inline std::ostream& operator<<(std::ostream &os, const AReal &v) {
    return os << convert<Real>(v);
}

inline void set_adjoint(const AReal &v, const Real adj) {
    g_ADGraph->adjoints[v.varId] = adj;
}

inline Real get_adjoint(const AReal &v) {
    return v.varId != -1 ?
        g_ADGraph->adjoints[v.varId] : Real(0);
}

inline void propagate_adjoint() {
    for (int i = (int)g_ADGraph->edges.size() - 1; i >= 0; i--) {
        const ADEdge &e = g_ADGraph->edges[i];
        float adj = g_ADGraph->adjoints[e.from];
        if (adj != 0.f) {
            g_ADGraph->adjoints[e.to] += e.w * adj;
        }
    }
}

} //namespace autodiff

#endif // AUTODIFF_H__
