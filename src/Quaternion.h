#pragma once

#include <hiprt/hiprt.h>

#define M_PI 3.14159265358979323846 // pi

typedef float4 Quaternion;

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/(const float3& a, const float& b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 normalize(const float3& a)
{
    return a / sqrtf(dot(a, a));
}

Quaternion angleAxisf(const float3& axis, float angle)
{
    auto s = std::sinf(angle / 2);
    auto u = normalize(axis);
    return {u.x * s, u.y * s, u.z * s, std::cosf(angle / 2)};
}

Quaternion lookAt(float3 sourcePoint, float3 destPoint, float3 front, float3 up)
{
    const float3 dir = destPoint - sourcePoint;
    const float3 toVector = normalize(dir);

    // compute rotation axis
    auto rotAxis = normalize(cross(front, toVector));
    if (dot(rotAxis, rotAxis) == 0)
        rotAxis = up;

    // find the angle around rotation axis
    float _dot = dot(front, toVector);
    float ang = std::acosf(_dot);

    // convert axis angle to quaternion
    return angleAxisf(rotAxis, ang);
}
