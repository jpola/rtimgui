#pragma once
#include "Geometry.h"
#include "assert.h"

bool CreateGeometry(hiprtContext context, hipStream_t stream, hiprtBuildFlags buildFlags, const hiprtGeometryBuildInput& geometryBuildInput, hiprtGeometry& geometry)
{
    hiprtBuildOptions options;
    options.buildFlags = buildFlags;

    size_t geomTempSize{0};
    hiprtDevicePtr geomTemp{nullptr};

    HIP_ASSERT(hiprtGetGeometryBuildTemporaryBufferSize(context, geometryBuildInput, options, geomTempSize) == hiprtSuccess, "Failed to get geometry buffer size");
    HIP_ASSERT(hipMalloc(&geomTemp, geomTempSize) == hipSuccess, "Malloc failed");
    HIP_ASSERT(hiprtCreateGeometry(context, geometryBuildInput, options, geometry) == hiprtSuccess, "Failed to create geometry");
    HIP_ASSERT(hiprtBuildGeometry(context, hiprtBuildOperationBuild, geometryBuildInput, options, geomTemp, stream, geometry) == hiprtSuccess, "Failed to build geomtery");
    HIP_ASSERT(hipFree(geomTemp) == hipSuccess, "Failed to release temp ptr");

    return true;
}

bool CreateGeometries(hiprtContext context,
                      hipStream_t stream,
                      hiprtBuildFlags buildFlags,
                      const std::vector<hiprtGeometryBuildInput>& geometryBuildInputs,
                      std::vector<hiprtGeometry>& geometries)
{
    hiprtBuildOptions options;
    options.buildFlags = buildFlags;

    size_t geomTempSize{0};
    hiprtDevicePtr geomTemp{nullptr};

    HIP_ASSERT(hiprtGetGeometriesBuildTemporaryBufferSize(context, static_cast<uint32_t>(geometryBuildInputs.size()), geometryBuildInputs.data(), options, geomTempSize) ==
                   hiprtSuccess,
               "Failed to get geometries buffer size");
    HIP_ASSERT(hipMalloc(&geomTemp, geomTempSize) == hipSuccess, "Malloc failed");

    std::vector<hiprtGeometry*> geomAddrs;
    for (auto& geometry : geometries)
    {
        geomAddrs.push_back(&geometry);
    }
    HIP_ASSERT(hiprtCreateGeometries(context, static_cast<uint32_t>(geometryBuildInputs.size()), geometryBuildInputs.data(), options, geomAddrs.data()) == hiprtSuccess,
               "Failed to create geometry");

    HIP_ASSERT(hiprtBuildGeometries(context,
                                    hiprtBuildOperationBuild,
                                    static_cast<uint32_t>(geometryBuildInputs.size()),
                                    geometryBuildInputs.data(),
                                    options,
                                    geomTemp,
                                    stream,
                                    geometries.data()) == hiprtSuccess,
               "Failed to build geomtery");
    HIP_ASSERT(hipFree(geomTemp) == hipSuccess, "Failed to release temp ptr");

    return true;
}
