#pragma once
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

bool CreateGeometry(hiprtContext context, hipStream_t stream, hiprtBuildFlags buildFlags, const hiprtGeometryBuildInput& geometryBuildInput, hiprtGeometry& geometry);

bool CreateGeometries(hiprtContext context,
                      hipStream_t stream,
                      hiprtBuildFlags buildFlags,
                      const std::vector<hiprtGeometryBuildInput>& geometryBuildInputs,
                      std::vector<hiprtGeometry>& geometries);