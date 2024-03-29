#pragma once
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

bool CreateScene(hiprtContext context, hipStream_t stream, const hiprtSceneBuildInput& sceneBuildInput, hiprtScene& outScene);

bool CreateInstancesOneToOneFullMask(hiprtSceneBuildInput& sceneBuildInput, const std::vector<hiprtGeometry>& geometries);

bool CreateInstancesOneToOneFullMaskMBRight(hiprtSceneBuildInput& sceneBuildInput, const std::vector<hiprtGeometry>& geometries);

bool CreateInstancesOneToOneFullMask2InstancesMB(hiprtSceneBuildInput& sceneBuildInput, const std::vector<hiprtGeometry>& geometries);