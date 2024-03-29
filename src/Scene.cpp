#include "Scene.h"
#include "assert.h"

bool CreateInstancesOneToOneFullMask(hiprtSceneBuildInput& sceneBuildInput, const std::vector<hiprtGeometry>& geometries)
{
    uint32_t numInstances = (uint32_t) geometries.size();
    std::vector<hiprtInstance> instances(numInstances);
    size_t geomIndex{0};
    for (auto& instance : instances)
    {
        instance.geometry = geometries[geomIndex++];
        instance.type = hiprtInstanceTypeGeometry;
    }

    uint32_t rayMask = hiprtFullRayMask;
    hiprtFrameSRT transform;
    {
        transform.translation = hiprtFloat3(0.0f, 0.0f, 0.0f);
        transform.scale = hiprtFloat3(1.0f, 1.0f, 1.0f);
        transform.rotation = hiprtFloat4(0.0f, 0.0f, 1.0f, 0.0f);
    }
    std::vector<hiprtFrameSRT> frames(numInstances, transform);

    std::vector<unsigned int> instanceMasks(numInstances, rayMask);

    sceneBuildInput.frameCount = numInstances; // no mb transformations
    sceneBuildInput.frameType = hiprtFrameTypeSRT;
    sceneBuildInput.instanceCount = numInstances;
    sceneBuildInput.instanceTransformHeaders = nullptr;

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceFrames, frames.size() * sizeof(hiprtFrameSRT)) == hipSuccess, "malloc frames");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instanceFrames, frames.data(), frames.size() * sizeof(hiprtFrameSRT)) == hipSuccess, "frames copy");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instances, instances.size() * sizeof(hiprtInstance)) == hipSuccess, "malloc instances");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instances, instances.data(), instances.size() * sizeof(hiprtInstance)) == hipSuccess, "instances copy");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceMasks, instanceMasks.size() * sizeof(uint32_t)) == hipSuccess, "malloc instances");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instanceMasks, instanceMasks.data(), instanceMasks.size() * sizeof(uint32_t)) == hipSuccess, "instances copy");

    return true;
}

void SetupTransformation(uint32_t nTransformationSteps, hiprtTransformHeader& outTransformHeader, std::vector<hiprtFrameSRT>& outFrames)
{
    outFrames.resize(nTransformationSteps);

    float time_fraction{1.f};
    if (nTransformationSteps > 1)
        time_fraction = 1.f / nTransformationSteps;

    outTransformHeader.frameIndex = 0;
    outTransformHeader.frameCount = nTransformationSteps;

    float time{0.f};

    float distanceOffset = 1.5;
    for (uint32_t step = 0; step < nTransformationSteps; step++)
    {
        auto& frame = outFrames[step];
        frame.translation.x += step * distanceOffset;
        frame.translation.y += step * distanceOffset;
        frame.time = time;
        time += time_fraction;
        // distanceOffset += distanceOffset;
    }
}

bool CreateInstancesOneToOneFullMaskMBRight(hiprtSceneBuildInput& sceneBuildInput, const std::vector<hiprtGeometry>& geometries)
{
    uint32_t numInstances = (uint32_t) geometries.size();
    std::vector<hiprtInstance> instances(numInstances);
    uint32_t rayMask = hiprtFullRayMask;
    size_t geomIndex{0};
    uint32_t nTransformatiationSteps = 3;

    hiprtFrameSRT transform;
    {
        // Lookout we are starting for -1. in y;
        transform.translation = hiprtFloat3(-1.0f, -1.0f, 0.0f);
        transform.scale = hiprtFloat3(1.0f, 1.0f, 1.0f);
        transform.rotation = hiprtFloat4(0.0f, 0.0f, 1.0f, 0.0f);
        transform.time = 0;
    }

    // buffers for all instances

    std::vector<hiprtFrameSRT> instancesFrames;
    instancesFrames.reserve(nTransformatiationSteps * numInstances);
    std::vector<hiprtTransformHeader> instancesTransformHeaders;
    instancesTransformHeaders.reserve(numInstances);

    uint32_t instanceIndex{0};
    for (auto& instance : instances)
    {
        instance.geometry = geometries[geomIndex++];
        instance.type = hiprtInstanceTypeGeometry;

        std::vector<hiprtFrameSRT> frames(nTransformatiationSteps, transform);
        hiprtTransformHeader transformHeader;
        SetupTransformation(nTransformatiationSteps, transformHeader, frames);

        for (auto& f : frames) instancesFrames.push_back(f);

        instancesTransformHeaders.push_back(transformHeader);
    }
    assert(instancesFrames.size() == nTransformatiationSteps * numInstances);

    std::vector<unsigned int> instanceMasks(numInstances, rayMask);

    sceneBuildInput.frameCount = instancesFrames.size();
    sceneBuildInput.frameType = hiprtFrameTypeSRT;
    sceneBuildInput.instanceCount = numInstances;

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceTransformHeaders, instancesTransformHeaders.size() * sizeof(hiprtTransformHeader)) == hipSuccess, "malloc transform header");
    HIP_ASSERT(
        hipMemcpyHtoD(sceneBuildInput.instanceTransformHeaders, instancesTransformHeaders.data(), instancesTransformHeaders.size() * sizeof(hiprtTransformHeader)) == hipSuccess,
        "copy instance headers frames");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceFrames, instancesFrames.size() * sizeof(hiprtFrameSRT)) == hipSuccess, "malloc frames");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instanceFrames, instancesFrames.data(), instancesFrames.size() * sizeof(hiprtFrameSRT)) == hipSuccess, "frames copy");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instances, instances.size() * sizeof(hiprtInstance)) == hipSuccess, "malloc instances");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instances, instances.data(), instances.size() * sizeof(hiprtInstance)) == hipSuccess, "instances copy");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceMasks, instanceMasks.size() * sizeof(uint32_t)) == hipSuccess, "malloc instances");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instanceMasks, instanceMasks.data(), instanceMasks.size() * sizeof(uint32_t)) == hipSuccess, "instances copy");

    return true;
}

bool CreateInstancesOneToOneFullMask2InstancesMB(hiprtSceneBuildInput& sceneBuildInput, const std::vector<hiprtGeometry>& geometries)
{
    constexpr uint32_t numInstances = 2;

    std::vector<hiprtInstance> instances(numInstances);
    uint32_t rayMask = hiprtFullRayMask;
    size_t geomIndex{0};

    // buffers for all instances

    std::vector<hiprtFrameSRT> instancesFrames;
    instancesFrames.reserve(numInstances);
    std::vector<hiprtTransformHeader> instancesTransformHeaders;
    instancesTransformHeaders.reserve(numInstances);

    uint32_t instanceIndex{0};
    uint32_t sumTransforms{0};
    for (auto& instance : instances)
    {
        instance.geometry = geometries[0];
        instance.type = hiprtInstanceTypeGeometry;

        //// setup first. It will sit in place;
        if (instanceIndex == 0)
        {
            hiprtFrameSRT transform;
            {
                // Lookout we are starting for -1. in y;
                transform.translation = hiprtFloat3(-1.0f, 1.0f, 0.0f);
                transform.scale = hiprtFloat3(1.0f, 1.0f, 1.0f);
                transform.rotation = hiprtFloat4(0.0f, 0.0f, 1.0f, 0.0f);
                transform.time = 0;
            }
            // setup other
            std::vector<hiprtFrameSRT> frames(1, transform);
            hiprtTransformHeader transformHeader;
            transformHeader.frameCount = 1;
            transformHeader.frameIndex = 0;
            for (auto& f : frames) instancesFrames.push_back(f);
            instancesTransformHeaders.push_back(transformHeader);
            sumTransforms += transformHeader.frameCount;
        }
        // setup other
        else
        {
            hiprtFrameSRT transform;
            {
                // Lookout we are starting for -1. in y;
                transform.translation = hiprtFloat3(-1.f, -1.0f, 0.0f);
                transform.scale = hiprtFloat3(1.0f, 1.0f, 1.0f);
                transform.rotation = hiprtFloat4(0.0f, 0.0f, 1.0f, 0.0f);
                transform.time = 0;
            }
            // just to quickly debug
            uint32_t nTransformatiationSteps = 2;
            if (nTransformatiationSteps == 0)
            {
                // setup other
                std::vector<hiprtFrameSRT> frames(1, transform);
                hiprtTransformHeader transformHeader{0, 0};
                transformHeader.frameCount = 1;
                //this is important
                // frameIndex points where the next set of transformation starts for the next instance 
                transformHeader.frameIndex += sumTransforms;
                for (auto& f : frames) instancesFrames.push_back(f);
                instancesTransformHeaders.push_back(transformHeader);
                sumTransforms += transformHeader.frameCount;
            }
            else
            {
                std::vector<hiprtFrameSRT> frames(nTransformatiationSteps, transform);
                hiprtTransformHeader transformHeader;
                SetupTransformation(nTransformatiationSteps, transformHeader, frames);
                transformHeader.frameIndex += sumTransforms;
                for (auto& f : frames) instancesFrames.push_back(f);

                instancesTransformHeaders.push_back(transformHeader);
                sumTransforms += transformHeader.frameCount;
            }
        }
        instanceIndex++;
    }
    assert(instancesFrames.size() == sumTransforms);

    std::vector<unsigned int> instanceMasks(numInstances, rayMask);

    sceneBuildInput.frameCount = instancesFrames.size();
    sceneBuildInput.frameType = hiprtFrameTypeSRT;
    sceneBuildInput.instanceCount = numInstances;

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceTransformHeaders, instancesTransformHeaders.size() * sizeof(hiprtTransformHeader)) == hipSuccess, "malloc transform header");
    HIP_ASSERT(
        hipMemcpyHtoD(sceneBuildInput.instanceTransformHeaders, instancesTransformHeaders.data(), instancesTransformHeaders.size() * sizeof(hiprtTransformHeader)) == hipSuccess,
        "copy instance headers frames");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceFrames, instancesFrames.size() * sizeof(hiprtFrameSRT)) == hipSuccess, "malloc frames");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instanceFrames, instancesFrames.data(), instancesFrames.size() * sizeof(hiprtFrameSRT)) == hipSuccess, "frames copy");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instances, instances.size() * sizeof(hiprtInstance)) == hipSuccess, "malloc instances");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instances, instances.data(), instances.size() * sizeof(hiprtInstance)) == hipSuccess, "instances copy");

    HIP_ASSERT(hipMalloc(&sceneBuildInput.instanceMasks, instanceMasks.size() * sizeof(uint32_t)) == hipSuccess, "malloc instances");
    HIP_ASSERT(hipMemcpyHtoD(sceneBuildInput.instanceMasks, instanceMasks.data(), instanceMasks.size() * sizeof(uint32_t)) == hipSuccess, "instances copy");

    return true;
}

bool CreateScene(hiprtContext context, hipStream_t stream, const hiprtSceneBuildInput& sceneBuildInput, hiprtScene& outScene)
{
    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t sceneTempSize{0};
    hiprtDevicePtr sceneTemp{nullptr};

    HIP_ASSERT(hiprtGetSceneBuildTemporaryBufferSize(context, sceneBuildInput, options, sceneTempSize) == hiprtSuccess, "scenetempsize");
    HIP_ASSERT(hipMalloc(&sceneTemp, sceneTempSize) == hiprtSuccess, "malloc");
    HIP_ASSERT(hiprtCreateScene(context, sceneBuildInput, options, outScene) == hiprtSuccess, "create scene");
    HIP_ASSERT(hiprtBuildScene(context, hiprtBuildOperationBuild, sceneBuildInput, options, sceneTemp, stream, outScene) == hiprtSuccess, "buildScene");
    HIP_ASSERT(hipFree(sceneTemp) == hipSuccess, "free");

    return true;
}
