#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <math.h>
#include <iostream>
#include "../kernels/shared.h"
#include "Geometry.h"
#include "ImageWriter.h"
#include "MeshReader.h"
#include "Scene.h"
#include "TriangleMesh.h"
#include "assert.h"

#include "DisplayWindow.h"


void launchKernel(hipFunction_t func, int nx, int ny, void** args, hipStream_t stream = 0, size_t threadPerBlockX = 8, size_t threadPerBlockY = 8, size_t threadPerBlockZ = 1)
{
    size_t nBx = (nx + threadPerBlockX - 1) / threadPerBlockX;
    size_t nBy = (ny + threadPerBlockY - 1) / threadPerBlockY;
    HIP_ASSERT(hipModuleLaunchKernel(
                   func, (uint32_t) nBx, (uint32_t) nBy, 1, (uint32_t) threadPerBlockX, (uint32_t) threadPerBlockY, (uint32_t) threadPerBlockZ, 0, stream, args, 0) == hipSuccess,
               "Launch kernel");
}

struct GeometryData
{
    float3* vertices{nullptr};
    uint3* triangles{nullptr};
    float3* vertex_normals{nullptr};
    float3* triangle_normals{nullptr};
    uint32_t nUniqueTriangles{0};
    uint32_t nUniqueVertices{0};
    uint32_t nTriangles{0};
    uint32_t nVertices{0};
    uint32_t nDeformations{0};
    uint32_t geometryID{0};
    uint32_t instanceID{0};
};


#include "RenderCases.h"

int main(int argc, char const* argv[])
{
    std::cout << "Current working directory: " << fs::current_path() << "\n";
    HIP_ASSERT(hipInit(0) == hipSuccess, "hipInit");

    int deviceCount{-1};
    HIP_ASSERT(hipGetDeviceCount(&deviceCount) == hipSuccess, "hipGetDeviceCount");
    std::cout << "Device count: " << deviceCount << std::endl;

    int deviceId{0};
    HIP_ASSERT(hipSetDevice(deviceId) == hipSuccess, "hipSetDevice");

    hipCtx_t hipContext{nullptr};
    HIP_ASSERT(hipCtxCreate(&hipContext, 0, deviceId) == hipSuccess, "hipCtxCreate");

    hipStream_t stream{nullptr};
    HIP_ASSERT(hipStreamCreate(&stream) == hipSuccess, "hipStreamCreate");

    hipDeviceProp_t deviceProperties;
    HIP_ASSERT(hipGetDeviceProperties(&deviceProperties, deviceId) == hipSuccess, "hipGetDeviceProperties");
    std::cout << "Device: " << deviceProperties.name << std::endl;

    constexpr int hiprtApiVersion{2003};
    hiprtContextCreationInput input;
    input.ctxt = hipContext;
    input.device = deviceId;
    input.deviceType = hiprtDeviceAMD;

    hiprtContext rtContext{nullptr};
    HIP_ASSERT(hiprtCreateContext(hiprtApiVersion, input, rtContext) == hiprtSuccess, "hiprtCreateContext");



     std::vector<TriangleMesh> meshes;
    if (ReadObjMesh("../../scenes/sphere/s.obj", "../../scens/sphere/", meshes) == false)
    {
        return false;
    }

    TriangleMesh& mesh = *meshes.begin();

    mesh.Build();
    auto geometryBuildInput = mesh.CreateBuildInput(GEOMETRY_TYPE::TRIANGLE_MESH);
    hiprtGeometry geometry{};
    // CreateGeometry(context, stream, hiprtBuildFlagBitPreferFastBuild, geometryBuildInput, geometry);

    hiprtGeometryBuildInput geomInput;
    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.primitive.triangleMesh = mesh.mesh;

    size_t geomTempSize;
    hiprtDevicePtr geomTemp;
    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    HIP_ASSERT(hiprtGetGeometryBuildTemporaryBufferSize(rtContext, geomInput, options, geomTempSize) == hiprtSuccess, "build geometry");
    HIP_ASSERT(hipMalloc(&geomTemp, geomTempSize) == hipSuccess, "GeoTemp malloc");

    // hiprtGeometry geometry;
    HIP_ASSERT(hiprtSuccess == hiprtCreateGeometry(rtContext, geomInput, options, geometry), "Create geometry");
    HIP_ASSERT(hiprtSuccess == hiprtBuildGeometry(rtContext, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geometry));

    Camera camera;
    camera.m_translation = make_float3(0.0f, 0.f, 5.8f);
    camera.m_rotation = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    camera.m_fov = 45.0f * hiprt::Pi / 180.f;

    constexpr unsigned int height = 540;
    constexpr unsigned int width = 960;

    constexpr int stackSize = 64;
    constexpr int sharedStackSize = 16;
    constexpr int blockWidth = 8;
    constexpr int blockHeight = 8;
    constexpr int blockSize = blockWidth * blockHeight;
    float aoRadius = 1.4f;

    hiprtDevicePtr outputImage;
    HIP_ASSERT(hipMalloc(&outputImage, width * height * 4) == hipSuccess, "malloc");

    int2 resolution{width, height};

    hiprtGlobalStackBufferInput stackInput{hiprtStackTypeGlobal /*hiprtStackTypeDynamic*/, hiprtStackEntryTypeInteger, stackSize, height * width};

    hiprtGlobalStackBuffer globalStackBuffer;
    HIP_ASSERT(hiprtCreateGlobalStackBuffer(rtContext, stackInput, globalStackBuffer) == hiprtSuccess, "globalStack");

    hipModule_t module{nullptr};
    HIP_ASSERT(hipModuleLoad(&module, "trace.hipfb") == hipSuccess, "module load");
    hipFunction_t kernel{nullptr};
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "SimpleMeshIntersectionKernelCamera") == hipSuccess, "kernel load");

    void* kernel_args[] = {&geometry, &outputImage, &resolution, &camera};

    MainDisplayWindow MainWindow;      

    while (!MainWindow.ShouldClose())
    {
        MainWindow.PollEvents();
        MainWindow.Update();

        launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
        //HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

        MainWindow.Render();
    }

   

    
    
    
    writeImageFromDevice("test_image.png", width, height, outputImage);

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(rtContext, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyGeometry(rtContext, geometry) == hiprtSuccess, "Destroy geometries");



    /*
    Render<CASE_TYPE::GEOMETRY_HIT_DISTANCE>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "geometry_hit_distance.png");
    Render<CASE_TYPE::GEOMETRY_DEBUG>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "geometry_hit_distance.png");
    Render<CASE_TYPE::GEOMETRY_DEBUG_WITH_CAMERA>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "geometry_hit_distance.png");

    Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_SAMPLING>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "sampling_mb.png");
    Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_SLERP>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "trannsform_slerp.png");
    Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_AO_SLERP_2_INSTANCES>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "slerp_2_instances.png");
    Render<CASE_TYPE::SCENE_AMBIENT_OCCLUSION>(rtContext, stream, "../../scenes/cornellbox/cornellbox.obj", "../../scens/cornellbox/", "cb.png");
    */

    //Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_DEFORMATION>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "scene_transform_MB_deformation.png");

    HIP_ASSERT(hiprtDestroyContext(rtContext) == hiprtSuccess, "hiprtDestroyContext");
    HIP_ASSERT(hipStreamDestroy(stream) == hipSuccess, "hipStreamDestroy");
    HIP_ASSERT(hipCtxDestroy(hipContext) == hipSuccess, "hipCtxDestroy");

    std::cout << "Finished\n";
    return 0;
}
