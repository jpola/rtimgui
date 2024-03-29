#pragma once


enum class CASE_TYPE : uint32_t
{
    GEOMETRY_HIT_DISTANCE = 0,
    SCENE_HIT_DISTANCE,
    SCENE_AMBIENT_OCCLUSION,
    GEOMETRY_DEBUG,
    GEOMETRY_DEBUG_WITH_CAMERA,
    SCENE_TRANSFORMATION_MB_SAMPLING,
    SCENE_TRANSFORMATION_MB_SLERP,
    SCENE_TRANSFORMATION_MB_AO_SLERP_2_INSTANCES,
    SCENE_TRANSFORMATION_MB_DEFORMATION,

};

template<CASE_TYPE type>
bool Render(hiprtContext context, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    static_assert("Not implemented");
}

template<>
bool Render<CASE_TYPE::GEOMETRY_DEBUG>(hiprtContext context, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    hiprtTriangleMeshPrimitive mesh;
    mesh.triangleCount = 1;
    mesh.triangleStride = sizeof(int3);
    HIP_ASSERT(hipSuccess == hipMalloc(&mesh.triangleIndices, mesh.triangleCount * sizeof(int3)), "indices malloc");
    uint32_t idx[] = {0, 1, 2};
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(mesh.triangleIndices, &idx, mesh.triangleCount * sizeof(int3)), "indices cpy");

    mesh.vertexCount = 3;
    mesh.vertexStride = sizeof(float3);
    HIP_ASSERT(hipSuccess == hipMalloc(&mesh.vertices, mesh.vertexCount * sizeof(float3)), "vertices malloc");
    float3 v[] = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 0.0f}};
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(mesh.vertices, &v, mesh.vertexCount * sizeof(float3)), "vertices cpy");

    hiprtGeometryBuildInput geomInput;
    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.primitive.triangleMesh = mesh;

    size_t geomTempSize;
    hiprtDevicePtr geomTemp;
    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    HIP_ASSERT(hiprtGetGeometryBuildTemporaryBufferSize(context, geomInput, options, geomTempSize) == hiprtSuccess, "build geometry");
    HIP_ASSERT(hipMalloc(&geomTemp, geomTempSize) == hipSuccess, "GeoTemp malloc");

    hiprtGeometry geom;
    HIP_ASSERT(hiprtSuccess == hiprtCreateGeometry(context, geomInput, options, geom), "Create geometry");
    HIP_ASSERT(hiprtSuccess == hiprtBuildGeometry(context, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom));

    hipModule_t module{nullptr};
    HIP_ASSERT(hipModuleLoad(&module, "trace.hipfb") == hipSuccess, "module load");
    hipFunction_t kernel{nullptr};
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "SimpleMeshIntersectionKernel") == hipSuccess, "kernel load");

    constexpr unsigned int height = 540;
    constexpr unsigned int width = 960;

    constexpr int stackSize = 64;
    constexpr int sharedStackSize = 16;
    constexpr int blockWidth = 8;
    constexpr int blockHeight = 8;
    constexpr int blockSize = blockWidth * blockHeight;
    float aoRadius = 1.4f;

    uint8_t* dst;
    HIP_ASSERT(hipMalloc(&dst, width * height * 4) == hipSuccess, "dest malloc");
    int2 res = make_int2(width, height);

    void* args[] = {&geom, &dst, &res};
    launchKernel(kernel, width, height, args, stream);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");
    writeImageFromDevice(output.string().c_str(), width, height, dst);

    hipFree(mesh.triangleIndices);
    hipFree(mesh.vertices);
    hipFree(geomTemp);
    hipFree(dst);
    HIP_ASSERT(hiprtSuccess == hiprtDestroyGeometry(context, geom), "destroy geometry");

    return true;
}

template<>
bool Render<CASE_TYPE::GEOMETRY_DEBUG_WITH_CAMERA>(hiprtContext context, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    hiprtTriangleMeshPrimitive mesh;
    mesh.triangleCount = 1;
    mesh.triangleStride = sizeof(int3);
    HIP_ASSERT(hipSuccess == hipMalloc(&mesh.triangleIndices, mesh.triangleCount * sizeof(int3)), "indices malloc");
    uint32_t idx[] = {0, 1, 2};
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(mesh.triangleIndices, &idx, mesh.triangleCount * sizeof(int3)), "indices cpy");

    mesh.vertexCount = 3;
    mesh.vertexStride = sizeof(float3);
    HIP_ASSERT(hipSuccess == hipMalloc(&mesh.vertices, mesh.vertexCount * sizeof(float3)), "vertices malloc");
    float3 v[] = {{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.f, 2.0f, 0.0f}};
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(mesh.vertices, &v, mesh.vertexCount * sizeof(float3)), "vertices cpy");

    hiprtGeometryBuildInput geomInput;
    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.primitive.triangleMesh = mesh;

    size_t geomTempSize;
    hiprtDevicePtr geomTemp;
    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    HIP_ASSERT(hiprtGetGeometryBuildTemporaryBufferSize(context, geomInput, options, geomTempSize) == hiprtSuccess, "build geometry");
    HIP_ASSERT(hipMalloc(&geomTemp, geomTempSize) == hipSuccess, "GeoTemp malloc");

    hiprtGeometry geom;
    HIP_ASSERT(hiprtSuccess == hiprtCreateGeometry(context, geomInput, options, geom), "Create geometry");
    HIP_ASSERT(hiprtSuccess == hiprtBuildGeometry(context, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom));

    hipModule_t module{nullptr};
    HIP_ASSERT(hipModuleLoad(&module, "trace.hipfb") == hipSuccess, "module load");
    hipFunction_t kernel{nullptr};
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "SimpleMeshIntersectionKernelCamera") == hipSuccess, "kernel load");

    constexpr unsigned int height = 10;
    constexpr unsigned int width = 16;
    //
    Camera camera;
    camera.m_translation = make_float3(0.0f, 0.f, 5.8f);
    camera.m_rotation = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    camera.m_fov = 45.0f * hiprt::Pi / 180.f;

    constexpr int stackSize = 64;
    constexpr int sharedStackSize = 16;
    constexpr int blockWidth = 8;
    constexpr int blockHeight = 8;
    constexpr int blockSize = blockWidth * blockHeight;
    float aoRadius = 1.4f;

    uint8_t* dst;
    HIP_ASSERT(hipMalloc(&dst, width * height * 4) == hipSuccess, "dest malloc");
    int2 res = make_int2(width, height);

    void* args[] = {&geom, &dst, &res, &camera};
    launchKernel(kernel, width, height, args, stream);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");
    writeImageFromDevice(output.string().c_str(), width, height, dst);

    hipFree(mesh.triangleIndices);
    hipFree(mesh.vertices);
    hipFree(geomTemp);
    hipFree(dst);

    HIP_ASSERT(hiprtSuccess == hiprtDestroyGeometry(context, geom), "destroy geometry");

    return true;
}

template<>
bool Render<CASE_TYPE::GEOMETRY_HIT_DISTANCE>(hiprtContext context, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    std::vector<TriangleMesh> meshes;
    if (ReadObjMesh(meshPath, mtlPath, meshes) == false)
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
    HIP_ASSERT(hiprtGetGeometryBuildTemporaryBufferSize(context, geomInput, options, geomTempSize) == hiprtSuccess, "build geometry");
    HIP_ASSERT(hipMalloc(&geomTemp, geomTempSize) == hipSuccess, "GeoTemp malloc");

    // hiprtGeometry geometry;
    HIP_ASSERT(hiprtSuccess == hiprtCreateGeometry(context, geomInput, options, geometry), "Create geometry");
    HIP_ASSERT(hiprtSuccess == hiprtBuildGeometry(context, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geometry));

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
    HIP_ASSERT(hiprtCreateGlobalStackBuffer(context, stackInput, globalStackBuffer) == hiprtSuccess, "globalStack");

    hipModule_t module{nullptr};
    HIP_ASSERT(hipModuleLoad(&module, "trace.hipfb") == hipSuccess, "module load");
    hipFunction_t kernel{nullptr};
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "SimpleMeshIntersectionKernelCamera") == hipSuccess, "kernel load");

    void* kernel_args[] = {&geometry, &outputImage, &resolution, &camera};
    launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

    writeImageFromDevice(output.string().c_str(), width, height, outputImage);

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(context, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyGeometry(context, geometry) == hiprtSuccess, "Destroy geometries");

    return true;
}

template<>
bool Render<CASE_TYPE::SCENE_AMBIENT_OCCLUSION>(hiprtContext rtContext, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    std::vector<TriangleMesh> meshes;

    if (ReadObjMesh(meshPath, mtlPath, meshes) == false)
    {
        return false;
    }

    std::vector<hiprtGeometryBuildInput> geometryBuildInputs;
    BuildMeshes(meshes);
    CollectGeometryBuildInputs(geometryBuildInputs, meshes);
    std::vector<hiprtGeometry> geometries(meshes.size());
    hiprtBuildOptions geomBuildOptions;
    geomBuildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    CreateGeometries(rtContext, stream, geomBuildOptions.buildFlags, geometryBuildInputs, geometries);

    hiprtSceneBuildInput sceneBuildInput;
    memset(&sceneBuildInput, 0, sizeof(hiprtSceneBuildInput)); // fuck!, this is important
    CreateInstancesOneToOneFullMask(sceneBuildInput, geometries);

    hiprtScene scene;
    CreateScene(rtContext, stream, sceneBuildInput, scene);

    std::vector<GeometryData> geometryData(meshes.size());
    int index{0};
    for (auto& mesh : meshes)
    {
        GeometryData& data = geometryData[index++];
        data.geometryID = index;
        data.instanceID = index;
        data.nTriangles = mesh.indices.size();
        data.nVertices =  mesh.vertices.size();
        data.nDeformations = mesh.deformation_count;
        data.triangles = reinterpret_cast<uint3*>(mesh.mesh.triangleIndices);
        data.vertices = reinterpret_cast<float3*>(mesh.mesh.vertices);
    }

    hiprtDevicePtr deviceGeometryData{nullptr};
    HIP_ASSERT(hipMalloc(&deviceGeometryData, geometryData.size() * sizeof(GeometryData)) == hipSuccess, "malloc");
    HIP_ASSERT(hipMemcpyHtoD(deviceGeometryData, geometryData.data(), geometryData.size() * sizeof(GeometryData)) == hipSuccess, "cpy");

    hiprtFuncDataSet funcDataSet;
    funcDataSet.intersectFuncData = (void*) deviceGeometryData;
    funcDataSet.filterFuncData = (void*) deviceGeometryData;

    hiprtFuncTable funcTable;
    hiprtError result = hiprtCreateFuncTable(rtContext, 1, 1, funcTable);

    // Camera
    Camera camera;
    camera.m_translation = make_float3(0.0f, 2.0f, 4.8f);
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
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "AoRayKernel") == hipSuccess, "kernel load");

    int maxThreadsPerBlock{0};
    int numRegs{0};
    int constSizeBytes{0};
    int localSizeBytes{0};
    int maxDynamicSharedSizeBytes{0};
    int sharedSizeBytes{0};

    void* kernel_args[] = {&scene, &outputImage, &resolution, &globalStackBuffer, &camera, &aoRadius, &funcTable};
    launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

    writeImageFromDevice(output.string().c_str(), width, height, outputImage);

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instances) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instanceFrames) == hipSuccess, "free");
    HIP_ASSERT(hipFree(deviceGeometryData) == hipSuccess, "free");

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(rtContext, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyFuncTable(rtContext, funcTable) == hiprtSuccess, "functioniTable");
    HIP_ASSERT(hiprtDestroyGeometries(rtContext, geometries.size(), geometries.data()) == hiprtSuccess, "Destroy geometries");
    HIP_ASSERT(hiprtDestroyScene(rtContext, scene) == hiprtSuccess, "destroyScene");

    return true;
}

template<>
bool Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_SAMPLING>(hiprtContext rtContext, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    std::vector<TriangleMesh> meshes;

    if (ReadObjMesh(meshPath, mtlPath, meshes) == false)
    {
        return false;
    }

    std::vector<hiprtGeometryBuildInput> geometryBuildInputs;
    BuildMeshes(meshes);
    CollectGeometryBuildInputs(geometryBuildInputs, meshes);
    std::vector<hiprtGeometry> geometries(meshes.size());
    hiprtBuildOptions geomBuildOptions;
    geomBuildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    CreateGeometries(rtContext, stream, geomBuildOptions.buildFlags, geometryBuildInputs, geometries);

    hiprtSceneBuildInput sceneBuildInput;
    memset(&sceneBuildInput, 0, sizeof(hiprtSceneBuildInput)); // fuck!, this is important
    CreateInstancesOneToOneFullMaskMBRight(sceneBuildInput, geometries);

    hiprtScene scene;
    CreateScene(rtContext, stream, sceneBuildInput, scene);

    std::vector<GeometryData> geometryData(meshes.size());
    int index{0};
    for (auto& mesh : meshes)
    {
        GeometryData& data = geometryData[index++];
        data.geometryID = index;
        data.instanceID = index;
        data.nTriangles = mesh.indices.size();
        data.nVertices = mesh.vertices.size();
        data.nDeformations = mesh.deformation_count;
        data.triangles = reinterpret_cast<uint3*>(mesh.mesh.triangleIndices);
        data.vertices = reinterpret_cast<float3*>(mesh.mesh.vertices);
    }

    hiprtDevicePtr deviceGeometryData{nullptr};
    HIP_ASSERT(hipMalloc(&deviceGeometryData, geometryData.size() * sizeof(GeometryData)) == hipSuccess, "malloc");
    HIP_ASSERT(hipMemcpyHtoD(deviceGeometryData, geometryData.data(), geometryData.size() * sizeof(GeometryData)) == hipSuccess, "cpy");

    hiprtFuncDataSet funcDataSet;
    funcDataSet.intersectFuncData = (void*) deviceGeometryData;
    funcDataSet.filterFuncData = (void*) deviceGeometryData;

    hiprtFuncTable funcTable;
    hiprtError result = hiprtCreateFuncTable(rtContext, 1, 1, funcTable);

    // Camera
    Camera camera;
    camera.m_translation = make_float3(0.0f, 0.0f, 5.8f);
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
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "MotionBlurrRayKernelSampling") == hipSuccess, "kernel load");

    int maxThreadsPerBlock{0};
    int numRegs{0};
    int constSizeBytes{0};
    int localSizeBytes{0};
    int maxDynamicSharedSizeBytes{0};
    int sharedSizeBytes{0};

    void* kernel_args[] = {&scene, &outputImage, &resolution, &globalStackBuffer, &camera, &aoRadius, &funcTable};
    launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

    writeImageFromDevice(output.string().c_str(), width, height, outputImage);

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instances) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instanceFrames) == hipSuccess, "free");
    HIP_ASSERT(hipFree(deviceGeometryData) == hipSuccess, "free");

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(rtContext, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyFuncTable(rtContext, funcTable) == hiprtSuccess, "functioniTable");
    HIP_ASSERT(hiprtDestroyGeometries(rtContext, geometries.size(), geometries.data()) == hiprtSuccess, "Destroy geometries");
    HIP_ASSERT(hiprtDestroyScene(rtContext, scene) == hiprtSuccess, "destroyScene");

    return true;
}

template<>
bool Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_SLERP>(hiprtContext rtContext, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    std::vector<TriangleMesh> meshes;

    if (ReadObjMesh(meshPath, mtlPath, meshes) == false)
    {
        return false;
    }

    std::vector<hiprtGeometryBuildInput> geometryBuildInputs;
    BuildMeshes(meshes);
    CollectGeometryBuildInputs(geometryBuildInputs, meshes);
    std::vector<hiprtGeometry> geometries(meshes.size());
    hiprtBuildOptions geomBuildOptions;
    geomBuildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    CreateGeometries(rtContext, stream, geomBuildOptions.buildFlags, geometryBuildInputs, geometries);

    hiprtSceneBuildInput sceneBuildInput;
    memset(&sceneBuildInput, 0, sizeof(hiprtSceneBuildInput)); // fuck!, this is important
    CreateInstancesOneToOneFullMaskMBRight(sceneBuildInput, geometries);

    hiprtScene scene;
    CreateScene(rtContext, stream, sceneBuildInput, scene);

    std::vector<GeometryData> geometryData(meshes.size());
    int index{0};
    for (auto& mesh : meshes)
    {
        GeometryData& data = geometryData[index++];
        data.geometryID = index;
        data.instanceID = index;
        data.nTriangles = mesh.indices.size();
        data.nVertices = mesh.vertices.size();
        data.nDeformations = mesh.deformation_count;
        data.triangles = reinterpret_cast<uint3*>(mesh.mesh.triangleIndices);
        data.vertices = reinterpret_cast<float3*>(mesh.mesh.vertices);
    }

    hiprtDevicePtr deviceGeometryData{nullptr};
    HIP_ASSERT(hipMalloc(&deviceGeometryData, geometryData.size() * sizeof(GeometryData)) == hipSuccess, "malloc");
    HIP_ASSERT(hipMemcpyHtoD(deviceGeometryData, geometryData.data(), geometryData.size() * sizeof(GeometryData)) == hipSuccess, "cpy");

    hiprtFuncDataSet funcDataSet;
    funcDataSet.intersectFuncData = (void*) deviceGeometryData;
    funcDataSet.filterFuncData = (void*) deviceGeometryData;

    hiprtFuncTable funcTable;
    hiprtError result = hiprtCreateFuncTable(rtContext, 1, 1, funcTable);

    // Camera
    Camera camera;
    camera.m_translation = make_float3(0.0f, 0.0f, 5.8f);
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
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "MotionBlurrRayKernelSlerp") == hipSuccess, "kernel load");

    int maxThreadsPerBlock{0};
    int numRegs{0};
    int constSizeBytes{0};
    int localSizeBytes{0};
    int maxDynamicSharedSizeBytes{0};
    int sharedSizeBytes{0};

    void* kernel_args[] = {&scene, &outputImage, &resolution, &globalStackBuffer, &camera, &aoRadius, &funcTable};
    launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

    writeImageFromDevice(output.string().c_str(), width, height, outputImage);

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instances) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instanceFrames) == hipSuccess, "free");
    HIP_ASSERT(hipFree(deviceGeometryData) == hipSuccess, "free");

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(rtContext, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyFuncTable(rtContext, funcTable) == hiprtSuccess, "functioniTable");
    HIP_ASSERT(hiprtDestroyGeometries(rtContext, geometries.size(), geometries.data()) == hiprtSuccess, "Destroy geometries");
    HIP_ASSERT(hiprtDestroyScene(rtContext, scene) == hiprtSuccess, "destroyScene");

    return true;
}

template<>
bool Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_AO_SLERP_2_INSTANCES>(hiprtContext rtContext, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath,const fs::path output)
{
    std::vector<TriangleMesh> meshes;

    if (ReadObjMesh(meshPath, mtlPath, meshes) == false)
    {
        return false;
    }

    std::vector<hiprtGeometryBuildInput> geometryBuildInputs;
    BuildMeshes(meshes);
    CollectGeometryBuildInputs(geometryBuildInputs, meshes);
    std::vector<hiprtGeometry> geometries(meshes.size());
    hiprtBuildOptions geomBuildOptions;
    geomBuildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    CreateGeometries(rtContext, stream, geomBuildOptions.buildFlags, geometryBuildInputs, geometries);

    hiprtSceneBuildInput sceneBuildInput;
    memset(&sceneBuildInput, 0, sizeof(hiprtSceneBuildInput)); // fuck!, this is important
    CreateInstancesOneToOneFullMask2InstancesMB(sceneBuildInput, geometries);

    hiprtScene scene;
    CreateScene(rtContext, stream, sceneBuildInput, scene);

    std::vector<GeometryData> geometryData(meshes.size());
    int index{0};
    for (auto& mesh : meshes)
    {
        GeometryData& data = geometryData[index++];
        data.geometryID = index;
        data.instanceID = index;
        data.nTriangles = mesh.indices.size();
        data.nVertices = mesh.vertices.size();
        data.nDeformations = mesh.deformation_count;
        data.triangles = reinterpret_cast<uint3*>(mesh.mesh.triangleIndices);
        data.vertices = reinterpret_cast<float3*>(mesh.mesh.vertices);
    }

    hiprtDevicePtr deviceGeometryData{nullptr};
    HIP_ASSERT(hipMalloc(&deviceGeometryData, geometryData.size() * sizeof(GeometryData)) == hipSuccess, "malloc");
    HIP_ASSERT(hipMemcpyHtoD(deviceGeometryData, geometryData.data(), geometryData.size() * sizeof(GeometryData)) == hipSuccess, "cpy");

    hiprtFuncDataSet funcDataSet;
    funcDataSet.intersectFuncData = (void*) deviceGeometryData;
    funcDataSet.filterFuncData = (void*) deviceGeometryData;

    hiprtFuncTable funcTable;
    hiprtError result = hiprtCreateFuncTable(rtContext, 1, 1, funcTable);

    // Camera
    Camera camera;
    camera.m_translation = make_float3(0.0f, 0.0f, 5.8f);
    camera.m_rotation = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    camera.m_fov = 45.0f * hiprt::Pi / 180.f;

    constexpr unsigned int height = 540;
    constexpr unsigned int width = 960;

    constexpr int stackSize = 64;
    constexpr int sharedStackSize = 16;
    constexpr int blockWidth = 8;
    constexpr int blockHeight = 8;
    constexpr int blockSize = blockWidth * blockHeight;
    float aoRadius = 0.4f;

    hiprtDevicePtr outputImage;
    HIP_ASSERT(hipMalloc(&outputImage, width * height * 4) == hipSuccess, "malloc");

    int2 resolution{width, height};

    hiprtGlobalStackBufferInput stackInput{hiprtStackTypeGlobal /*hiprtStackTypeDynamic*/, hiprtStackEntryTypeInteger, stackSize, height * width};

    hiprtGlobalStackBuffer globalStackBuffer;
    HIP_ASSERT(hiprtCreateGlobalStackBuffer(rtContext, stackInput, globalStackBuffer) == hiprtSuccess, "globalStack");

    hipModule_t module{nullptr};
    HIP_ASSERT(hipModuleLoad(&module, "trace.hipfb") == hipSuccess, "module load");
    hipFunction_t kernel{nullptr};
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "AoRayKernelMotionBlurSlerp") == hipSuccess, "kernel load");

    int maxThreadsPerBlock{0};
    int numRegs{0};
    int constSizeBytes{0};
    int localSizeBytes{0};
    int maxDynamicSharedSizeBytes{0};
    int sharedSizeBytes{0};

    void* kernel_args[] = {&scene, &outputImage, &resolution, &globalStackBuffer, &camera, &aoRadius, &funcTable};
    launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

    writeImageFromDevice(output.string().c_str(), width, height, outputImage);

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instances) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instanceFrames) == hipSuccess, "free");
    HIP_ASSERT(hipFree(deviceGeometryData) == hipSuccess, "free");

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(rtContext, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyFuncTable(rtContext, funcTable) == hiprtSuccess, "functioniTable");
    HIP_ASSERT(hiprtDestroyGeometries(rtContext, geometries.size(), geometries.data()) == hiprtSuccess, "Destroy geometries");
    HIP_ASSERT(hiprtDestroyScene(rtContext, scene) == hiprtSuccess, "destroyScene");

    return true;
}

template<>
bool Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_DEFORMATION>(hiprtContext rtContext, hipStream_t stream, const fs::path& meshPath, const fs::path& mtlPath, const fs::path output)
{
    std::vector<TriangleMesh> meshes;

    if (ReadObjMesh(meshPath, mtlPath, meshes) == false)
    {
        return false;
    }

    for (auto& mesh: meshes)
    {
        ApplyDeformation(2, mesh);
    }

    std::vector<hiprtGeometryBuildInput> geometryBuildInputs;
    BuildMeshes(meshes);
    CollectGeometryBuildInputs(geometryBuildInputs, meshes);
    std::vector<hiprtGeometry> geometries(meshes.size());
    hiprtBuildOptions geomBuildOptions;
    geomBuildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    CreateGeometries(rtContext, stream, geomBuildOptions.buildFlags, geometryBuildInputs, geometries);

    hiprtSceneBuildInput sceneBuildInput;
    memset(&sceneBuildInput, 0, sizeof(hiprtSceneBuildInput)); // fuck!, this is important
    CreateInstancesOneToOneFullMaskMBRight(sceneBuildInput, geometries);

    hiprtScene scene;
    CreateScene(rtContext, stream, sceneBuildInput, scene);

    std::vector<GeometryData> geometryData(meshes.size());
    int index{0};
    for (auto& mesh : meshes)
    {
        GeometryData& data = geometryData[index++];
        data.geometryID = index;
        data.instanceID = index;
        data.nUniqueTriangles = mesh.indices.size() / mesh.deformation_count;
        data.nUniqueVertices = mesh.vertices.size() / mesh.deformation_count;
        data.nTriangles = mesh.indices.size();
        data.nVertices = mesh.vertices.size();
        data.nDeformations = mesh.deformation_count;
        data.triangles = reinterpret_cast<uint3*>(mesh.mesh.triangleIndices);
        data.vertices = reinterpret_cast<float3*>(mesh.mesh.vertices);
        data.vertex_normals = reinterpret_cast<float3*>(mesh.device_vertex_normals);
        data.triangle_normals = reinterpret_cast<float3*>(mesh.device_tirangle_normals);
    }

    hiprtDevicePtr deviceGeometryData{nullptr};
    HIP_ASSERT(hipMalloc(&deviceGeometryData, geometryData.size() * sizeof(GeometryData)) == hipSuccess, "malloc");
    HIP_ASSERT(hipMemcpyHtoD(deviceGeometryData, geometryData.data(), geometryData.size() * sizeof(GeometryData)) == hipSuccess, "cpy");

    hiprtFuncDataSet funcDataSet;
    funcDataSet.intersectFuncData = (void*) deviceGeometryData;
    funcDataSet.filterFuncData = (void*) deviceGeometryData;

    constexpr uint32_t numRays = 1;
    constexpr uint32_t numGeometries = static_cast<uint32_t>(GEOMETRY_TYPE::GEOMETRY_TYPE_COUNT);
    hiprtFuncTable funcTable;
    HIP_ASSERT(hiprtSuccess == hiprtCreateFuncTable(rtContext, numGeometries, numRays, funcTable), "function table");

    // set the data for anyhit_filter (Ray = 1, G=0 (tirangle mesh));
    for (uint32_t ray = 0; ray < numRays; ray++)
    {
        for (uint32_t geometry = 0; geometry < numGeometries; geometry++)
        {
            HIP_ASSERT(hiprtSuccess == hiprtSetFuncTable(rtContext, funcTable, geometry, ray, funcDataSet), "Function table set"); 
        }
    }

    // Camera
    Camera camera;
    camera.m_translation = make_float3(0.0f, 0.0f, 5.8f);
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
    HIP_ASSERT(hipModuleGetFunction(&kernel, module, "MotionBlurrRayKernelSamplingDeformation") == hipSuccess, "kernel load");

    int maxThreadsPerBlock{0};
    int numRegs{0};
    int constSizeBytes{0};
    int localSizeBytes{0};
    int maxDynamicSharedSizeBytes{0};
    int sharedSizeBytes{0};

    void* kernel_args[] = {&scene, &outputImage, &resolution, &globalStackBuffer, &camera, &aoRadius, &funcTable, &deviceGeometryData};
    launchKernel(kernel, width, height, kernel_args, stream, blockWidth, blockHeight);
    HIP_ASSERT(hipStreamSynchronize(stream) == hipSuccess, "stream sync");

    writeImageFromDevice(output.string().c_str(), width, height, outputImage);

    HIP_ASSERT(hipFree(outputImage) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instances) == hipSuccess, "free");
    HIP_ASSERT(hipFree(sceneBuildInput.instanceFrames) == hipSuccess, "free");
    HIP_ASSERT(hipFree(deviceGeometryData) == hipSuccess, "free");

    HIP_ASSERT(hipModuleUnload(module) == hipSuccess, "module unload");

    HIP_ASSERT(hiprtDestroyGlobalStackBuffer(rtContext, globalStackBuffer) == hiprtSuccess, "stack buffer");
    HIP_ASSERT(hiprtDestroyFuncTable(rtContext, funcTable) == hiprtSuccess, "functioniTable");
    HIP_ASSERT(hiprtDestroyGeometries(rtContext, geometries.size(), geometries.data()) == hiprtSuccess, "Destroy geometries");
    HIP_ASSERT(hiprtDestroyScene(rtContext, scene) == hiprtSuccess, "destroyScene");

    return true;
}
