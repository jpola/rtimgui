#pragma once

#include <vector>
#include <hip/hip_vector_types.h>
#include <hiprt/hiprt.h>

#include "Aabb.h"

struct BoundingBox3D
{
    float3 min_point{0, 0, 0};
    float3 max_point{0, 0, 0};
};

BoundingBox3D compute_axis_aligned_bounding_box(const std::vector<float3>& vertices);


enum class GEOMETRY_TYPE : uint32_t
{
    TRIANGLE_MESH = 0,
    //TRIANGLE_MESH_DEFORMED,

    AABB_LIST,
    
    GEOMETRY_TYPE_COUNT
};

struct TriangleMesh
{
    std::vector<float3> vertices;
    std::vector<uint3>  indices;
    std::vector<float3> vertex_normals;
    std::vector<float3> triangle_normals;
    std::vector<hiprt::Aabb> aabb;

    uint32_t deformation_count{1};

    hiprtTriangleMeshPrimitive mesh{};
    hiprtDevicePtr device_vertex_normals{nullptr}; 
    hiprtDevicePtr device_tirangle_normals {nullptr};
    hiprtDevicePtr device_aabb{nullptr};

    uint32_t GetNumVertices();
    void Build();
    void BuildAABB();
    hiprtGeometryBuildInput CreateBuildInput( GEOMETRY_TYPE type);

    TriangleMesh();
    TriangleMesh(const TriangleMesh& other) = default;
    TriangleMesh(TriangleMesh&& other) = default;
    
    ~TriangleMesh();
};

void BuildMeshes(std::vector<TriangleMesh>& meshes);

void CollectGeometryBuildInputs(std::vector<hiprtGeometryBuildInput>& inputs, std::vector<TriangleMesh>& meshes);

void ApplyDeformation(uint32_t numDeformations, TriangleMesh& mesh);