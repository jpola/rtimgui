#include "TriangleMesh.h"
#include "assert.h"
#include "../kernels/Math.h"

#include <hip/hip_runtime.h>

 TriangleMesh::TriangleMesh()
 {
    mesh.triangleIndices = nullptr;
    mesh.vertices = nullptr;
 }

uint32_t TriangleMesh::GetNumVertices()
 {
    return vertices.size() / deformation_count;
 }

 void TriangleMesh::Build()
{
    BuildAABB();

    HIP_ASSERT( hipSuccess == hipMalloc(&device_aabb, aabb.size()*sizeof(hiprt::Aabb)), "aabb malloc");
    HIP_ASSERT( hipSuccess == hipMemcpyHtoD(device_aabb, aabb.data(), aabb.size()*sizeof(hiprt::Aabb)), "aabb copy");

    mesh.triangleCount = (uint32_t) indices.size();
    mesh.triangleStride = sizeof( uint3 );

    HIP_ASSERT( hipSuccess == hipMalloc(&mesh.triangleIndices, mesh.triangleCount * mesh.triangleStride), "hipMalloc");
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(mesh.triangleIndices, indices.data(), mesh.triangleCount * mesh.triangleStride), "memcpyH2D");
    
    mesh.vertexCount = vertices.size(); // GetNumVertices();
    mesh.vertexStride = sizeof( float3 );

    // mesh.vertexCount - 1st deformation
    float3* ptr = &vertices[0];
    
    HIP_ASSERT(hipSuccess == hipMalloc(&mesh.vertices, mesh.vertexCount * mesh.vertexStride), "hipMalloc");
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(mesh.vertices, ptr, mesh.vertexCount * mesh.vertexStride), "memcpyH2D");


    // send normals to device;
    HIP_ASSERT(hipSuccess == hipMalloc(&device_vertex_normals, vertex_normals.size()* sizeof(float3)), "hipMalloc");
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(device_vertex_normals, vertex_normals.data(), vertex_normals.size() * sizeof(float3)), "memcpyH2D");

    HIP_ASSERT(hipSuccess == hipMalloc(&device_tirangle_normals, triangle_normals.size() * sizeof(float3)), "hipMalloc");
    HIP_ASSERT(hipSuccess == hipMemcpyHtoD(device_tirangle_normals, triangle_normals.data(), triangle_normals.size() * sizeof(float3)), "memcpyH2D"); 
}

void TriangleMesh::BuildAABB()
{
    aabb.clear();
    aabb.reserve(indices.size());
    for(const auto& t : indices)
    {
        auto& v0 = vertices[t.x];
        auto& v1 = vertices[t.y];
        auto& v2 = vertices[t.z];

        hiprt::Aabb box(v0);
        box.grow(v1);
        box.grow(v2);

        aabb.push_back(box);
    }
}


hiprtGeometryBuildInput TriangleMesh::CreateBuildInput( GEOMETRY_TYPE type)
{
   
    hiprtGeometryBuildInput geometryBuildInput;
    memset(&geometryBuildInput, 0, sizeof(hiprtGeometryBuildInput));
    geometryBuildInput.geomType = static_cast<uint32_t>(type);
    switch (type)
    {
    case GEOMETRY_TYPE::TRIANGLE_MESH:
        geometryBuildInput.type = hiprtPrimitiveTypeTriangleMesh;
        geometryBuildInput.primitive.triangleMesh = mesh;
        break;
    case GEOMETRY_TYPE::AABB_LIST:
        geometryBuildInput.type = hiprtPrimitiveTypeAABBList;
        geometryBuildInput.primitive.aabbList.aabbCount = aabb.size();
        geometryBuildInput.primitive.aabbList.aabbStride = sizeof(hiprt::Aabb);
        geometryBuildInput.primitive.aabbList.aabbs = device_aabb;
        break;
    default:
        std::cerr << "Error to create GeometryType\n";
    }
    return geometryBuildInput; 
}

TriangleMesh::~TriangleMesh()
{
    HIP_ASSERT( hipSuccess == hipFree(mesh.vertices), "hipfree");
    HIP_ASSERT( hipSuccess == hipFree(mesh.triangleIndices), "hipfree");
    HIP_ASSERT(hipSuccess == hipFree(device_vertex_normals), "free normals");
    HIP_ASSERT(hipSuccess == hipFree(device_tirangle_normals), "free normals");
    HIP_ASSERT(hipSuccess == hipFree(device_aabb), "free aabbs");
}

void BuildMeshes(std::vector<TriangleMesh>& meshes)
{
    for(auto& mesh : meshes)
    {
        mesh.Build();
    }
}

void CollectGeometryBuildInputs(std::vector<hiprtGeometryBuildInput>& inputs, std::vector<TriangleMesh>& meshes)
{
    inputs.clear();
    inputs.reserve(meshes.size());
    for(auto& mesh: meshes)
    {
        if (mesh.deformation_count > 1)
        {
            inputs.push_back(mesh.CreateBuildInput(GEOMETRY_TYPE::AABB_LIST));
        }
        else 
            inputs.push_back(mesh.CreateBuildInput(GEOMETRY_TYPE::TRIANGLE_MESH));
    }
}


BoundingBox3D compute_axis_aligned_bounding_box(const std::vector<float3>& vertices)
{
    BoundingBox3D bbox;

    bbox.max_point = bbox.min_point = vertices[0];

    for (const auto& vertex : vertices)
    {
        bbox.min_point.x = std::min(bbox.min_point.x, vertex.x);
        bbox.min_point.y = std::min(bbox.min_point.y, vertex.y);
        bbox.min_point.z = std::min(bbox.min_point.z, vertex.z);
        bbox.max_point.x = std::max(bbox.max_point.x, vertex.x);
        bbox.max_point.y = std::max(bbox.max_point.y, vertex.y);
        bbox.max_point.z = std::max(bbox.max_point.z, vertex.z);
    }

    return bbox;
}

float3 compute_displacement_position(const float3& n, const float3& v, const float d)
{
    auto nn = hiprt::normalize(n);
    float3 displacement = nn * d;
    float3 dv = v + displacement;
    return dv;
}
    // explode the object in a direction of nomal for every face
void ApplyDeformation(uint32_t numDeformations, TriangleMesh& mesh )
{
    mesh.deformation_count += numDeformations;
    std::vector<float3>& vertices = mesh.vertices;
    std::vector<float3>& vertex_normals = mesh.vertex_normals;
    uint32_t numUnuqueVertuces = vertices.size();
    uint32_t numIndices = mesh.indices.size();
    std::vector<uint3> current_indices = mesh.indices;
    std::vector<float3> current_vertex_normals = mesh.vertex_normals;

    constexpr float displacement = 1.5;
    //easier to understand
    for (uint32_t step = 0; step < numDeformations; step++)
    {
        
        std::vector<float3> deformedVertices(numUnuqueVertuces, {0.f, 0.f, 0.f}); 
        std::vector<float3> deformedTriangleNormals(mesh.indices.size(), {0.f, 0.f, 0.f});

        uint32_t index{0};
        for (const auto& t : mesh.indices)
        {
            auto v0 = vertices[t.x];
            auto v1 = vertices[t.y];
            auto v2 = vertices[t.z];
    
            auto n0 = vertex_normals[t.x];
            auto n1 = vertex_normals[t.y];
            auto n2 = vertex_normals[t.z];

            v0 = compute_displacement_position(v0, n0, displacement);
            v1 = compute_displacement_position(v1, n1, displacement);
            v2 = compute_displacement_position(v2, n2, displacement);

            auto tn = mesh.triangle_normals[index];

            // same as tn, no need to compute
            /*{
                float3 ab = v1 - v0;
                float3 ac = v2 - v0;

                float3 vtn = hiprt::cross(ab, ac);
                vtn = hiprt::normalize(vtn);
            }*/

            deformedVertices[t.x] = v0;
            deformedVertices[t.y] = v1;
            deformedVertices[t.z] = v2;

            deformedTriangleNormals[index] = tn;

            index++;       
        }
        vertices.insert(vertices.end(), deformedVertices.begin(), deformedVertices.end());
        mesh.triangle_normals.insert(mesh.triangle_normals.end(), deformedTriangleNormals.begin(), deformedTriangleNormals.end());
        //// this are still the same values, but expanded for the other deformation steps;
        //mesh.indices.insert(mesh.indices.end(), current_indices.begin(), current_indices.end());
        //mesh.vertex_normals.insert(mesh.vertex_normals.end(), current_vertex_normals.begin(), current_vertex_normals.end());       
    }

}