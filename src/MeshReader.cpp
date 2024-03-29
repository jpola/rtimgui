#pragma once

#include "MeshReader.h"
#include <stl_reader.h>
#include <tiny_obj_loader.h>
#include <iostream>
#include "../kernels/Math.h"

namespace fs = std::filesystem;

bool ReadStlMesh(const fs::path& path, TriangleMesh& mesh)
{
    stl_reader::StlMesh <float, unsigned int> stl(path.string());

    std::cout << "Stl mesh solids:" << stl.num_solids() << " triangles:" << stl.num_tris() << " vertices:" << stl.num_vrts() << "\n";
    
    mesh.indices.reserve(stl.num_tris());
    mesh.vertices.reserve(stl.num_vrts());

	for (size_t itri = stl.solid_tris_begin(0); itri < stl.solid_tris_end(0); ++itri) 
	{
		float3 v = {*stl.vrt_coords(stl.tri_corner_ind(itri, 0)),
                    *stl.vrt_coords(stl.tri_corner_ind(itri, 1)),
					*stl.vrt_coords(stl.tri_corner_ind(itri, 2)) };
		auto ti = stl.tri_corner_inds(itri);
		uint3 t = { ti[0], ti[1], ti[2] };

		mesh.indices.push_back(t);
		mesh.vertices.push_back(v);
	}
	return true;
}


bool ReadObjMesh(const fs::path& meshPath, const fs::path& mtlBaseDir, std::vector<TriangleMesh>& meshes)
{

	tinyobj::attrib_t				 attrib;
	std::vector<tinyobj::shape_t>	 shapes;
	std::vector<tinyobj::material_t> materials;
	std::string						 err;
	std::string						 warning;

	bool ret = tinyobj::LoadObj( &attrib, &shapes, &materials, &warning, &err, meshPath.string().c_str(), mtlBaseDir.string().c_str() );
	
	if ( !warning.empty() )
	{
		std::cerr << "OBJ Loader WARN : " << warning << std::endl;
	}

	if ( !err.empty() )
	{
		std::cerr << "OBJ Loader ERROR : " << err << std::endl;
		return false;
	}

	if ( !ret )
	{
		std::cerr << "Failed to load obj file" << std::endl;
		return false;
	}

	if ( shapes.empty() )
	{
		std::cerr << "No shapes in obj file (run 'git lfs fetch' and 'git lfs pull' in 'test/common/meshes/lfs')" << std::endl;
		return false;
	}

	meshes.clear();
	meshes.resize(shapes.size());

	auto compare = []( const tinyobj::index_t& a, const tinyobj::index_t& b ) {
		if ( a.vertex_index < b.vertex_index ) return true;
		if ( a.vertex_index > b.vertex_index ) return false;

		if ( a.normal_index < b.normal_index ) return true;
		if ( a.normal_index > b.normal_index ) return false;

		if ( a.texcoord_index < b.texcoord_index ) return true;
		if ( a.texcoord_index > b.texcoord_index ) return false;

		return false;
	};

	for ( size_t i = 0; i < shapes.size(); ++i )
	{
		TriangleMesh& current_mesh = meshes[i];

		auto& vertices = current_mesh.vertices;
		auto& normals = current_mesh.vertex_normals;
		
		std::vector<uint32_t>									  indices;
		float3*													  v = reinterpret_cast<float3*>( attrib.vertices.data() );
		std::map<tinyobj::index_t, uint32_t, decltype( compare )> knownIndex( compare );

		for ( size_t face = 0; face < shapes[i].mesh.num_face_vertices.size(); face++ )
		{
			tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * face + 0];
			tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * face + 1];
			tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * face + 2];

			if ( knownIndex.find( idx0 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx0] );
			}
			else
			{
				knownIndex[idx0] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx0] );
				vertices.push_back( v[idx0.vertex_index] );
				normals.push_back( v[idx0.normal_index] );
			}

			if ( knownIndex.find( idx1 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx1] );
			}
			else
			{
				knownIndex[idx1] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx1] );
				vertices.push_back( v[idx1.vertex_index] );
				normals.push_back( v[idx1.normal_index] );
			}

			if ( knownIndex.find( idx2 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx2] );
			}
			else
			{
				knownIndex[idx2] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx2] );
				vertices.push_back( v[idx2.vertex_index] );
				normals.push_back( v[idx2.normal_index] );
			}	
		}

		current_mesh.indices.resize(indices.size() / 3);
		std::memcpy(current_mesh.indices.data(), indices.data(), indices.size() * sizeof(uint32_t));

		//compute vertex normals
        current_mesh.triangle_normals.reserve(current_mesh.indices.size());
        for (const auto& ti : current_mesh.indices)
        {
            auto& n0 = current_mesh.vertex_normals[ti.x];
            auto& n1 = current_mesh.vertex_normals[ti.y];
            auto& n2 = current_mesh.vertex_normals[ti.z];

			float3 tn = (n0 + n1 + n2) / 3.f;
            tn = hiprt::normalize(tn);
            current_mesh.triangle_normals.push_back(tn);       
		}
	}
	return true;
}
