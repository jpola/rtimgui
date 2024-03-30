#pragma once


#include <filesystem>
#include "TriangleMesh.h"

namespace fs = std::filesystem;

bool ReadStlMesh(const fs::path& path, TriangleMesh& mesh);

bool ReadObjMesh(const fs::path& path, const fs::path& mtlBaseDir, std::vector<TriangleMesh>& meshes);