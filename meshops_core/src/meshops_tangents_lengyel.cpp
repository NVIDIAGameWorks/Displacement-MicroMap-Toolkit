//
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//

#include "meshops_tangents_lengyel.hpp"

namespace meshops {

void createLengyelTangents(meshops::MutableMeshView& mesh)
{
  // This is more or less a copy of GltfScene::createTangents from nvpro_core.
  // Maybe we could expose createTangents in nvpro_core and use it directly?

  std::vector<nvmath::vec3f> tangent(mesh.vertexCount());
  std::vector<nvmath::vec3f> bitangent(mesh.vertexCount());

  // Current implementation
  // http://foundationsofgameenginedev.com/FGED2-sample.pdf
  for(size_t tri = 0; tri < mesh.triangleCount(); tri++)
  {
    // local index
    const nvmath::vec3ui indices = mesh.triangleVertices[tri];
    assert(indices[0] < mesh.vertexCount());
    assert(indices[1] < mesh.vertexCount());
    assert(indices[2] < mesh.vertexCount());

    const auto& p0 = mesh.vertexPositions[indices[0]];
    const auto& p1 = mesh.vertexPositions[indices[1]];
    const auto& p2 = mesh.vertexPositions[indices[2]];

    const auto& uv0 = mesh.vertexTexcoords0[indices[0]];
    const auto& uv1 = mesh.vertexTexcoords0[indices[1]];
    const auto& uv2 = mesh.vertexTexcoords0[indices[2]];

    nvmath::vec3f e1 = p1 - p0;
    nvmath::vec3f e2 = p2 - p0;

    nvmath::vec2f duvE1 = uv1 - uv0;
    nvmath::vec2f duvE2 = uv2 - uv0;

    float r = 1.0F;
    float a = duvE1.x * duvE2.y - duvE2.x * duvE1.y;
    if(fabs(a) > 0)  // Catch degenerated UV
    {
      r = 1.0f / a;
    }

    nvmath::vec3f t = (e1 * duvE2.y - e2 * duvE1.y) * r;
    nvmath::vec3f b = (e2 * duvE1.x - e1 * duvE2.x) * r;

    tangent[indices[0]] += t;
    tangent[indices[1]] += t;
    tangent[indices[2]] += t;

    bitangent[indices[0]] += b;
    bitangent[indices[1]] += b;
    bitangent[indices[2]] += b;
  }

  for(uint32_t a = 0; a < mesh.vertexCount(); a++)
  {
    const auto& t = tangent[a];
    const auto& b = bitangent[a];
    const auto& n = mesh.vertexNormals[a];

    // Gram-Schmidt orthogonalize
    nvmath::vec3f otangent = nvmath::normalize(t - (nvmath::dot(n, t) * n));

    // In case the tangent is invalid
    if(otangent == nvmath::vec3f(0, 0, 0))
    {
      if(abs(n.x) > abs(n.y))
        otangent = nvmath::vec3f(n.z, 0, -n.x) / sqrtf(n.x * n.x + n.z * n.z);
      else
        otangent = nvmath::vec3f(0, -n.z, n.y) / sqrtf(n.y * n.y + n.z * n.z);
    }

    // Calculate handedness
    float handedness       = (nvmath::dot(nvmath::cross(n, t), b) < 0.0F) ? 1.0F : -1.0F;
    mesh.vertexTangents[a] = nvmath::vec4f(otangent, handedness);   
  }
}

}  // namespace meshops