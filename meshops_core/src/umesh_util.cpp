//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <meshops_internal/umesh_util.hpp>
#include "micromesh/micromesh_utils.h"
#include "nvmath/nvmath.h"
#include "nvmath/nvmath_types.h"
#include <glm/matrix.hpp>
#include <micromesh/micromesh_format_types.h>
#include <nvh/nvprint.hpp>
#include <algorithm>
#include <cinttypes>
#include <glm/gtc/type_ptr.hpp>
#include <thread>
#include <bary/bary_core.h>

// Replace displacement mapped meshes that are just a single quad with a tessellated plane, matching the tessellation of
// the heightmap. Returns true if any tessellation was done.
bool tessellateQuads(int targetSubdivisionDiff, const std::vector<nvmath::vec2ui>& meshHeightmapSizes, meshops::DynamicMeshSetView& meshSet)
{
  std::vector<meshops::MeshSlice> newSlices;
  std::vector<size_t>             tessellateSlices;
  std::vector<nvmath::vec2ui>     sliceTessellation;
  std::vector<nvmath::vec4ui>     quadVertices;
  for(size_t meshIdx = 0; meshIdx < meshSet.slices.size(); ++meshIdx)
  {
    meshops::MeshView mesh          = meshSet.slice(meshIdx);
    nvmath::vec2ui    heightmapSize = meshHeightmapSizes[meshIdx];
    newSlices.push_back(meshSet.slices[meshIdx]);

    // Don't bother if there is not heightmap
    if(heightmapSize.x == 0 || heightmapSize.y == 0)
      continue;

    // Check if topology could form a quad
    if(mesh.triangleCount() != 2)
      continue;

    // Check the 4 vertex positions all lie on a plane
    nvmath::vec3f tri1Normal =
        nvmath::cross(mesh.vertexPositions[1] - mesh.vertexPositions[0], mesh.vertexPositions[2] - mesh.vertexPositions[0]);
    float relDistToPlane = nvmath::dot(mesh.vertexPositions[3] - mesh.vertexPositions[0], tri1Normal);
    if(std::fabs(relDistToPlane) > 1e-6)
      continue;

    // Expect meshes with 4 vertices. If 5 or 6, they may be split due to differing UVs, in which case we couldn't
    // tessellate anyway.
    if(mesh.vertexCount() != 4)
    {
      LOGI("Note: mesh %zu has two triangles and is on a plane but has %" PRIu64
           " vertices. Require 4 for quad pre-tessellation.\n",
           meshIdx, mesh.vertexCount());
      continue;
    }

    // Find which vertices are shared for both triangles
    bool tri0SharedVert[3]{};
    bool tri1SharedVert[3]{};
    int  sharedVertCount = 0;
    for(uint32_t i = 0; i < 3; ++i)
    {
      for(uint32_t j = 0; j < 3; ++j)
      {
        if(mesh.triangleVertices[0][i] == mesh.triangleVertices[1][j])
        {
          tri0SharedVert[i] = true;
          tri1SharedVert[j] = true;
          ++sharedVertCount;
        }
      }
    }
    if(sharedVertCount != 2)
    {
      LOGW("Warning: almost-quad mesh %zu has %i shared vertices. Expected 2.\n", meshIdx, sharedVertCount);
      continue;
    }

    // Form a quad from the vertices of...
    // - the non-shared vertex of triangle zero
    // - the next vertex of triangle zero
    // - the non-shared vertex of triangle one
    // - the remaining vertex of triangle zero
    nvmath::vec4ui quadIdx;
    for(uint32_t i = 0; i < 3; ++i)
    {
      if(!tri0SharedVert[i])
      {
        quadIdx[0] = mesh.triangleVertices[0][(i + 0) % 3];
        quadIdx[1] = mesh.triangleVertices[0][(i + 1) % 3];
        quadIdx[3] = mesh.triangleVertices[0][(i + 2) % 3];
      }
      if(!tri1SharedVert[i])
      {
        quadIdx[2] = mesh.triangleVertices[1][(i + 0) % 3];
      }
    }
    assert(quadIdx[0] < 4 && quadIdx[1] < 4 && quadIdx[2] < 4 && quadIdx[3] < 4);

    // Compute desired tessellation
    const nvmath::vec2f quadUVs[]    = {mesh.vertexTexcoords0[quadIdx[0]], mesh.vertexTexcoords0[quadIdx[1]],
                                     mesh.vertexTexcoords0[quadIdx[2]], mesh.vertexTexcoords0[quadIdx[3]]};
    float               edgePixels[] = {nvmath::length((quadUVs[1] - quadUVs[0]) * nvmath::vec2f(heightmapSize)),
                          nvmath::length((quadUVs[2] - quadUVs[1]) * nvmath::vec2f(heightmapSize)),
                          nvmath::length((quadUVs[3] - quadUVs[2]) * nvmath::vec2f(heightmapSize)),
                          nvmath::length((quadUVs[0] - quadUVs[3]) * nvmath::vec2f(heightmapSize))};

    float    targetTessDiff = static_cast<float>(1 << std::max(0, targetSubdivisionDiff));
    uint32_t tessellationU  = static_cast<uint32_t>(std::ceil(std::max(edgePixels[0], edgePixels[2]) / targetTessDiff));
    uint32_t tessellationV  = static_cast<uint32_t>(std::ceil(std::max(edgePixels[1], edgePixels[3]) / targetTessDiff));

    // Skip this quad if we won't actually tessellate it
    if(tessellationU < 2 && tessellationV < 2)
      continue;

    // Compute the space needed to tessellate the quad
    meshops::MeshSlice& slice = newSlices.back();
    slice.vertexCount         = (tessellationU + 1) * (tessellationV + 1);
    slice.triangleCount       = (tessellationU) * (tessellationV)*2;
    sliceTessellation.push_back({tessellationU, tessellationV});
    tessellateSlices.push_back(meshIdx);
    quadVertices.push_back(quadIdx);
  }

  if(tessellateSlices.empty())
    return false;

  // Compute offsets for new meshes
  for(size_t meshIdx = 1; meshIdx < meshSet.slices.size(); ++meshIdx)
  {
    newSlices[meshIdx].triangleOffset = newSlices[meshIdx - 1].triangleOffset + newSlices[meshIdx - 1].triangleCount;
    newSlices[meshIdx].vertexOffset   = newSlices[meshIdx - 1].vertexOffset + newSlices[meshIdx - 1].vertexCount;
  }

  meshSet.flat.triangleVertices.resize(newSlices.back().triangleOffset + newSlices.back().triangleCount, {~0, ~0, ~0});

  // Resize to fit the tessellated meshes
  meshSet.flat.resize_nonempty(newSlices.back().triangleOffset + newSlices.back().triangleCount,
                               newSlices.back().vertexOffset + newSlices.back().vertexCount);

  // Move the old data in-place. This is safe because meshes only increase in size and copy_backward is used.
  for(size_t meshIdx = 0; meshIdx < meshSet.slices.size(); ++meshIdx)
  {
    size_t                    meshIdsRev = meshSet.slices.size() - 1 - meshIdx;
    const meshops::MeshSlice& fromSlice  = meshSet.slices[meshIdsRev];
    const meshops::MeshSlice& toSlice    = newSlices[meshIdsRev];
    if(fromSlice != toSlice)
      meshSet.flat.slice(toSlice).copy_backward_from(meshSet.flat.slice(fromSlice));
  }
  meshSet.slices = std::move(newSlices);

  // Generate tessellated quad meshes
  for(size_t jobIdx = 0; jobIdx < tessellateSlices.size(); ++jobIdx)
  {
    auto&                    meshIdx      = tessellateSlices[jobIdx];
    nvmath::vec2ui           tessellation = sliceTessellation[jobIdx];
    nvmath::vec4ui           quadIdx      = quadVertices[jobIdx];
    meshops::MutableMeshView mesh         = meshSet.slice(meshIdx);

    // Create a temporary copy of the original quad data, which now appears at the end of the slice due to the resize
    // and re-packing done above.
    meshops::MeshData        quadMesh;
    meshops::DynamicMeshView quadMeshView(quadMesh);
    quadMeshView.resize_nonempty(2, 4, mesh);
    quadMeshView.copy_from(mesh.slice(
        meshops::MeshSlice{meshSet.slices[meshIdx].triangleCount - 2, 2, meshSet.slices[meshIdx].vertexCount - 4, 4}));

    // Generate mesh vertices
    auto interpAttr = [&](auto quadAttr, auto outAttr) {
      if(outAttr.empty())
        return;
      for(uint32_t y = 0; y < tessellation.y + 1; ++y)
      {
        for(uint32_t x = 0; x < tessellation.x + 1; ++x)
        {
          nvmath::vec2f coord{x, y};
          nvmath::vec2f fcoord = coord / nvmath::vec2f(tessellation);
          auto          u0     = quadAttr[quadIdx[0]] * (1.0F - fcoord.x) + quadAttr[quadIdx[1]] * fcoord.x;
          auto          u1     = quadAttr[quadIdx[3]] * (1.0F - fcoord.x) + quadAttr[quadIdx[2]] * fcoord.x;
          auto          v      = u0 * (1.0F - fcoord.y) + u1 * fcoord.y;
          outAttr[y * (tessellation.x + 1) + x] = v;
        }
      }
    };
    interpAttr(quadMesh.vertexPositions, mesh.vertexPositions);
    interpAttr(quadMesh.vertexNormals, mesh.vertexNormals);
    interpAttr(quadMesh.vertexTexcoords0, mesh.vertexTexcoords0);
    interpAttr(quadMesh.vertexTangents, mesh.vertexTangents);
    interpAttr(quadMesh.vertexDirections, mesh.vertexDirections);
    interpAttr(quadMesh.vertexDirectionBounds, mesh.vertexDirectionBounds);

    // Generate mesh triangle indices
    for(uint32_t y = 0; y < tessellation.y; ++y)
    {
      for(uint32_t x = 0; x < tessellation.x; ++x)
      {
        uint32_t       vertIdx = y * (tessellation.x + 1) + x;
        const uint32_t u       = 1;
        const uint32_t v       = tessellation.x + 1;
        auto off = mesh.triangleVertices.begin() + ((y * tessellation.x + x) * 2 + 1) - meshSet.flat.triangleVertices.begin();
        assert(static_cast<size_t>(off) < meshSet.flat.triangleVertices.size());
        mesh.triangleVertices[(y * tessellation.x + x) * 2 + 0] = {vertIdx, vertIdx + u, vertIdx + u + v};
        mesh.triangleVertices[(y * tessellation.x + x) * 2 + 1] = {vertIdx, vertIdx + u + v, vertIdx + v};
      }
    }
  }

  return true;
}

void makeDisplacementDirections(const meshops::MeshView&          meshView,
                                const micromesh::MeshTopology&    topology,
                                meshops::ArrayView<nvmath::vec3f> outDisplacementDirections,
                                NormalReduceOp                    normalReduceOp)
{
  micromesh::MeshTopologyUtil topoUtil(topology);
  std::vector<float>          averageWeight(outDisplacementDirections.size(), 0);

  assert(!meshView.vertexNormals.empty());

  // Average normals into watertight vertices
  meshops::ArrayView<const micromesh::Vector_uint32_3> triangleIndices(meshView.triangleVertices);
  for(uint32_t triIdx = 0; triIdx < triangleIndices.size(); ++triIdx)
  {
    micromesh::Vector_uint32_3 tri   = triangleIndices[triIdx];
    micromesh::Vector_uint32_3 triWt = topoUtil.getTriangleVertices(triIdx);

    // Skip degenerate triangles
    if(micromesh::meshIsTriangleDegenerate(triWt))
      continue;

    // For each vertex
    for(uint32_t vertIdx = 0; vertIdx < 3; ++vertIdx)
    {
      uint32_t vert   = (&tri.x)[vertIdx];
      uint32_t vertWt = (&triWt.x)[vertIdx];

      // Compute the angle between the vertex's adjacent edges for a weight to the smoothed normal.
      int                        weightEdgeVectorsCount = 0;
      nvmath::vec3f              weightEdgeVectors[2];
      micromesh::Vector_uint32_3 edges = topoUtil.getTriangleEdges(triIdx);
      for(uint32_t edgeIdx = 0; edgeIdx < 3; ++edgeIdx)
      {
        uint32_t                   edge      = (&edges.x)[edgeIdx];
        micromesh::Vector_uint32_2 edgeVerts = topoUtil.getEdgeVertices(edge);
        if(edgeVerts.x == vertWt)
          weightEdgeVectors[weightEdgeVectorsCount++] =
              meshView.vertexPositions[vertWt] - meshView.vertexPositions[edgeVerts.y];
        else if(edgeVerts.y == vertWt)
          weightEdgeVectors[weightEdgeVectorsCount++] =
              meshView.vertexPositions[vertWt] - meshView.vertexPositions[edgeVerts.x];
      }
      assert(weightEdgeVectorsCount == 2);
      float edgeDot = nvmath::dot(nvmath::normalize(weightEdgeVectors[0]), nvmath::normalize(weightEdgeVectors[1]));


      // Floating-point math may provide values slightly > 1 or < 1 (e.g. 1.00000012)
      edgeDot = std::min(1.f, std::max(-1.f, edgeDot));

      // Prevent weight == 0, that will create NaN displacement vector coordinates
      float weight = std::max(FLT_MIN, acosf(edgeDot));
      // Take direction vectors directly from object space normals. Note that we still bake in world space, so if a tool
      // applies a world space transform to the high res mesh before baking, any heightmap scale/bias would effectively
      // jump to being applied in world space, whereas the highres geometry used for baking would not change.
      const nvmath::vec3f& normal = meshView.vertexNormals[vert];

      if(normalReduceOp == eNormalReduceTangent)
      {
        // To make a sharp edge, intersect the two normals as though they define planes at their length.
        // Uses the intersection point of 3 planes, where the third is perpendicular to the first two at zero distance.
        float         runningLength = nvmath::length(outDisplacementDirections[vertWt]);
        nvmath::vec3f n1            = outDisplacementDirections[vertWt] / runningLength;
        nvmath::vec3f n2            = normal;
        nvmath::vec3f n3            = nvmath::normalize(nvmath::cross(n1, n2));
        float         det           = glm::determinant(glm::mat3(n1, n2, n3));
        outDisplacementDirections[vertWt] =
            runningLength < 1e-6 || det < 1e-6 ? normal : (runningLength * nvmath::cross(n2, n3) + nvmath::cross(n3, n1)) / det;
      }
      else
      {
        // Accumulate a linear average
        outDisplacementDirections[vertWt] =
            outDisplacementDirections[vertWt] * (averageWeight[vertWt] / (averageWeight[vertWt] + weight))
            + normal * (weight / (averageWeight[vertWt] + weight));
        averageWeight[vertWt] += weight;
      }
    }
  }

  // Copy back average displacements for duplicate vertices, just in case something uses the original scene indices.
  for(uint32_t triIdx = 0; triIdx < triangleIndices.size(); ++triIdx)
  {
    micromesh::Vector_uint32_3 tri   = triangleIndices[triIdx];
    micromesh::Vector_uint32_3 triWt = topoUtil.getTriangleVertices(triIdx);

    // Skip degenerate triangles
    if(micromesh::meshIsTriangleDegenerate(triWt))
      continue;

    for(uint32_t vertIdx = 0; vertIdx < 3; ++vertIdx)
    {
      uint32_t vert   = (&tri.x)[vertIdx];
      uint32_t vertWt = (&triWt.x)[vertIdx];
      if(vert != vertWt)
        outDisplacementDirections[vert] = outDisplacementDirections[vertWt];
    }
  }

  // Optionally normalize the interpolated normal to expand corners of welded seams
  if(normalReduceOp == eNormalReduceNormalizedLinear)
  {
    for(auto& direction : outDisplacementDirections)
    {
      direction = nvmath::normalize(direction);
    }
  }
}

micromesh::Result tessellateMesh(micromesh::OpContext      context,
                                 const meshops::MeshView&  meshView,
                                 uint32_t                  maxSubdivLevel,
                                 meshops::DynamicMeshView& outMesh)
{
  auto generateTessellatedVertex = [&](const micromesh::VertexGenerateInfo* vertexInfo, micromesh::VertexDedup dedupState,
                                       uint32_t threadIndex, void* beginResult) -> uint32_t {
    nvmath::vec3f  baryCoord(vertexInfo->vertexWUVfloat.w, vertexInfo->vertexWUVfloat.u, vertexInfo->vertexWUVfloat.v);
    nvmath::vec3ui triVertices = meshView.triangleVertices[vertexInfo->meshTriangleIndex];
    stabilizeTriangleVerticesOrder(triVertices, baryCoord);

    nvmath::vec3f vertexPosition;
    nvmath::vec3f vertexNormal;
    nvmath::vec2f vertexTexcoord0;
    nvmath::vec4f vertexTangent;
    nvmath::vec3f vertexBitangent;
    nvmath::vec3f vertexDirection;
    nvmath::vec2f vertexDirectionBound;

    auto interpAttrib = [&dedupState, &triVertices, &baryCoord](const auto& attribs, auto& attrib) {
      if(!attribs.empty())
      {
        attrib = baryInterp(attribs, triVertices, baryCoord);
        if(dedupState)
          micromeshVertexDedupAppendAttribute(dedupState, sizeof(attrib), &attrib);
      }
    };

    interpAttrib(meshView.vertexPositions, vertexPosition);
    interpAttrib(meshView.vertexNormals, vertexNormal);
    interpAttrib(meshView.vertexTexcoords0, vertexTexcoord0);
    interpAttrib(meshView.vertexTangents, vertexTangent);
    interpAttrib(meshView.vertexDirections, vertexDirection);
    interpAttrib(meshView.vertexDirectionBounds, vertexDirectionBound);

    uint32_t index = dedupState ? micromeshVertexDedupGetIndex(dedupState) : vertexInfo->nonDedupIndex;

    if(!meshView.vertexPositions.empty())
      outMesh.vertexPositions[index] = vertexPosition;
    if(!meshView.vertexNormals.empty())
      outMesh.vertexNormals[index] = vertexNormal;
    if(!meshView.vertexTexcoords0.empty())
      outMesh.vertexTexcoords0[index] = vertexTexcoord0;
    if(!meshView.vertexTangents.empty())
      outMesh.vertexTangents[index] = vertexTangent;
    if(!meshView.vertexDirections.empty())
      outMesh.vertexDirections[index] = vertexDirection;
    if(!meshView.vertexDirectionBounds.empty())
      outMesh.vertexDirectionBounds[index] = vertexDirectionBound;

    return index;
  };

  // Wrapper to extract the above lambda's captured data form userData and call it.
  auto generateTessellatedVertexWrapper = [](const micromesh::VertexGenerateInfo* vertexInfo, micromesh::VertexDedup dedupState,
                                             uint32_t threadIndex, void* beginResult, void* userData) {
    auto f = reinterpret_cast<decltype(generateTessellatedVertex)*>(userData);
    return (*f)(vertexInfo, dedupState, threadIndex, beginResult);
  };

  micromesh::OpTessellateMesh_input input;
  input.useVertexDeduplication = true;
  input.maxSubdivLevel         = maxSubdivLevel;
  input.userData               = &generateTessellatedVertex;
  input.pfnGenerateVertex      = generateTessellatedVertexWrapper;
  micromesh::arraySetDataVec(input.meshTrianglePrimitiveFlags, meshView.trianglePrimitiveFlags);
  micromesh::arraySetDataVec(input.meshTriangleSubdivLevels, meshView.triangleSubdivisionLevels);

  micromesh::OpTessellateMesh_output output;
  micromesh::Result                  result = micromesh::micromeshOpTessellateMeshBegin(context, &input, &output);
  if(result != micromesh::Result::eSuccess)
    return result;

  // Resize output mesh for worst case tesselltaion
  outMesh.resize_nonempty(output.meshTriangleVertices.count, output.vertexCount, meshView);
  output.meshTriangleVertices.data = outMesh.triangleVertices.data();

  // Generate vertices
  result = micromesh::micromeshOpTessellateMeshEnd(context, &input, &output);
  if(result != micromesh::Result::eSuccess)
    return result;

  // shrink vertex buffers due to dedup
  outMesh.resize_nonempty(output.meshTriangleVertices.count, output.vertexCount, meshView);
  assert(output.meshTriangleVertices.count && output.vertexCount);
  return result;
}


micromesh::Result generateDisplacementLines(micromesh::OpContext        context,
                                            const meshops::MeshView&    meshView,
                                            const bary::BasicView&      basic,
                                            const bary::Group&          baryGroup,
                                            std::vector<uint32_t>&      indices,
                                            std::vector<nvmath::vec3f>& positions,
                                            const float*                displacements)
{
  assert(meshView.triangleCount() == baryGroup.triangleCount);

  auto generateTessellatedVertex = [&](const micromesh::VertexGenerateInfo* vertexInfo, micromesh::VertexDedup dedupState,
                                       uint32_t threadIndex, void* beginResult) -> uint32_t {
    nvmath::vec3f  baryCoord(vertexInfo->vertexWUVfloat.w, vertexInfo->vertexWUVfloat.u, vertexInfo->vertexWUVfloat.v);
    nvmath::vec3ui triVertices = meshView.triangleVertices[vertexInfo->meshTriangleIndex];
    stabilizeTriangleVerticesOrder(triVertices, baryCoord);

    nvmath::vec3f triPos[3];
    nvmath::vec3f triDir[3];
    for(uint32_t v = 0; v < 3; v++)
    {
      nvmath::vec3f vpos    = meshView.vertexPositions[triVertices[v]];
      nvmath::vec3f vdir    = meshView.vertexDirections[triVertices[v]];
      nvmath::vec2f vbounds = meshView.vertexDirectionBounds[triVertices[v]];
      triPos[v]             = vpos + vbounds.x * vdir;
      triDir[v]             = vbounds.y * vdir;
    }
    nvmath::vec3f pos = baryInterp(triPos[0], triPos[1], triPos[2], baryCoord);
    nvmath::vec3f dir = baryInterp(triDir[0], triDir[1], triDir[2], baryCoord);

    const bary::Triangle& baryTri = basic.triangles[baryGroup.triangleFirst + vertexInfo->meshTriangleIndex];
    uint32_t              displacementIdx =
        bary::baryValueLayoutGetIndex(basic.valuesInfo->valueLayout, bary::ValueFrequency::ePerVertex,
                                      vertexInfo->vertexUV.u, vertexInfo->vertexUV.v, 0, vertexInfo->subdivLevel);
    float displacement = displacements[baryGroup.valueFirst + baryTri.valuesOffset + displacementIdx];

    // TODO: use micromeshQuantizedToFloatValues in a pfnBeginTriangle call instead, like in meshops_tessellation.cpp
    displacement = displacement * baryGroup.floatScale.r + baryGroup.floatBias.r;

    nvmath::vec3f posDisp = pos + dir * displacement;

    uint32_t index = vertexInfo->nonDedupIndex;
    if(dedupState)
    {
      micromeshVertexDedupAppendAttribute(dedupState, sizeof(pos), &pos);
      micromeshVertexDedupAppendAttribute(dedupState, sizeof(posDisp), &posDisp);
      index = micromeshVertexDedupGetIndex(dedupState);
    }
    positions[index * 2 + 0] = pos;
    positions[index * 2 + 1] = posDisp;
    indices[index * 2 + 0]   = index * 2 + 0;
    indices[index * 2 + 1]   = index * 2 + 1;
    return index;
  };

  // Wrapper to extract the above lambda's captured data form userData and call it.
  auto generateTessellatedVertexWrapper = [](const micromesh::VertexGenerateInfo* vertexInfo, micromesh::VertexDedup dedupState,
                                             uint32_t threadIndex, void* beginResult, void* userData) {
    auto f = reinterpret_cast<decltype(generateTessellatedVertex)*>(userData);
    return (*f)(vertexInfo, dedupState, threadIndex, beginResult);
  };

  micromesh::OpTessellateMesh_input input;
  input.useVertexDeduplication = true;
  input.maxSubdivLevel         = baryGroup.maxSubdivLevel;
  input.userData               = &generateTessellatedVertex;
  input.pfnGenerateVertex      = generateTessellatedVertexWrapper;
  micromesh::arraySetDataVec(input.meshTrianglePrimitiveFlags, meshView.trianglePrimitiveFlags);
  input.meshTriangleSubdivLevels = micromesh::ArrayInfo_uint16(&basic.triangles[baryGroup.triangleFirst].subdivLevel,
                                                               baryGroup.triangleCount, sizeof(bary::Triangle));

  micromesh::OpTessellateMesh_output output;
  micromesh::Result                  result = micromesh::micromeshOpTessellateMeshBegin(context, &input, &output);
  if(result != micromesh::Result::eSuccess)
    return result;

  // Resize line indices and positions for a start and end point for every vertex
  indices.resize(output.vertexCount * 2);
  positions.resize(output.vertexCount * 2);

  // Create a dummy triangle indices array, even though it won't be used
  std::vector<nvmath::vec3ui> triangleIndices(output.meshTriangleVertices.count);
  output.meshTriangleVertices.data = triangleIndices.data();

  // Generate vertices
  result = micromesh::micromeshOpTessellateMeshEnd(context, &input, &output);
  if(result != micromesh::Result::eSuccess)
    return result;

  // shrink vertex buffers due to dedup
  indices.resize(output.vertexCount * 2);
  positions.resize(output.vertexCount * 2);
  return result;
}

meshops::MeshAttributeFlags generationRequiresTopology(meshops::MeshAttributeFlags existing, meshops::MeshAttributeFlags required)
{
  constexpr meshops::MeshAttributeFlags requireTopology = meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit
                                                          | meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit
                                                          | meshops::MeshAttributeFlagBits::eMeshAttributeVertexNormalBit
                                                          | meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit;
  const meshops::MeshAttributeFlags meshAttrNew = (~existing) & required;
  return meshAttrNew & requireTopology;
}

micromesh::Result generateMeshAttributes(meshops::Context                           context,
                                         meshops::MeshAttributeFlags                meshAttrFlags,
                                         meshops::OpGenerateSubdivisionLevel_input* subdivisionLevelSettings,
                                         const micromesh::MeshTopology*             topology,
                                         meshops::ResizableMeshView&                meshView,
                                         uint32_t&                                  maxSubdivLevel,
                                         NormalReduceOp                             directionsGenOp,
                                         meshops::TangentSpaceAlgorithm             tangentAlgorithm)
{
  // Keep these in sync with below code to generate them.
  constexpr meshops::MeshAttributeFlags meshAttrCanGenerate =
      meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit
      | meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit
      | meshops::MeshAttributeFlagBits::eMeshAttributeVertexNormalBit | meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit
      | meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit
      | meshops::MeshAttributeFlagBits::eMeshAttributeVertexTangentBit;

  // Direction generation currently requires normals
  if(!!(meshAttrFlags & meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit))
  {
    meshAttrFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeVertexNormalBit;
  }

  const meshops::MeshAttributeFlags meshAttrNew            = (~meshView.getMeshAttributeFlags()) & meshAttrFlags;
  const meshops::MeshAttributeFlags meshAttrCannotGenerate = (~meshAttrCanGenerate) & meshAttrNew;

  if(meshAttrCannotGenerate)
  {
    LOGE("Error: Request to generate missing mesh attributes %s but generating %s is not implemented\n",
         meshops::meshAttribBitsString(meshAttrNew).c_str(), meshops::meshAttribBitsString(meshAttrCannotGenerate).c_str());
    return micromesh::Result::eFailure;
  }

  LOGI("Generating mesh attributes %s\n", meshops::meshAttribBitsString(meshAttrNew).c_str());

  meshView.resize(meshAttrNew, meshView.triangleCount(), meshView.vertexCount());

  // Subdivision levels
  if((meshAttrNew & meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit) != 0)
  {
    // Generate target levels
    {
      meshops::OpGenerateSubdivisionLevel_input input;
      if(subdivisionLevelSettings)
        input = *subdivisionLevelSettings;
      meshops::OpGenerateSubdivisionLevel_modified modifieds{meshView};
      micromesh::Result result = meshops::meshopsOpGenerateSubdivisionLevel(context, 1, &input, &modifieds);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Error: Failed to generate subdivision levels\n");
        return result;
      }

      // Record the maximum value generated
      maxSubdivLevel = modifieds.maxSubdivLevel;
    }

    // Sanitize levels, such that neighboring triangles differ by at most one level.
    {
      assert(topology->triangleVertices.count != 0);
      meshops::OpSanitizeSubdivisionLevel_input input;
      // Input the real maximum in the array, not the maximum possible
      input.maxSubdivLevel = maxSubdivLevel;
      input.meshTopology   = topology;
      meshops::OpSanitizeSubdivisionLevel_modified modifieds{meshView};
      micromesh::Result result = meshops::meshopsOpSanitizeSubdivisionLevel(context, 1, &input, &modifieds);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Error: Failed to sanitize subdivision levels\n");
        return result;
      }
    }
  }

  // Edge flags
  if((meshAttrNew & meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit) != 0)
  {
    assert(topology->triangleVertices.count != 0);
    meshops::OpBuildPrimitiveFlags_input input;
    input.meshTopology = topology;
    meshops::OpBuildPrimitiveFlags_modified modifieds{meshView};
    micromesh::Result result = meshops::meshopsOpBuildPrimitiveFlags(context, 1, &input, &modifieds);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Error: Failed to generate edge flags\n");
      return result;
    }
  }

  // Normal vectors
  if((meshAttrNew & meshops::MeshAttributeFlagBits::eMeshAttributeVertexNormalBit) != 0)
  {
    assert(topology->triangleVertices.count != 0);
    // Currently using direction vector generation code as it's based on positions anyway
    // This assumes smooth normals across the whole mesh!
    meshops::OpGenerateVertexDirections_input input;
    input.triangleUniqueVertexIndices =
        meshops::ArrayView(reinterpret_cast<const micromesh::Vector_uint32_3*>(topology->triangleVertices.data),
                           topology->triangleVertices.count, topology->triangleVertices.byteStride);
    meshops::OpGenerateVertexDirections_modified modifieds{meshView};
    modifieds.targetAttribute = meshops::eMeshAttributeVertexNormalBit;
    micromesh::Result result  = meshops::meshopsOpGenerateVertexDirections(context, 1, &input, &modifieds);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Error: Failed to generate vertex normals\n");
      return result;
    }
  }

  // Direction vectors
  if((meshAttrNew & meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit) != 0)
  {
    assert(topology->triangleVertices.count != 0);
#if 0
      meshops::OpGenerateVertexDirections_input    input;
      input.triangleUniqueVertexIndices =
        meshops::ArrayView(reinterpret_cast<const micromesh::Vector_uint32_3*>(topology->triangleVertices.data),
          topology->triangleVertices.count, topology->triangleVertices.byteStride);
      meshops::OpGenerateVertexDirections_modified modifieds{meshView};
      modifieds.targetAttribute = meshops::eMeshAttributeVertexDirectionBit;
      micromesh::Result result = meshops::meshopsOpGenerateVertexDirections(context, 1, &input, &modifieds);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Error: Failed to generate vertex directions\n");
        return result;
      }
#else
    // FIXME why not use above?
    makeDisplacementDirections(meshView, *topology, meshView.vertexDirections, directionsGenOp);
#endif
  }

  // Direction bounds
  if((meshAttrNew & meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit) != 0)
  {
    std::fill(meshView.vertexDirectionBounds.begin(), meshView.vertexDirectionBounds.end(), nvmath::vec2f(0.0f, 1.0f));
  }

  // Tangent space
  if((meshAttrNew & meshops::MeshAttributeFlagBits::eMeshAttributeVertexTangentBit) != 0)
  {
    meshops::OpGenerateVertexTangentSpace_input    input{tangentAlgorithm};
    meshops::OpGenerateVertexTangentSpace_modified modifieds{meshView};
    micromesh::Result result = meshops::meshopsOpGenerateVertexTangentSpace(context, 1, &input, &modifieds);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Error: Failed to generate vertex tangents\n");
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}
