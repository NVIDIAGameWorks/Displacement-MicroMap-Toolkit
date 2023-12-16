/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <meshops_internal/meshops_context.h>
#include <meshops_internal/octant_encoding.h>
#include <meshops_internal/meshops_texture.h>
#include <meshops/meshops_operations.h>
#include <microutils/microutils_compression.hpp>
#include <type_traits>
#include <meshops_internal/heightmap.hpp>
#include <meshops_internal/pn_triangles.hpp>

namespace meshops {

//////////////////////////////////////////////////////////////////////////

namespace {

template <class T>
inline T baryInterp(const T& a, const T& b, const T& c, const glm::vec3& baryCoord)
{
  return a * baryCoord.x + b * baryCoord.y + c * baryCoord.z;
}

template <typename Vector>
inline auto baryInterp(const Vector& attr, const nvmath::vec3ui& tri, const nvmath::vec3f& baryCoord)
{
  return baryInterp(attr[tri.x], attr[tri.y], attr[tri.z], baryCoord);
}

// Sort a triangle's vertex indices so that any rounding in baryInterp is consistent along tessellated edges
inline void stabilizeTriangleVerticesOrder(nvmath::vec3ui& triVertices, nvmath::vec3f& baryCoord)
{
  if(triVertices.y < triVertices.x)
  {
    std::swap(triVertices.y, triVertices.x);
    std::swap(baryCoord.y, baryCoord.x);
  }
  if(triVertices.z < triVertices.y)
  {
    std::swap(triVertices.z, triVertices.y);
    std::swap(baryCoord.z, baryCoord.y);
  }
  if(triVertices.y < triVertices.x)
  {
    std::swap(triVertices.y, triVertices.x);
    std::swap(baryCoord.y, baryCoord.x);
  }
}

using MicroVertexInfoVector = std::vector<micromesh::MicroVertexInfo>;

struct TessellateConfig
{
  uint32_t                       maxSubdivLevel             = 0;
  const bary::BasicView*         baryDisplacement           = nullptr;
  uint32_t                       baryDisplacementGroupIndex = 0;
  uint32_t                       baryDisplacementMapOffset  = 0;
  const bary::BasicView*         baryNormal                 = nullptr;
  uint32_t                       baryNormalGroupIndex       = 0;
  uint32_t                       baryNormalMapOffset        = 0;
  HeightMap*                     heightmapTexture           = nullptr;
  Heightmap                      heightmapDesc              = {};
  const micromesh::MeshTopology* topology                   = nullptr;
};

struct TessPayload
{
  Context&                            meshopsContext;
  MeshView                            inMeshView;
  ResizableMeshView&                  outMeshView;
  TessellateConfig                    config;
  uint32_t                            maxMicroVertices;
  micromesh::MessageCallbackInfo*     messageCallback;
  microutils::ThreadedTriangleDecoder threadedDecoder;
  std::vector<float>                  threadDistances;
  const micromesh::ArrayInfo_uint16*  triangleSubdivLevels = nullptr;
  std::vector<MicroVertexInfoVector>  threadSanitizeMicroVertices;
  bool                                tessellationError = false;
};

struct TessVertex
{
  nvmath::vec3f vertexPosition;
  nvmath::vec3f vertexNormal;
  nvmath::vec2f vertexTexcoord0;
  nvmath::vec4f vertexTangent;
  nvmath::vec3f vertexBitangent;
  nvmath::vec3f vertexDirection;
  nvmath::vec2f vertexDirectionBound;
};

void* tessBeginTriangleUncompressed(uint32_t meshTriangleIndex, uint32_t micromapTriangleIndex, uint32_t threadIndex, void* userData)
{
  TessPayload&          payload = *reinterpret_cast<TessPayload*>(userData);
  const bary::Group&    group   = payload.config.baryDisplacement->groups[payload.config.baryDisplacementGroupIndex];
  const bary::Triangle& tri =
      payload.config.baryDisplacement->triangles[group.triangleFirst + micromapTriangleIndex + payload.config.baryDisplacementMapOffset];

  // decode to uint11
  const void* triUncompressed =
      payload.config.baryDisplacement->values
      + size_t(payload.config.baryDisplacement->valuesInfo->valueByteSize) * (group.valueFirst + tri.valuesOffset);
  // from uint11 to expanded float
  float*   triFloats = payload.threadDistances.data() + payload.maxMicroVertices * threadIndex;
  uint32_t numValues = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, tri.subdivLevel);

  micromesh::ArrayInfo inputQuantized = {const_cast<void*>(triUncompressed), numValues,
                                         microutils::getMicromeshFormat(payload.config.baryDisplacement->valuesInfo->valueFormat),
                                         payload.config.baryDisplacement->valuesInfo->valueByteSize};
  micromesh::ArrayInfo outputFloat    = {triFloats, numValues, micromesh::Format::eR32_sfloat, sizeof(float)};
  micromesh::MicromapValueFloatExpansion inputExp;
  inputExp.bias[0]  = group.floatBias.r;
  inputExp.scale[0] = group.floatScale.r;
  micromesh::MicromapValueFloatExpansion outputExp;

  micromesh::FormatInfo inputFormatInfo;
  micromesh::Result     result = micromesh::micromeshFormatGetInfo(inputQuantized.format, &inputFormatInfo);
  if(result != micromesh::Result::eSuccess)
  {
    if(!payload.tessellationError)  // not synchronized, but printing an extra error per thread won't hurt
    {
      MESHOPS_LOGE(payload.meshopsContext, "micromesh::micromeshFormatGetInfo() returned %s", micromeshResultGetName(result));
    }
    payload.tessellationError = true;
    assert(false);
  }
  if(inputFormatInfo.isCompressedOrPacked)
  {
    result = micromesh::micromeshQuantizedPackedToFloatValues(false, &inputQuantized, &inputExp, &outputFloat,
                                                              &outputExp, payload.messageCallback);
  }
  else
  {
    result = micromesh::micromeshQuantizedToFloatValues(false, &inputQuantized, &inputExp, &outputFloat, &outputExp,
                                                        payload.messageCallback);
  }
  if(result != micromesh::Result::eSuccess)
  {
    if(!payload.tessellationError)  // not synchronized, but printing an extra error per thread won't hurt
    {
      MESHOPS_LOGE(payload.meshopsContext, "converting quantized to float values returned %s", micromeshResultGetName(result));
    }
    payload.tessellationError = true;
    assert(false);
  }

  return triFloats;
}

void* tessBeginTriangleCompressed(uint32_t meshTriangleIndex, uint32_t micromapTriangleIndex, uint32_t threadIndex, void* userData)
{
  TessPayload&          payload = *reinterpret_cast<TessPayload*>(userData);
  const bary::Group&    group   = payload.config.baryDisplacement->groups[payload.config.baryDisplacementGroupIndex];

  // decode to uint11
  uint32_t  numValues;
  uint16_t* triUncompressed = payload.threadedDecoder.tempThreadDecode(
      threadIndex, *payload.config.baryDisplacement, payload.config.baryDisplacementGroupIndex,
      group.triangleFirst + micromapTriangleIndex + payload.config.baryDisplacementMapOffset, numValues);
  // from uint11 to expanded float
  float* triFloats = payload.threadDistances.data() + payload.maxMicroVertices * threadIndex;

  micromesh::ArrayInfo inputQuantized = {triUncompressed, numValues, micromesh::Format::eR11_unorm_pack16, sizeof(uint16_t)};
  micromesh::ArrayInfo outputFloat = {triFloats, numValues, micromesh::Format::eR32_sfloat, sizeof(float)};
  micromesh::MicromapValueFloatExpansion inputExp;
  inputExp.bias[0]  = group.floatBias.r;
  inputExp.scale[0] = group.floatScale.r;
  micromesh::MicromapValueFloatExpansion outputExp;
  micromesh::Result result = micromesh::micromeshQuantizedToFloatValues(false, &inputQuantized, &inputExp, &outputFloat,
                                                                        &outputExp, payload.messageCallback);
  if(result != micromesh::Result::eSuccess)
  {
    if(!payload.tessellationError)  // not synchronized, but printing an extra error per thread won't hurt
    {
      MESHOPS_LOGE(payload.meshopsContext, "micromesh::micromeshQuantizedToFloatValues() returned %s",
                   micromeshResultGetName(result));
    }
    payload.tessellationError = true;
    assert(false);
  }

  return triFloats;
}

template <typename Tvec>
void interpAttrib(const Tvec& attribs, std::remove_cv_t<typename Tvec::value_type>& attrib, nvmath::vec3ui triVertices, nvmath::vec3f baryCoord)
{
  if(!attribs.empty())
  {
    attrib = baryInterp(attribs, triVertices, baryCoord);
  }
}

template <typename Tvec>
void hashAttrib(const Tvec& attribs, std::remove_cv_t<typename Tvec::value_type>& attrib, micromesh::VertexDedup dedupState)
{
  if(!attribs.empty() && dedupState)
  {
    micromeshVertexDedupAppendAttribute(dedupState, sizeof(attrib), &attrib);
  }
}

template <typename Tvec>
void writeAttrib(const Tvec& attribs, std::remove_cv_t<typename Tvec::value_type>& attrib, size_t index)
{
  if(!attribs.empty())
  {
    attribs[index] = attrib;
  }
}

template <bool DISPLACED, bool BARY_DISPLACEMENT>
static TessVertex makeVertex(const micromesh::VertexGenerateInfo* vertexInfo, uint32_t threadIndex, void* beginTriangleResult, void* userData)
{
  TessPayload&             payload  = *reinterpret_cast<TessPayload*>(userData);
  const MeshView&          meshView = payload.inMeshView;

  nvmath::vec3f  baryCoord(vertexInfo->vertexWUVfloat.w, vertexInfo->vertexWUVfloat.u, vertexInfo->vertexWUVfloat.v);
  nvmath::vec3ui triVertices = meshView.triangleVertices[vertexInfo->meshTriangleIndex];
  stabilizeTriangleVerticesOrder(triVertices, baryCoord);

  TessVertex result;

  interpAttrib(meshView.vertexTexcoords0, result.vertexTexcoord0, triVertices, baryCoord);
  interpAttrib(meshView.vertexTangents, result.vertexTangent, triVertices, baryCoord);
  interpAttrib(meshView.vertexDirections, result.vertexDirection, triVertices, baryCoord);
  interpAttrib(meshView.vertexDirectionBounds, result.vertexDirectionBound, triVertices, baryCoord);

  if(DISPLACED)
  {
    nvmath::vec3f dir;
    nvmath::vec3f pos;

    if(!BARY_DISPLACEMENT || meshView.vertexDirectionBounds.empty())
    {
      if(payload.config.heightmapDesc.pnTriangles)
      {
        const nvmath::vec3f& v0 = meshView.vertexPositions[triVertices.x];
        const nvmath::vec3f& v1 = meshView.vertexPositions[triVertices.y];
        const nvmath::vec3f& v2 = meshView.vertexPositions[triVertices.z];
        const nvmath::vec3f& n0 = meshView.vertexDirections[triVertices.x];
        const nvmath::vec3f& n1 = meshView.vertexDirections[triVertices.y];
        const nvmath::vec3f& n2 = meshView.vertexDirections[triVertices.z];
        PNTriangles          pnTriangle(v0, v1, v2, n0, n1, n2);

        // Heightmap tessellation is smoothed with PN triangle interpolation
        pos = pnTriangle.position(baryCoord);
        dir = pnTriangle.normal(baryCoord);
      }
      else
      {
        // Regular linear interpolation
        pos = baryInterp(meshView.vertexPositions, triVertices, baryCoord);
        dir = baryInterp(meshView.vertexDirections, triVertices, baryCoord);
      }
    }
    else
    {
      // Bary displacement has direction bounds, which must be applied before interpolation.
      nvmath::vec3f triPos[3];
      nvmath::vec3f triDir[3];

      for(uint32_t v = 0; v < 3; v++)
      {
        nvmath::vec3f vpos    = meshView.vertexPositions[triVertices[v]];
        nvmath::vec3f vdir    = meshView.vertexDirections[triVertices[v]];
        nvmath::vec2f vbounds = meshView.vertexDirectionBounds[triVertices[v]];

        triPos[v] = vpos + vbounds.x * vdir;
        triDir[v] = vbounds.y * vdir;
      }

      pos = baryInterp(triPos, {0, 1, 2}, baryCoord);
      dir = baryInterp(triDir, {0, 1, 2}, baryCoord);
    }

    float distance;
    if(BARY_DISPLACEMENT)
    {
      uint32_t distanceIndex = bary::baryValueLayoutGetIndex(payload.config.baryDisplacement->valuesInfo->valueLayout,
                                                             bary::ValueFrequency::ePerVertex, vertexInfo->vertexUV.u,
                                                             vertexInfo->vertexUV.v, 0, vertexInfo->subdivLevel);
      const float* triDistances = reinterpret_cast<float*>(beginTriangleResult);
      distance                  = triDistances[distanceIndex];
    }
    else
    {
      distance = payload.config.heightmapTexture->bilinearFetch(result.vertexTexcoord0) * payload.config.heightmapDesc.scale
                 + payload.config.heightmapDesc.bias;
      if(payload.config.heightmapDesc.normalizeDirections)
      {
        dir.normalize();
      }
    }

    result.vertexPosition = pos + dir * distance;

    if(payload.config.baryNormal)
    {
      const bary::Group&    group = payload.config.baryNormal->groups[payload.config.baryNormalGroupIndex];
      const bary::Triangle& triShading =
          payload.config.baryNormal->triangles[group.triangleFirst + vertexInfo->micromapTriangleIndex + payload.config.baryNormalMapOffset];

      uint32_t valueIndex = bary::baryValueLayoutGetIndex(payload.config.baryNormal->valuesInfo->valueLayout,
                                                          bary::ValueFrequency::ePerVertex, vertexInfo->vertexUV.u,
                                                          vertexInfo->vertexUV.v, 0, triShading.subdivLevel);

      result.vertexNormal = shaders::oct32_to_vec(reinterpret_cast<const uint32_t*>(
          payload.config.baryNormal->values)[group.valueFirst + triShading.valuesOffset + valueIndex]);
    }
    else
    {
      interpAttrib(meshView.vertexNormals, result.vertexNormal, triVertices, baryCoord);
    }
  }
  else
  {
    interpAttrib(meshView.vertexPositions, result.vertexPosition, triVertices, baryCoord);
    interpAttrib(meshView.vertexNormals, result.vertexNormal, triVertices, baryCoord);
  }

  return result;
}

template <bool DISPLACED, bool BARY_DISPLACEMENT>
static uint32_t tessPerVertex(const micromesh::VertexGenerateInfo* vertexInfo,
                              micromesh::VertexDedup               dedupState,
                              uint32_t                             threadIndex,
                              void*                                beginTriangleResult,
                              void*                                userData)
{
  TessPayload&             payload  = *reinterpret_cast<TessPayload*>(userData);
  const ResizableMeshView& outMesh  = payload.outMeshView;
  TessVertex vertex = makeVertex<DISPLACED, BARY_DISPLACEMENT>(vertexInfo, threadIndex, beginTriangleResult, userData);

  // Heightmap displacement may result in cracks (e.g. at duplicate vertex with
  // different normal/direction vector or different values across UV seams).
  // Take the average position for all vertices at the same position.
  if(DISPLACED && !BARY_DISPLACEMENT)
  {
    // Iterate over a list of all microvertices matching the current one. These
    // could be shared along base triangle edges or vertices.
    MicroVertexInfoVector      sanitizeMicroVertices = payload.threadSanitizeMicroVertices[threadIndex];
    micromesh::MicroVertexInfo queryVertex{vertexInfo->meshTriangleIndex, vertexInfo->vertexUV};
    uint32_t                   count =
        micromesh::micromeshMeshTopologyGetVertexSanitizationList(payload.config.topology, payload.triangleSubdivLevels, nullptr,
                                                                  queryVertex, (uint32_t)sanitizeMicroVertices.size(),
                                                                  sanitizeMicroVertices.data());
    for(uint32_t i = 0; i < count; ++i)
    {
      micromesh::MicroVertexInfo otherMicroVertex = sanitizeMicroVertices[i];
      if(memcmp(&queryVertex, &otherMicroVertex, sizeof(queryVertex)) == 0)
      {
        continue;
      }

      // Add the other micro vertex position and normal.
      micromesh::VertexGenerateInfo otherVertexInfo{};
      otherVertexInfo.meshTriangleIndex = otherMicroVertex.triangleIndex;
      otherVertexInfo.vertexUV          = otherMicroVertex.vertexUV;
      otherVertexInfo.subdivLevel = micromesh::arrayGetV<uint16_t>(*payload.triangleSubdivLevels, otherMicroVertex.triangleIndex);
      otherVertexInfo.vertexWUVfloat = micromesh::baryUVtoWUV_float(otherMicroVertex.vertexUV, otherVertexInfo.subdivLevel);
      TessVertex otherVertex = makeVertex<DISPLACED, BARY_DISPLACEMENT>(&otherVertexInfo, threadIndex, nullptr, userData);
      vertex.vertexPosition += otherVertex.vertexPosition;
    }

    // Divide by the total to get the average position
    if(count > 1)
    {
      vertex.vertexPosition /= static_cast<nvmath::nv_scalar>(count);
    }
  }

  hashAttrib(outMesh.vertexPositions, vertex.vertexPosition, dedupState);
  hashAttrib(outMesh.vertexNormals, vertex.vertexNormal, dedupState);
  hashAttrib(outMesh.vertexTexcoords0, vertex.vertexTexcoord0, dedupState);
  hashAttrib(outMesh.vertexTangents, vertex.vertexTangent, dedupState);
  hashAttrib(outMesh.vertexDirections, vertex.vertexDirection, dedupState);
  hashAttrib(outMesh.vertexDirectionBounds, vertex.vertexDirectionBound, dedupState);

  uint32_t index = dedupState ? micromeshVertexDedupGetIndex(dedupState) : vertexInfo->nonDedupIndex;

  writeAttrib(outMesh.vertexPositions, vertex.vertexPosition, index);
  writeAttrib(outMesh.vertexNormals, vertex.vertexNormal, index);
  writeAttrib(outMesh.vertexTexcoords0, vertex.vertexTexcoord0, index);
  writeAttrib(outMesh.vertexTangents, vertex.vertexTangent, index);
  writeAttrib(outMesh.vertexDirections, vertex.vertexDirection, index);
  writeAttrib(outMesh.vertexDirectionBounds, vertex.vertexDirectionBound, index);

  return index;
};

micromesh::Result tessellateMesh(Context context, const meshops::MeshView& meshView, meshops::ResizableMeshView& outMesh, TessellateConfig& config)
{
  TessPayload payload      = {context, meshView, outMesh, config};
  payload.maxMicroVertices = micromesh::subdivLevelGetVertexCount(config.maxSubdivLevel);

  MeshAttributeFlags attribFlags = meshView.getMeshAttributeFlags();

  // Remove several flags that are inputs only.
  attribFlags &= ~(MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit
                   | MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit);

  if(config.baryDisplacement || config.heightmapTexture)
  {
    attribFlags &= ~(MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit
                     | MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit);
  }

  micromesh::OpContext           micromeshContext = context->m_micromeshContext;
  micromesh::MessageCallbackInfo messageCallback  = micromesh::micromeshOpContextGetMessageCallback(micromeshContext);
  payload.messageCallback                         = &messageCallback;

  micromesh::OpTessellateMesh_input input;
  input.useVertexDeduplication = true;
  input.maxSubdivLevel         = config.maxSubdivLevel;
  input.userData               = &payload;

  if(config.baryDisplacement)
  {
    uint32_t numThreads       = micromesh::micromeshOpContextGetConfig(micromeshContext).threadCount;
    uint32_t maxMicroVertices = payload.maxMicroVertices;

    payload.threadDistances.resize(maxMicroVertices * numThreads);
    payload.threadedDecoder.init(bary::Format::eDispC1_r11_unorm_block,
                                 config.baryDisplacement->valuesInfo->valueLayout, config.maxSubdivLevel, numThreads);

    const bary::Group& group = config.baryDisplacement->groups[config.baryDisplacementGroupIndex];

    micromesh::arraySetData(input.meshTriangleSubdivLevels,
                            &config.baryDisplacement->triangles[group.triangleFirst + config.baryDisplacementMapOffset].subdivLevel,
                            group.triangleCount, sizeof(bary::Triangle));
    arrayInfoTypedFromView(input.meshTrianglePrimitiveFlags, meshView.trianglePrimitiveFlags);
    // TODO future triangleMapping support

    input.pfnGenerateVertex = tessPerVertex<true, true>;
    if(config.baryDisplacement->valuesInfo->valueFormat == bary::Format::eDispC1_r11_unorm_block)
    {
      input.pfnBeginTriangle = tessBeginTriangleCompressed;
    }
    else
    {
      input.pfnBeginTriangle = tessBeginTriangleUncompressed;
    }
  }
  else if(config.heightmapTexture)
  {
    size_t maxAdjacentVertices(std::max(config.topology->maxEdgeTriangleValence, config.topology->maxVertexTriangleValence));
    uint32_t numThreads = micromesh::micromeshOpContextGetConfig(micromeshContext).threadCount;
    payload.threadSanitizeMicroVertices.resize(numThreads);
    for(auto& vec : payload.threadSanitizeMicroVertices)
      vec.resize(maxAdjacentVertices);

    arrayInfoTypedFromView(input.meshTrianglePrimitiveFlags, meshView.trianglePrimitiveFlags);
    arrayInfoTypedFromView(input.meshTriangleSubdivLevels, meshView.triangleSubdivisionLevels);

    payload.triangleSubdivLevels = &input.meshTriangleSubdivLevels;

    input.pfnGenerateVertex = tessPerVertex<true, false>;

    // Handle usesVertexNormalsAsDirections by replacing the vertexDirections
    // ArrayView. This is safe to do since vertexDirections will not be
    // generated in the output mesh.
    if(config.heightmapDesc.usesVertexNormalsAsDirections)
    {
      assert((attribFlags & MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit) == 0);
      if(payload.inMeshView.vertexNormals.empty())
      {
        MESHOPS_LOGE(context, "meshops::Heightmap::usesVertexNormalsAsDirections set but input mesh has no normals");
        return micromesh::Result::eInvalidValue;
      }
      payload.inMeshView.vertexDirections = payload.inMeshView.vertexNormals;
    }
  }
  else
  {
    arrayInfoTypedFromView(input.meshTrianglePrimitiveFlags, meshView.trianglePrimitiveFlags);
    arrayInfoTypedFromView(input.meshTriangleSubdivLevels, meshView.triangleSubdivisionLevels);

    input.pfnGenerateVertex = tessPerVertex<false, false>;
  }

  if((config.baryDisplacement || config.heightmapTexture) && payload.inMeshView.vertexDirections.empty())
  {
    MESHOPS_LOGE(context, "Cannot displace mesh without direction vectors");
    return micromesh::Result::eInvalidValue;
  }

  micromesh::OpTessellateMesh_output output;
  micromesh::Result result = micromesh::micromeshOpTessellateMeshBegin(micromeshContext, &input, &output);
  if(result != micromesh::Result::eSuccess)
  {
    assert(0);
    return result;
  }

  // Resize output mesh for worst case tessellation

  outMesh.resize(attribFlags, output.meshTriangleVertices.count, output.vertexCount);
  output.meshTriangleVertices.data = outMesh.triangleVertices.data();

  // Generate vertices
  result = micromesh::micromeshOpTessellateMeshEnd(micromeshContext, &input, &output);
  if(result != micromesh::Result::eSuccess)
  {
    assert(0);
    return result;
  }

  // shrink vertex buffers due to dedup
  outMesh.resize(attribFlags, output.meshTriangleVertices.count, output.vertexCount);
  assert(output.meshTriangleVertices.count && output.vertexCount);

  if(payload.tessellationError)
  {
    return micromesh::Result::eFailure;
  }
  return result;
}

}  // namespace

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpPreTessellate(Context                      context,
                                                                  size_t                       count,
                                                                  const OpPreTessellate_input* inputs,
                                                                  OpPreTessellate_output*      outputs)
{
  assert(context && inputs && outputs);

  for(size_t i = 0; i < count; ++i)
  {
    const OpPreTessellate_input& input  = inputs[i];
    OpPreTessellate_output&      output = outputs[i];

    TessellateConfig cfg;
    cfg.maxSubdivLevel = input.maxSubdivLevel;

    micromesh::Result result = tessellateMesh(context, input.meshView, *output.meshView, cfg);

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

MESHOPS_API void MESHOPS_CALL meshopsOpDisplacedGetProperties(Context context, OpDisplacedTessellate_properties& properties)
{
  properties.maxHeightmapTessellateLevel = baryutils::BaryLevelsMap::MAX_LEVEL;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpDisplacedTessellate(Context                            context,
                                                                        size_t                             count,
                                                                        const OpDisplacedTessellate_input* inputs,
                                                                        OpDisplacedTessellate_output*      outputs)
{
  assert(context && inputs && outputs);

  for(size_t i = 0; i < count; ++i)
  {
    const OpDisplacedTessellate_input& input  = inputs[i];
    OpDisplacedTessellate_output&      output = outputs[i];

    // Bary and heightmap displacement are mutually exclusive
    if((!input.baryDisplacement && !input.heightmap.texture) || (input.baryDisplacement && input.heightmap.texture))
    {
      assert(0);
      return micromesh::Result::eInvalidValue;
    }

    uint32_t   maxSubdivLevel{};
    HeightMap  heightmapTexture{};
    HeightMap* heightmapTexturePtr{};
    if(input.baryDisplacement)
    {
      if(input.baryNormal
         && input.baryNormal->groups[input.baryDisplacementGroupIndex].maxSubdivLevel
                != input.baryDisplacement->groups[input.baryNormalGroupIndex].maxSubdivLevel)
      {
        assert(0);
        return micromesh::Result::eInvalidValue;
      }

      maxSubdivLevel = input.baryDisplacement->groups[input.baryDisplacementGroupIndex].maxSubdivLevel;
    }
    else
    {
      if(input.heightmap.texture->m_mipSizes.size() != 1 || input.heightmap.texture->m_mipData.size() != 1
         || input.heightmap.texture->m_config.baseFormat != micromesh::Format::eR32_sfloat)
      {
        return micromesh::Result::eInvalidValue;
      }
      if(!input.meshTopology)
      {
        return micromesh::Result::eInvalidValue;
      }
      maxSubdivLevel   = input.heightmap.maxSubdivLevel;
      heightmapTexture = HeightMap(input.heightmap.texture->m_mipSizes[0].x, input.heightmap.texture->m_mipSizes[0].y,
                                   reinterpret_cast<float*>(input.heightmap.texture->m_mipData[0].data()));
      heightmapTexturePtr = &heightmapTexture;
    }

    TessellateConfig cfg;
    cfg.maxSubdivLevel             = maxSubdivLevel;
    cfg.baryDisplacement           = input.baryDisplacement;
    cfg.baryDisplacementGroupIndex = input.baryDisplacementGroupIndex;
    cfg.baryDisplacementMapOffset  = input.baryDisplacementMapOffset;
    cfg.baryNormal                 = input.baryNormal;
    cfg.baryNormalGroupIndex       = input.baryNormalGroupIndex;
    cfg.baryNormalMapOffset        = input.baryNormalMapOffset;
    cfg.heightmapTexture           = heightmapTexturePtr;
    cfg.heightmapDesc              = input.heightmap;
    cfg.topology                   = input.meshTopology;

    micromesh::Result result = tessellateMesh(context, input.meshView, *output.meshView, cfg);

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

}  // namespace meshops
