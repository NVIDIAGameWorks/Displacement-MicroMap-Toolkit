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

#include <baryutils/baryutils.h>
#include <cstdint>
#include <meshops_bake_vk.hpp>
#include <meshops_internal/meshops_context.h>
#include <meshops_internal/meshops_texture.h>
#include <meshops/meshops_operations.h>
#include <meshops/meshops_mesh_view.h>
#include <memory>
#include <micromesh/micromesh_types.h>
#include <micromesh/micromesh_operations.h>
#include <microutils/microutils.hpp>
#include <nvmath/nvmath_types.h>

namespace meshops {

class BakerOperator_c
{
};

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsBakeOperatorCreate(Context context, BakerOperator* pOp)
{
  assert(context);

  *pOp = new BakerOperator_c;
  return micromesh::Result::eSuccess;
}

MESHOPS_API void MESHOPS_CALL meshopsBakeOperatorDestroy(Context context, BakerOperator op)
{
  delete op;
}

void initBaryData(const meshops::MeshView& meshView, uint32_t defaultSubdivLevel, baryutils::BaryBasicData* baryBasic)
{
  *baryBasic = {};

  baryBasic->minSubdivLevel = ~0U;
  baryBasic->maxSubdivLevel = 0U;

  uint32_t valuesOffset = 0U;
  {
    // Add one group for the given MeshView
    bary::Group baryGroup;
    baryGroup.minSubdivLevel = ~0U;
    baryGroup.maxSubdivLevel = 0U;
    baryGroup.triangleFirst  = 0U;
    baryGroup.valueFirst     = 0U;
    baryGroup.floatBias      = bary::ValueFloatVector{0.0f, 0.0f, 0.0f, 0.0f};
    baryGroup.floatScale     = bary::ValueFloatVector{1.0f, 0.0f, 0.0f, 0.0f};
    baryGroup.triangleCount  = static_cast<uint32_t>(meshView.triangleCount());
    baryBasic->triangles.reserve(baryGroup.triangleCount);
    uint32_t defaultMicroVertexCount = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, defaultSubdivLevel);
    for(size_t i = 0; i < meshView.triangleCount(); ++i)
    {
      uint16_t subdivLevel = meshView.triangleSubdivisionLevels.empty() ? static_cast<uint16_t>(defaultSubdivLevel) :
                                                                          meshView.triangleSubdivisionLevels[i];
      uint32_t triangleMicroVertexCount = meshView.triangleSubdivisionLevels.empty() ?
                                              defaultMicroVertexCount :
                                              bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, subdivLevel);
      baryBasic->triangles.push_back(bary::Triangle{valuesOffset, subdivLevel, {0}});
      valuesOffset += triangleMicroVertexCount;

      baryGroup.minSubdivLevel = std::min(baryGroup.minSubdivLevel, uint32_t(subdivLevel));
      baryGroup.maxSubdivLevel = std::max(baryGroup.maxSubdivLevel, uint32_t(subdivLevel));
    }
    baryGroup.valueCount = valuesOffset;
    baryBasic->groups.push_back(baryGroup);

    // Update the min/max subdiv level
    baryBasic->minSubdivLevel = std::min(baryBasic->minSubdivLevel, baryGroup.minSubdivLevel);
    baryBasic->maxSubdivLevel = std::max(baryBasic->maxSubdivLevel, baryGroup.maxSubdivLevel);
  }

  // The total micro triangles is everything in the group
  uint32_t microTriangleCount = valuesOffset;

  // Allocate displacement values, populated by the baker
  baryBasic->valuesInfo.valueCount         = microTriangleCount;
  baryBasic->valuesInfo.valueLayout        = bary::ValueLayout::eTriangleBirdCurve;
  baryBasic->valuesInfo.valueFrequency     = bary::ValueFrequency::ePerVertex;
  baryBasic->valuesInfo.valueFormat        = bary::Format::eR32_sfloat;
  baryBasic->valuesInfo.valueByteAlignment = 4;
  baryBasic->valuesInfo.valueByteSize = (baryutils::baryDisplacementFormatGetNumBits(baryBasic->valuesInfo.valueFormat) + 7) / 8;
  assert(baryBasic->valuesInfo.valueByteSize != 0);
  baryBasic->values.resize(static_cast<size_t>(baryBasic->valuesInfo.valueCount) * baryBasic->valuesInfo.valueByteSize);

  // Allocate per-triangle min/max displacement, populated by the baker
  baryBasic->triangleMinMaxsInfo.elementFormat        = bary::Format::eR32_sfloat;
  baryBasic->triangleMinMaxsInfo.elementByteAlignment = 4;
  baryBasic->triangleMinMaxsInfo.elementByteSize =
      (baryutils::baryDisplacementFormatGetNumBits(baryBasic->triangleMinMaxsInfo.elementFormat) + 7) / 8;
  baryBasic->triangleMinMaxsInfo.elementCount = static_cast<uint32_t>(baryBasic->triangles.size() * 2);
  baryBasic->triangleMinMaxs.resize(baryBasic->triangleMinMaxsInfo.elementCount * baryBasic->triangleMinMaxsInfo.elementByteSize);
}

MESHOPS_API void MESHOPS_CALL meshopsBakeGetProperties(Context context, BakerOperator op, OpBake_properties& properties)
{
  // Defined in host_device.h
  properties.maxLevel                    = BAKER_MAX_SUBDIV_LEVEL;
  properties.maxResamplerTextures        = MAX_RESAMPLE_TEXTURES;
  properties.maxHeightmapTessellateLevel = baryutils::BaryLevelsMap::MAX_LEVEL;
}

MESHOPS_API void MESHOPS_CALL meshopsBakeGetRequirements(Context                          context,
                                                         BakerOperator                    op,
                                                         const OpBake_settings&           settings,
                                                         ArrayView<OpBake_resamplerInput> resamplerInput,
                                                         bool                             uniformSubdivLevels,
                                                         bool                             referenceHasHeightmap,
                                                         bool                 heightmapUsesNormalsAsDirections,
                                                         OpBake_requirements& properties)
{
  // Heightmaps may introduce cracks which need to be welded shut. This requires
  // a topology based on unique vertex positions. Heightmaps may also require
  // baking in batches (which require increasing each batch's triangle selection
  // for an overlap).
  properties.referenceMeshTopology = referenceHasHeightmap;

  // Minimum required attributes
  properties.baseMeshAttribFlags = MeshAttributeFlagBits::eMeshAttributeTriangleVerticesBit
                                   | MeshAttributeFlagBits::eMeshAttributeVertexPositionBit
                                   | MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit;
  properties.referenceMeshAttribFlags =
      MeshAttributeFlagBits::eMeshAttributeTriangleVerticesBit | MeshAttributeFlagBits::eMeshAttributeVertexPositionBit;

  if(!uniformSubdivLevels)
  {
    properties.baseMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit;
  }

  // Heightmaps require texture coordinates and either normals or direction vectors
  if(referenceHasHeightmap)
  {
    // Subdivision levels and edge flags should be generated to match the heightmap resolution
    properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit;
    properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit;

    // Texture coordinates for sampling
    properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexTexcoordBit;

    if(heightmapUsesNormalsAsDirections)
    {
      properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexNormalBit;
    }
    else
    {
      properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit;
    }
  }

  // Resampling requires texture coordinates
  if(!resamplerInput.empty())
  {
    properties.baseMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexTexcoordBit;
    properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexTexcoordBit;

    // Tangent space is required if resampling normal maps or quaternion maps
    bool requiresNormals = false;
    for(auto& resamplerInput : resamplerInput)
      requiresNormals = requiresNormals || resamplerInput.textureType == TextureType::eNormalMap
                        || resamplerInput.textureType == TextureType::eOffsetMap;

    if(requiresNormals)
    {
      properties.baseMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexNormalBit;
      properties.baseMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexTangentBit;
      properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexNormalBit;
      properties.referenceMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexTangentBit;
    }
  }

  // Bounds fitting currently requires initial bounds. This could probably be changed.
  if(settings.fitDirectionBounds)
  {
    properties.baseMeshAttribFlags |= MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit;
  }
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpBake(Context context, BakerOperator op, const OpBake_input& input, OpBake_output& output)
{
  assert(context && op);

  if(input.resamplerInput.size() != output.resamplerTextures.size())
  {
    MESHOPS_LOGE(context,
                 "OpBake_input::resamplerInput size (%zu) must match OpBake_output::resamplerTextures size (%zu)",
                 input.resamplerInput.size(), output.resamplerTextures.size());
    return micromesh::Result::eInvalidRange;
  }

  // Debugging
#if 0
  MESHOPS_LOGI(context, "OpBake_input::baseMeshView has %s|%s",
                triangleAttribBitsString(input.baseMeshView.getTriangleAttributeFlags()).c_str(),
                vertexAttribBitsString(input.baseMeshView.getVertexAttributeFlags()).c_str());
  MESHOPS_LOGI(context, "OpBake_input::referenceMeshView has %s|%s",
                triangleAttribBitsString(input.referenceMeshView.getTriangleAttributeFlags()).c_str(),
                vertexAttribBitsString(input.referenceMeshView.getVertexAttributeFlags()).c_str());
#endif

  {
    OpBake_requirements meshRequirements;
    meshopsBakeGetRequirements(context, op, input.settings, input.resamplerInput,
                               input.baseMeshView.triangleSubdivisionLevels.empty(), input.referenceMeshHeightmap.texture != nullptr,
                               input.referenceMeshHeightmap.usesVertexNormalsAsDirections, meshRequirements);

    // Validate the topology exists
    if(meshRequirements.referenceMeshTopology && !input.referenceMeshTopology)
    {
      MESHOPS_LOGE(context, "OpBake_input::referenceMeshTopology is null, but required by OpBake_requirements");
      return micromesh::Result::eInvalidValue;
    }

    // Validate required mesh attributes exist
    if(!input.baseMeshView.hasMeshAttributeFlags(meshRequirements.baseMeshAttribFlags))
    {
      auto missingAttributes = (~input.baseMeshView.getMeshAttributeFlags()) & meshRequirements.baseMeshAttribFlags;
      MESHOPS_LOGE(context, "OpBake_input::baseMeshView is missing %s mesh attribs",
                   meshAttribBitsString(missingAttributes).c_str());
      return micromesh::Result::eInvalidValue;
    }
    if(!input.referenceMeshView.hasMeshAttributeFlags(meshRequirements.referenceMeshAttribFlags))
    {
      auto missingAttributes = (~input.referenceMeshView.getMeshAttributeFlags()) & meshRequirements.referenceMeshAttribFlags;
      MESHOPS_LOGE(context, "OpBake_input::referenceMeshView is missing %s mesh attribs",
                   meshAttribBitsString(missingAttributes).c_str());
      return micromesh::Result::eInvalidValue;
    }
  }

  // Validate limits
  {
    OpBake_properties properties;
    meshopsBakeGetProperties(context, op, properties);

    if(input.settings.level > properties.maxLevel)
    {
      MESHOPS_LOGE(context, "OpBake_input::settings.level of %u is above the maximum, %u", input.settings.level,
                   properties.maxLevel);
      return micromesh::Result::eInvalidValue;
    }
    if(input.resamplerInput.size() > properties.maxResamplerTextures)
    {
      MESHOPS_LOGE(context, "OpBake_input::resamplerInput size of %zu is above the maximum, %u",
                   input.resamplerInput.size(), properties.maxResamplerTextures);
      return micromesh::Result::eInvalidValue;
    }
    if(input.referenceMeshHeightmap.texture != nullptr && input.referenceMeshHeightmap.maxSubdivLevel > properties.maxHeightmapTessellateLevel)
    {
      MESHOPS_LOGE(context, "OpBake_input::referenceMeshHeightmap.maxSubdivLevel of %u is above the maximum, %u",
                   input.referenceMeshHeightmap.maxSubdivLevel, properties.maxHeightmapTessellateLevel);
      return micromesh::Result::eInvalidValue;
    }
  }

  if(input.referenceMeshHeightmap.texture
     && (input.referenceMeshHeightmap.texture->m_mipData.size() != 1
         || input.referenceMeshHeightmap.texture->m_config.baseFormat != micromesh::Format::eR32_sfloat))
  {
    MESHOPS_LOGE(context, "OpBake_input::referenceMeshHeightmap must be eR32_sfloat and host-accessible");
    return micromesh::Result::eInvalidValue;
  }

  // Having input direction bounds implies uni-directional tracing (i.e. don't
  // trace below the lower bound). While bidirectional tracing would work, the
  // result would end up being clamped to the 0 to 1 range. This may be
  // unintuitive to silently ignore. Unfortunately we can't tell the difference
  // between input and output direction bounds as they're updated in-place, so
  // only error out when not fitting.
  if(!input.settings.uniDirectional && !input.settings.fitDirectionBounds && !input.baseMeshView.vertexDirectionBounds.empty())
  {
    MESHOPS_LOGE(context, "OpBake_input::settings.uniDirectional must be true when mesh has direction bounds.");
    return micromesh::Result::eInvalidValue;
  }

  // Create VkDescriptorImageInfo for all textures. Use the one sampler for everything
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  nvvk::SamplerPool samplerPool(context->m_vk->m_ptrs.context->m_device);
  auto              samplerDeleter = [&samplerPool](VkSampler_T* s) { samplerPool.releaseSampler(s); };
  auto              sampler = std::unique_ptr<VkSampler_T, decltype(samplerDeleter)>(samplerPool.acquireSampler(samplerCreateInfo), samplerDeleter);

  std::vector<VkDescriptorImageInfo> inputTextures;
  std::vector<VkDescriptorImageInfo> outputTextures;
  std::vector<VkDescriptorImageInfo> distanceTextures;
  for(auto& texture : input.resamplerInput)
  {
    if(texture.textureCoord != 0)
    {
      MESHOPS_LOGE(context, "Non-zero OpBake_input::ResamplerInput::texCoordIndex (%u) is not supported", texture.textureCoord);
      return micromesh::Result::eInvalidValue;
    }
    bool generatedTextureType = texture.textureType == TextureType::eQuaternionMap || texture.textureType == TextureType::eOffsetMap
                                || texture.textureType == TextureType::eHeightMap;
    if(texture.texture)
    {
      if(generatedTextureType)
      {
        MESHOPS_LOGW(context, "OpBake_input::ResamplerInput::texture should be null for non-resampled texture types");
      }
      if(texture.texture->m_vk.imageView == VK_NULL_HANDLE)
      {
        MESHOPS_LOGE(context, "Baker currently only supports vulkan images");
        return micromesh::Result::eInvalidValue;
      }
      inputTextures.push_back(
          VkDescriptorImageInfo{sampler.get(), texture.texture->m_vk.imageView, texture.texture->m_vk.imageLayout});
    }
    else
    {
      if(!generatedTextureType)
      {
        MESHOPS_LOGE(context, "OpBake_input::ResamplerInput::texture must be null for generated texture types");
        return micromesh::Result::eInvalidValue;
      }
      // Insert a null object so that input and output texture arrays remain 1:1.
      inputTextures.push_back(VkDescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    }
    distanceTextures.push_back(
        VkDescriptorImageInfo{sampler.get(), texture.distance->m_vk.imageView, texture.distance->m_vk.imageLayout});
  }
  for(auto& texture : output.resamplerTextures)
  {
    if(texture->m_vk.imageView == VK_NULL_HANDLE)
    {
      MESHOPS_LOGE(context, "Baker currently only supports vulkan images");
      return micromesh::Result::eInvalidValue;
    }
    outputTextures.push_back(VkDescriptorImageInfo{sampler.get(), texture->m_vk.imageView, texture->m_vk.imageLayout});
  }

  // TODO put BakerVK into BakerOperator_c and make it re-usable
  BakerVK baker(context->m_micromeshContext, context->m_vk->m_ptrs);

  // Allocate storage for the result. We compute displacements for every microvertex regardless of edge flags.
  initBaryData(input.baseMeshView, input.settings.level, output.uncompressedDisplacement);
  assert(output.uncompressedDisplacement->valuesInfo.valueFormat == bary::Format::eR32_sfloat);
  ArrayView distances(reinterpret_cast<float*>(output.uncompressedDisplacement->values.data()),
                      output.uncompressedDisplacement->valuesInfo.valueCount);
  ArrayView triangleMinMaxs(reinterpret_cast<nvmath::vec2f*>(output.uncompressedDisplacement->triangleMinMaxs.data()),
                            output.uncompressedDisplacement->triangleMinMaxsInfo.elementCount / 2);

  // Create GPU buffers for the base mesh and output
  baker.create(input, distances);

  // Compute remaining memory available for the baker reference mesh. Textures
  // and data for the base mesh have already been allocated.
  VkDeviceSize memoryBudget, memoryUsage;
  getMemoryUsageVk(context->m_vk->m_ptrs.context->m_physicalDevice, &memoryBudget, &memoryUsage);
  VkDeviceSize memoryAvailable = memoryBudget - memoryUsage;
  memoryAvailable              = (memoryAvailable * 9) / 10;  // Hard limit on 90%

  // The user defined limit can artificially set the memory to fit within.
  // Hopefully this hasn't already been exceeded.
  if(input.settings.memLimitBytes != 0)
  {
    memoryAvailable = input.settings.memLimitBytes > memoryUsage ? input.settings.memLimitBytes - memoryUsage : 0;
  }

  // Arbitrary low memory warning
  if(memoryAvailable < 512 * 1024 * 1024)
  {
    MESHOPS_LOGW(context, "remaining memory for baking is %.2f MiB", static_cast<double>(memoryAvailable) / 1024.0 / 1024.0);
  }

  auto batches = computeBatches(context, memoryAvailable, input.referenceMeshTopology, input.referenceMeshView);
  assert(batches.size() == 1 || input.referenceMeshHeightmap.texture);  // Batching is only supported when tessellating for heightmaps

  // Bake
  bool firstPassResamples = !input.settings.fitDirectionBounds;
  for(const auto& batch : batches)
  {
    baker.bakeAndResample(input, batch, firstPassResamples, inputTextures, outputTextures, distanceTextures, output.resamplerTextures);
  }

  // Fit direction bounds and re-bake
  if(input.settings.fitDirectionBounds)
  {
    // Compute min/max displacement and re-run the compute pass with updated direction vectors
    const int fitPasses = 1;
    for(int i = 0; i < fitPasses; ++i)
    {
      MESHOPS_LOGI(context, "Bounds fitting pass %i/%i (simple min/max)", i + 1, fitPasses);
      baker.fitDirectionBounds(input, distances);

      // Re-run all batches with the new direction bounds
      for(const auto& batch : batches)
      {
        baker.bakeAndResample(input, batch, true, inputTextures, outputTextures, distanceTextures, output.resamplerTextures);
      }
    }
  }

  // Retrieve data from buffer
  nvmath::vec2f globalMinMax;
  baker.getDistanceFromBuffer(input, output.vertexDirectionBounds, distances, triangleMinMaxs, globalMinMax);
  if(globalMinMax.y - globalMinMax.x <= 0.0000001f)
  {
    MESHOPS_LOGW(context,
                 "Displacement micromap was considered flat. Either there was a problem during baking or displacement "
                 "could be removed from this mesh.");
  }

  // Displacement distance post-processing
  assert(output.uncompressedDisplacement->groups.size() == 1);
  const micromesh::MeshTopology* baseMeshTopology = input.baseMeshTopology;
  baryutils::BaryBasicData&      baryData         = *output.uncompressedDisplacement;
  bary::BasicView                baryView         = baryData.getView();
  for(size_t group = 0; group < output.uncompressedDisplacement->groups.size(); ++group)
  {
    micromesh::MicromapGeneric micromap;
    micromesh::ArrayInfo       minMaxs;
    microutils::baryBasicViewToMicromap(baryView, static_cast<uint32_t>(group), micromap);
    microutils::baryBasicViewToMinMaxs(baryView, static_cast<uint32_t>(group), minMaxs);
    micromesh::Micromap& micromapFloat = micromap.uncompressed;

    // Fitted direction bounds will gurantee the values are between 0 and 1. If
    // we are not fitting, but direction bounds are provided, assume they're
    // good and don't try to re-normalize. Otherwise, normalize the
    // dispalcements to the 0 to 1 range and apply the inverse transform to the
    // bary group's bias and scale.
    bool normalize = input.baseMeshView.vertexDirectionBounds.empty();
    assert(!input.settings.fitDirectionBounds || !input.baseMeshView.vertexDirectionBounds.empty());  // Should have bounds when fitting
    assert(input.baseMeshView.vertexDirectionBounds.empty() || (globalMinMax.x == 0.0f && globalMinMax.y == 1.0f));  // Should be clamping when bounds are used

    // Min/max values are already populated by getDistanceFromBuffer(), although due to using
    // encodeMinMaxFp32/decodeMinMaxFp32 for atomics, results are slightly different
    if(normalize)
    {
      micromesh::OpComputeTriangleMinMaxs_output output;
      output.triangleMins = minMaxs;
      output.triangleMins.byteStride <<= 1;
      output.triangleMins.count >>= 1;
      output.triangleMaxs = minMaxs;
      reinterpret_cast<float*&>(output.triangleMaxs.data) += 1;
      output.triangleMaxs.byteStride <<= 1;
      output.triangleMaxs.count >>= 1;
      micromesh::Result result =
          micromesh::micromeshOpComputeTriangleMinMaxs(context->m_micromeshContext, &micromapFloat, &output);
      if(result != micromesh::Result::eSuccess)
      {
        MESHOPS_LOGE(context, "micromesh::micromeshOpComputeTriangleMinMaxs() failed");
        return result;
      }
      globalMinMax.x = output.globalMin.value_float[0];
      globalMinMax.y = output.globalMax.value_float[0];
    }

    // Scale both distances and minMaxs to keep them in the 0 to 1 range
    if(normalize)
    {
      micromesh::OpFloatToQuantized_input input;
      input.floatMicromap            = &micromapFloat;
      input.globalMin.value_float[0] = globalMinMax.x;
      input.globalMax.value_float[0] = globalMinMax.y;
      input.outputUnsignedSfloat     = true;

      micromesh::Result result = micromeshOpFloatToQuantized(context->m_micromeshContext, &input, &micromapFloat);
      if(result != micromesh::Result::eSuccess)
      {
        MESHOPS_LOGE(context, "micromesh::micromeshOpFloatToQuantized() failed");
        return result;
      }

      // The same transform needs to be applied to all the min/max values
      {
        // bit of a hack, we just override the values array, since that is the only thing
        // manipulated if the triangle arrays match
        micromesh::Micromap minMaxsAsMicromap;
        minMaxsAsMicromap.values = minMaxs;
        input.floatMicromap      = &minMaxsAsMicromap;
        result                   = micromeshOpFloatToQuantized(context->m_micromeshContext, &input, &minMaxsAsMicromap);
        if(result != micromesh::Result::eSuccess)
        {
          MESHOPS_LOGE(context, "micromesh::micromeshOpFloatToQuantized() failed");
          return result;
        }
      }

      // Save the transform so the values can be restored to their original range when rendering
      baryData.groups[group].floatScale.r = micromapFloat.valueFloatExpansion.scale[0];
      baryData.groups[group].floatBias.r  = micromapFloat.valueFloatExpansion.bias[0];
    }

    assert(micromapFloat.values.format == micromesh::Format::eR32_sfloat);

    // Seal cracks by forcing values along shared base triangle edges to match.
    {
      micromesh::OpSanitizeEdgeValues_input input;
      input.meshTopology = baseMeshTopology;
      micromesh::Result result = micromesh::micromeshOpSanitizeEdgeValues(context->m_micromeshContext, &input, &micromapFloat);
      if(result != micromesh::Result::eSuccess)
      {
        MESHOPS_LOGE(context, "micromesh::micromeshOpSanitizeEdgeValues() failed");
        return result;
      }
    }
  }

  return micromesh::Result::eSuccess;
}

}  // namespace meshops
