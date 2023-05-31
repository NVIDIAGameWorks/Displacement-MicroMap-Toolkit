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
#include <meshops/meshops_operations.h>

namespace meshops {

//////////////////////////////////////////////////////////////////////////

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpBuildTopology(Context                      context,
                                                                  size_t                       count,
                                                                  const OpBuildTopology_input* inputs,
                                                                  OpBuildTopology_output*      outputs)
{
  assert(context && inputs && outputs);

  for(size_t i = 0; i < count; ++i)
  {
    const OpBuildTopology_input& input  = inputs[i];
    OpBuildTopology_output&      output = outputs[i];

    ArrayView<const micromesh::Vector_float_3> positions(input.meshView.vertexPositions);

    micromesh::Result result;

    if(!input.triangleUniqueVertexIndices.empty())
    {
      ArrayView<const uint32_t> indices(input.triangleUniqueVertexIndices);

      result = output.meshTopology->buildFromIndicesAsIs(context->m_micromeshContext, indices.size(), indices.data(),
                                                         positions.size());
    }
    else
    {
      ArrayView<const uint32_t> indices(input.meshView.triangleVertices);
      result = output.meshTopology->buildFindingWatertightIndices(context->m_micromeshContext, indices.size(),
                                                                  indices.data(), positions.size(), positions.data(),
                                                                  static_cast<uint32_t>(positions.stride()));
    }

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpFindUniqueVertexIndices(Context context,
                                                                            size_t  count,
                                                                            const OpFindUniqueVertexIndices_input* inputs,
                                                                            OpFindUniqueVertexIndices_output* outputs)
{
  assert(context && inputs && outputs);

  for(size_t i = 0; i < count; ++i)
  {
    const OpFindUniqueVertexIndices_input& input  = inputs[i];
    OpFindUniqueVertexIndices_output&      output = outputs[i];

    micromesh::OpBuildMeshTopologyIndices_input opInput;
    arrayInfoTypedFromView(opInput.meshTriangleVertices, input.meshView.triangleVertices);
    arrayInfoTypedFromView(opInput.meshVertexPositions, input.meshView.vertexPositions);

    micromesh::OpBuildMeshTopologyIndices_output opOutput;
    arrayInfoTypedFromView(opOutput.meshTopologyTriangleVertices, output.triangleUniqueVertexIndices);

    micromesh::Result result = micromesh::micromeshOpBuildMeshTopologyIndices(context->m_micromeshContext, &opInput, &opOutput);

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }
  return micromesh::Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateSubdivisionLevel(Context context,
                                                                             size_t  count,
                                                                             const OpGenerateSubdivisionLevel_input* inputs,
                                                                             OpGenerateSubdivisionLevel_modified* modifieds)
{
  assert(context && inputs && modifieds);

  for(size_t i = 0; i < count; ++i)
  {
    const OpGenerateSubdivisionLevel_input& input    = inputs[i];
    OpGenerateSubdivisionLevel_modified&    modified = modifieds[i];

    micromesh::OpAdaptiveSubdivision_input opInput;
    opInput.maxSubdivLevel              = input.maxSubdivLevel;
    opInput.positionScale.x             = 1.0f;
    opInput.positionScale.y             = 1.0f;
    opInput.positionScale.z             = 1.0f;
    opInput.useRelativeValues           = !input.useTextureArea;
    opInput.onlyComputeRelativeMaxValue = false;
    opInput.useArea                     = input.useTextureArea;
    opInput.relativeWeight              = input.relativeWeight;
    opInput.texResolution.x             = float(input.textureWidth);
    opInput.texResolution.y             = float(input.textureHeight);
    opInput.subdivLevelBias             = input.subdivLevelBias;

    opInput.useRelativeMaxValueOverride = !input.useTextureArea && input.maxEdgeLengthOverride != 0.0f;
    opInput.relativeMaxValueOverride    = input.maxEdgeLengthOverride;

    arrayInfoTypedFromView(opInput.meshTriangleVertices, modified.meshView.triangleVertices);
    if(input.useTextureArea)
    {
      if(modified.meshView.vertexTexcoords0.empty())
      {
        MESHOPS_LOGE(context, "useTextureArea is set but meshView.vertexTexcoords0 is empty");
        return micromesh::Result::eInvalidValue;
      }
      if(input.textureWidth == 0 || input.textureHeight == 0)
      {
        MESHOPS_LOGE(context, "useTextureArea requires non-zero textureWidth and textureHeight");
        return micromesh::Result::eInvalidValue;
      }
      arrayInfoTypedFromView(opInput.meshVertexTexcoords, modified.meshView.vertexTexcoords0);
    }
    else
    {
      arrayInfoTypedFromView(opInput.meshVertexPositions, modified.meshView.vertexPositions);
    }

    micromesh::OpAdaptiveSubdivision_output opOutput;
    arrayInfoTypedFromView(opOutput.meshTriangleSubdivLevels, modified.meshView.triangleSubdivisionLevels);

    micromesh::Result result = micromesh::micromeshOpAdaptiveSubdivision(context->m_micromeshContext, &opInput, &opOutput);
    if(result != micromesh::Result::eSuccess)
    {
      return result;
    }

    modified.maxSubdivLevel = opOutput.maxSubdivLevel;
    modified.minSubdivLevel = opOutput.minSubdivLevel;
  }

  return micromesh::Result::eSuccess;
}


MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpSanitizeSubdivisionLevel(Context context,
                                                                             size_t  count,
                                                                             const OpSanitizeSubdivisionLevel_input* inputs,
                                                                             OpSanitizeSubdivisionLevel_modified* modifieds)
{
  assert(context && inputs && modifieds);

  for(size_t i = 0; i < count; ++i)
  {
    const OpSanitizeSubdivisionLevel_input& input    = inputs[i];
    OpSanitizeSubdivisionLevel_modified&    modified = modifieds[i];

    micromesh::OpSanitizeSubdivLevels_input opInput;
    opInput.maxSubdivLevel = input.maxSubdivLevel;
    opInput.meshTopo       = input.meshTopology;
    arrayInfoTypedFromView(opInput.meshTriangleSubdivLevels, modified.meshView.triangleSubdivisionLevels);

    micromesh::OpSanitizeSubdivLevels_output opOutput;
    arrayInfoTypedFromView(opOutput.meshTriangleSubdivLevels, modified.meshView.triangleSubdivisionLevels);

    micromesh::Result result = micromesh::micromeshOpSanitizeSubdivLevels(context->m_micromeshContext, &opInput, &opOutput);

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }

    modified.minSubdivLevel = opOutput.minSubdivLevel;
  }

  return micromesh::Result::eSuccess;
}


MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpBuildPrimitiveFlags(Context                            context,
                                                                        size_t                             count,
                                                                        const OpBuildPrimitiveFlags_input* inputs,
                                                                        OpBuildPrimitiveFlags_modified*    modifieds)
{
  assert(context && inputs && modifieds);

  for(size_t i = 0; i < count; ++i)
  {
    const OpBuildPrimitiveFlags_input& input    = inputs[i];
    OpBuildPrimitiveFlags_modified&    modified = modifieds[i];

    micromesh::OpBuildPrimitiveFlags_input opInput;
    opInput.meshTopo = input.meshTopology;
    arrayInfoTypedFromView(opInput.meshTriangleSubdivLevels, modified.meshView.triangleSubdivisionLevels);

    micromesh::OpBuildPrimitiveFlags_output opOutput;
    arrayInfoTypedFromView(opOutput.meshTrianglePrimitiveFlags, modified.meshView.trianglePrimitiveFlags);

    micromesh::Result result = micromesh::micromeshOpBuildPrimitiveFlags(context->m_micromeshContext, &opInput, &opOutput);

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

struct ReadSubdivPayload
{
  const bary::BasicView& baryView;
  MutableMeshView&       meshView;
  uint32_t               groupIndex = 0;

  static void worker(uint64_t itemFirst, uint64_t itemLast, uint32_t threadIndex, void* userData)
  {
    ReadSubdivPayload&    payload        = *reinterpret_cast<ReadSubdivPayload*>(userData);
    const bary::Group&    group          = payload.baryView.groups[payload.groupIndex];
    const bary::Triangle* groupTriangles = payload.baryView.triangles + group.triangleFirst;

    for(uint64_t idx = itemFirst; idx < itemLast; idx++)
    {
      const bary::Triangle& tri                       = groupTriangles[idx];
      payload.meshView.triangleSubdivisionLevels[idx] = tri.subdivLevel;
    }
  }
};


MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpReadSubdivisionLevel(Context                             context,
                                                                         size_t                              count,
                                                                         const OpReadSubdivisionLevel_input* inputs,
                                                                         OpReadSubdivisionLevel_modified*    modifieds)
{
  assert(context && inputs && modifieds);

  for(size_t i = 0; i < count; ++i)
  {
    const OpReadSubdivisionLevel_input& input    = inputs[i];
    OpReadSubdivisionLevel_modified&    modified = modifieds[i];

    ReadSubdivPayload payload = {*input.baryData, modified.meshView};

    micromesh::OpDistributeWork_input opInput;
    opInput.pfnGenericRangeWorkload = ReadSubdivPayload::worker;
    opInput.userData                = &payload;

    micromesh::Result result =
        micromeshOpDistributeWork(context->m_micromeshContext, &opInput, modified.meshView.triangleCount());

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

}  // namespace meshops
