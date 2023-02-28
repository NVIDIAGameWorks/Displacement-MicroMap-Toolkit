/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "meshops_remesher_internal.hpp"
#include "remeshing_operator.hpp"
#include "meshops_internal/meshops_device_mesh.h"
#include "nvh/parallel_work.hpp"
#include "vk_mem_alloc.h"

using namespace meshops;

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsRemeshingOperatorCreate(Context context, RemeshingOperator* pOp)
{
  (*pOp) = new RemeshingOperator_c();
  return (*pOp)->create(context) ? micromesh::Result::eSuccess : micromesh::Result::eFailure;
}
MESHOPS_API void MESHOPS_CALL meshopsRemeshingOperatorDestroy(Context context, RemeshingOperator op)
{
  op->destroy(context);
  delete op;
}

static bool hasRequiredSettings(const DeviceMeshSettings& settings, const DeviceMeshSettings& requiredSettings)
{
  for(uint32_t i = 0; i < static_cast<uint32_t>(sizeof(settings.attribFlags)) * 8u; i++)
  {
    if(((requiredSettings.attribFlags & (0x1ull << i)) != 0) && ((settings.attribFlags & (0x1ull << i)) == 0))
    {
      return false;
    }
  }
  for(uint32_t i = 0; i < static_cast<uint32_t>(sizeof(settings.usageFlags)) * 8u; i++)
  {
    if(((requiredSettings.usageFlags & (0x1ull << i)) != 0) && ((settings.usageFlags & (0x1ull << i)) == 0))
    {
      return false;
    }
  }
  return true;
}

MESHOPS_API micromesh::Result MESHOPS_CALL
meshopsOpRemesh(Context context, RemeshingOperator op, size_t count, const OpRemesh_input* inputs, OpRemesh_modified* modifieds)
{
  bool checkDeviceMeshes = op->inputDeviceMeshes.empty();
  if(!modifieds[0].meshView->resizable())
  {
    LOGE("Non resizable meshview\n");
    return micromesh::Result::eFailure;
  }
  if(checkDeviceMeshes)
  {
    op->inputDeviceMeshes.reserve(count);
    op->modifiedDeviceMeshes.reserve(count);
  }

  bool finished = true;
  for(size_t i = 0; i < count; i++)
  {
    DeviceMesh modifiedMesh = modifieds[i].deviceMesh;

    DeviceMeshSettings requiredSettings{};

    requiredSettings.attribFlags = eMeshAttributeVertexPositionBit | eMeshAttributeVertexNormalBit | eMeshAttributeVertexTangentBit
                                   | eMeshAttributeVertexDirectionBit | eMeshAttributeVertexDirectionBoundsBit
                                   | eMeshAttributeTriangleVerticesBit | eMeshAttributeTrianglePrimitiveFlagsBit
                                   | eMeshAttributeTriangleSubdivLevelsBit | eMeshAttributeVertexImportanceBit;

    if(checkDeviceMeshes)
    {

      if(modifiedMesh == nullptr)
      {
        micromesh::Result r = meshopsDeviceMeshCreate(context, *modifieds[i].meshView, requiredSettings, &modifiedMesh);
        if(r != micromesh::Result::eSuccess)
          return r;
        op->localDeviceMeshes.push_back(modifiedMesh);
      }
      op->modifiedDeviceMeshes.push_back(modifiedMesh);
    }
    if(!hasRequiredSettings(modifiedMesh->getSettings(), requiredSettings))
    {
      return micromesh::Result::eFailure;
    }

    uint32_t outputTriangleCount{0}, outputVertexCount{0};

    micromesh::Result r = op->remesh(context, inputs[i], modifieds[i], modifiedMesh, &outputTriangleCount, &outputVertexCount);

    if(r == micromesh::Result::eContinue)
    {
      finished = false;
    }

    // Readback and destroy only if remeshing is finished
    if(r == micromesh::Result::eSuccess)
    {
      // need to resize the MutableView to the right size so the readback can work
      modifieds[i].meshView->resize(modifieds[i].meshView->getMeshAttributeFlags(), outputTriangleCount, outputVertexCount);

      modifieds[i].deviceMesh->readback(context, *modifieds[i].meshView);
    }
  }
  // Destroy all the device meshes when all remeshing is finished. Since we had them all in memory during processing
  // there is no gain by removing them one by one upon finishing, and it is simpler to remove them all at the end.
  if(finished)
  {
    for(size_t i = 0; i < op->localDeviceMeshes.size(); i++)
    {
      meshopsDeviceMeshDestroy(context, op->localDeviceMeshes[i]);
    }
    op->localDeviceMeshes.clear();
    op->inputDeviceMeshes.clear();
    op->modifiedDeviceMeshes.clear();
  }

  return micromesh::Result::eSuccess;
}
