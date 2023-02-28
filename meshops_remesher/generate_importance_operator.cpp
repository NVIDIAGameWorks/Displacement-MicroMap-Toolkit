/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "generate_importance_operator.hpp"

#include "meshops_internal/meshops_context.h"
#include "meshops_internal/meshops_device_mesh.h"

#include "_autogen/generate_importance.comp.h"
#include "meshops/meshops_operations.h"
#include "micromesh/micromesh_types.h"
using namespace micromesh;

namespace meshops {

#define PRINT_AND_ASSERT_FALSE(...)                                                                                    \
  {                                                                                                                    \
    printf(__VA_ARGS__);                                                                                               \
    assert(false);                                                                                                     \
  }

#define TEST_TRUE(a)                                                                                                   \
  if(!(a))                                                                                                             \
  {                                                                                                                    \
    PRINT_AND_ASSERT_FALSE("Test failed: " #a "\n");                                                                   \
    return false;                                                                                                      \
  }

#define TEST_SUCCESS(a)                                                                                                \
  if((a) != Result::eSuccess)                                                                                          \
  {                                                                                                                    \
    PRINT_AND_ASSERT_FALSE("Test did not return Result::eSuccess: " #a " \n");                                         \
    return false;                                                                                                      \
  }


static void basicMessageCallback(MessageSeverity severity, const char* message, uint32_t threadIndex, const void* userData)
{
  if(severity == MessageSeverity::eInfo)
  {
    printf("INFO: %s\n", message);
  }
  else if(severity == MessageSeverity::eWarning)
  {
    printf("WARNING: %s\n", message);
  }
  else if(severity == MessageSeverity::eError)
  {
    PRINT_AND_ASSERT_FALSE("ERROR: %s\n", message);
  }
}

static const MessageCallbackInfo messenger{basicMessageCallback, nullptr};


static nvvk::Context* getContext(meshops::Context context)
{
  if(!context || !context->m_vk)
    return nullptr;
  return context->m_vk->m_ptrs.context;
}

bool GenerateImportanceOperator_c::create(Context context)
{
  auto ctx = getContext(context);
  if(!ctx)
  {
    LOGE("GenerateImportanceOperator_c::create(): cannot get Vulkan context\n");
    return false;
  }

  // Setup the texture sampling / raytracing kernel
  m_generateImportance.addBufferBinding(eModifiedVertexPositionNormalBuffer);
  m_generateImportance.addBufferBinding(eModifiedVertexTangentSpaceBuffer);
  m_generateImportance.addBufferBinding(eModifiedVertexTexcoordBuffer);
  m_generateImportance.addBufferBinding(eModifiedVertexDirectionsBuffer);
  m_generateImportance.addBufferBinding(eModifiedVertexImportanceBuffer);
  m_generateImportance.addAccelerationStructureBinding(eMeshAccel);
  m_generateImportance.addSampledImageBinding(eInputImportanceMap);
  m_generateImportance.setCode(ctx->m_device, (void*)generate_importance_comp, sizeof(generate_importance_comp));
  m_generateImportance.finalizePipeline(ctx->m_device);
  m_isInitialized = true;


  // Create a dummy texture for use in case no importance map is provided, to avoid VK validation errors
  VkImageCreateInfo imageCreateInfo =
      nvvk::makeImage2DCreateInfo({1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);
  m_dummyMap = context->m_vk->m_resourceAllocator.createImage(imageCreateInfo);
  {
    nvvk::CommandPool cmdPool(getContext(context)->m_device, getContext(context)->m_queueC.familyIndex);
    VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_dummyMap.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cmdPool.submitAndWait(cmdBuf);
  }

  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  samplerCreateInfo.maxLod     = 0;


  VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(m_dummyMap.image, imageCreateInfo);
  m_dummyTex = context->m_vk->m_resourceAllocator.createTexture(m_dummyMap, ivInfo, samplerCreateInfo);

  m_isInitialized = true;

  return true;
}


bool GenerateImportanceOperator_c::destroy(Context context)
{
  if(!m_isInitialized)
    return true;
  context->m_vk->m_resourceAllocator.destroy(m_dummyTex);
  m_generateImportance.destroy(getContext(context)->m_device);
  m_isInitialized = false;
  return true;
}


bool GenerateImportanceOperator_c::generateImportance(Context context, size_t inputCount, OpGenerateImportance_modified* inputs)
{
  if(inputs == nullptr)
    return false;

  if(!m_isInitialized)
  {
    LOGE("generateImportance(): called before GenerateImportanceOperator_c::create succeeds\n");
    return false;
  }

  for(size_t i = 0; i < inputCount; i++)
  {
    OpGenerateImportance_modified& input = inputs[i];
    // Create the device mesh if needed
    DeviceMesh deviceMesh = input.deviceMesh;
    if(deviceMesh == nullptr)
    {
      DeviceMeshSettings settings;
      settings.usageFlags  = DeviceMeshUsageBlasBit;
      settings.attribFlags = eMeshAttributeVertexPositionBit | eMeshAttributeVertexNormalBit
                             | eMeshAttributeVertexImportanceBit | eMeshAttributeVertexTexcoordBit;
      micromesh::Result result = meshopsDeviceMeshCreate(context, input.meshView, settings, &deviceMesh);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("%s: error generating temporary device mesh: %u\n", __func__, unsigned(result));
        return false;
      }
    }

    if(!deviceMesh)
    {
      LOGE("%s: cannot access device mesh\n", __func__);
      return false;
    }

    DeviceMeshVK* deviceMeshVk = meshopsDeviceMeshGetVK(deviceMesh);
    if(!deviceMeshVk)
    {
      LOGE("%s: cannot access Vulkan device mesh\n", __func__);
      return false;
    }
    // Get the importance texture or use the dummy
    meshops::TextureVK* importanceTextureMeshops = meshopsTextureGetVK(inputs->importanceTexture);
    VkImageView         importanceImageView      = m_dummyTex.descriptor.imageView;
    if(importanceTextureMeshops != nullptr)
    {
      importanceImageView = importanceTextureMeshops->imageView;
    }

    // Update the shader bindings

    m_generateImportance.updateBufferBinding(eModifiedVertexPositionNormalBuffer,
                                             deviceMeshVk->vertexPositionNormalBuffer.buffer);
    m_generateImportance.updateBufferBinding(eModifiedVertexTangentSpaceBuffer, deviceMeshVk->vertexTangentSpaceBuffer.buffer);
    m_generateImportance.updateBufferBinding(eModifiedVertexTexcoordBuffer, deviceMeshVk->vertexTexcoordBuffer.buffer);
    m_generateImportance.updateBufferBinding(eModifiedVertexDirectionsBuffer, deviceMeshVk->vertexDirectionsBuffer.buffer);
    m_generateImportance.updateBufferBinding(eModifiedVertexImportanceBuffer, deviceMeshVk->vertexImportanceBuffer.buffer);
    m_generateImportance.updateAccelerationStructureBinding(eMeshAccel, deviceMeshVk->blas);
    m_generateImportance.updateSampledImageBinding(eInputImportanceMap, m_dummyTex.descriptor.sampler, importanceImageView);

    // Set the push constant values
    m_constants.curvatureMaxDist = input.rayTracingDistance;
    m_constants.hasImportanceMap = (importanceTextureMeshops != nullptr) ? 1 : 0;
    m_constants.texCoordCount    = 1;
    m_constants.texCoordIndex    = 0;
    m_constants.vertexCount      = static_cast<uint32_t>(input.meshView.vertexCount());
    m_constants.curvaturePower   = input.importancePower;

    // Run the generator
    {
      nvvk::CommandPool cmdPool(context->m_vk->m_ptrs.context->m_device, context->m_vk->m_ptrs.context->m_queueC);
      VkCommandBuffer   cmd = cmdPool.createCommandBuffer();
      m_generateImportance.dispatchThreads(cmd, m_constants.vertexCount, &m_constants);
      cmdPool.submitAndWait(cmd);
    }

    // Read back the importance values to the meshView
    DeviceMeshSettings readbackAttributes{};
    readbackAttributes.attribFlags = eMeshAttributeVertexImportanceBit;
    meshopsDeviceMeshReadbackSpecific(context, deviceMesh, input.meshView, readbackAttributes);


    // Delete the device mesh if it has been created locally
    if(input.deviceMesh == nullptr)
    {
      meshopsDeviceMeshDestroy(context, deviceMesh);
    }
  }
  return true;
}


}  // namespace meshops