/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


// Remeshing of a GLTF or OBJ file, and save into a GLTF


#include "remeshing_operator.hpp"

#include "meshops_internal/meshops_context.h"
#include "meshops_internal/meshops_device_mesh.h"

#include "_autogen/vertex_merge.comp.h"
#include "_autogen/vertex_copy.comp.h"
#include "meshops/meshops_operations.h"
#include "micromesh/micromesh_gpu.h"
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


std::string getRemeshingErrorString(RemesherErrorState e)
{
#define REMESHING_ERROR_CASE(x_)                                                                                       \
  case x_:                                                                                                             \
    return #x_

  switch(e)
  {
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorNone);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorVertexHashNotFound);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorEdgeHashNotFound);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorDebug);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorOutOfEdgeStorage);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorNoTriangleFound);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorNoVertexHistoryFound);
    REMESHING_ERROR_CASE(RemesherErrorState::eRemesherErrorInvalidConstantValue);
    default:
      return "Unknown remesher error";
  }

#undef REMESHING_ERROR_CASE
}


static nvvk::Context* getContext(meshops::Context context)
{
  return context->m_vk->m_ptrs.context;
}

nvvk::Buffer allocateRemesherBuffer(micromesh::gpu::ResourceAllocInfo allocInfo, nvvk::ResourceAllocator* alloc, bool isConstantBuffer = false)
{
  if(allocInfo.type != gpu::DescriptorType::eBufferRead && allocInfo.type != gpu::DescriptorType::eBufferReadWrite
     && allocInfo.type != gpu::DescriptorType::eConstantBuffer)
  {
    LOGE("Wrong resource type for allocateRemesherBuffer");
    return nvvk::Buffer();
  }

  auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if(isConstantBuffer)
    usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  return alloc->createBuffer(allocInfo.buffer.size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}


bool RemeshingOperator_c::create(Context context)
{
  auto                          ctx = getContext(context);
  gpu::AvailableShaderCodeTypes availableTypes;
  if(Result::eSuccess != (micromeshGpuRemeshingGetAvailableShaderCodeTypes(&availableTypes)))
  {
    LOGE("Could not fetch remesher available code types\n");
    return false;
  }

  if(!availableTypes.isAvailable[gpu::eShaderCodeSPIRV])
  {
    LOGE("No SPIR-V code available");
    return false;
  }

  gpu::GpuRemeshing_config config;
  RemeshingMode            mode = RemeshingMode::eDecimate;
  config.codeType               = gpu::eShaderCodeSPIRV;
  config.supportedModeCount     = 1;
  config.supportedModes         = &mode;
  if(Result::eSuccess != (gpu::micromeshGpuRemeshingCreate(&config, &m_remesher, &messenger)))
  {
    LOGE("Could not create remesher\n");
    return false;
  }


  if(Result::eSuccess != (gpu::micromeshGpuRemeshingGetSetup(m_remesher, &m_remesherSetupInfo)))
  {
    LOGE("Could not setup the remesher\n");
    return false;
  }

  // constant buffer
  if(m_remesherSetupInfo.globalConstantBuffer.buffer.size)
  {
    m_globalConstantBuffer =
        allocateRemesherBuffer(m_remesherSetupInfo.globalConstantBuffer, &context->m_vk->m_resourceAllocator, true);
  }

  // read resources
  m_readResourceDatas.resize(m_remesherSetupInfo.readResourcesMaxCount);
  m_readResourceSizes.resize(m_remesherSetupInfo.readResourcesMaxCount);
  m_readResourceInfos.resize(m_remesherSetupInfo.readResourcesMaxCount);


  std::vector<gpu::ResourceAllocInfo> scratchPersistentAllocs(m_remesherSetupInfo.scratchPersistentCount);
  gpu::PersistentResourceInfo         persistent;
  persistent.scratchPersistentCount  = m_remesherSetupInfo.scratchPersistentCount;
  persistent.scratchPersistentAllocs = scratchPersistentAllocs.data();
  TEST_SUCCESS(gpu::micromeshGpuRemeshingGetPersistent(m_remesher, &persistent));

  m_scratchPersistentResources.resize(m_remesherSetupInfo.scratchPersistentCount);
  for(uint32_t i = 0; i < m_remesherSetupInfo.scratchPersistentCount; i++)
  {
    m_scratchPersistentResources[i] = allocateRemesherBuffer(scratchPersistentAllocs[i], &context->m_vk->m_resourceAllocator);
  }

  m_pipelineLayouts.resize(m_remesherSetupInfo.pipelineLayoutCount);
  for(uint32_t i = 0; i < m_remesherSetupInfo.pipelineLayoutCount; i++)
  {
    gpu::PipelineLayoutInfo pipeLayoutInfo;
    TEST_SUCCESS(gpu::micromeshGpuRemeshingGetPipelineLayout(m_remesher, i, &pipeLayoutInfo));
    createPipelineLayout(context, pipeLayoutInfo, i);
  }

  m_pipelines.resize(m_remesherSetupInfo.pipelineCount);
  for(uint32_t i = 0; i < m_remesherSetupInfo.pipelineCount; i++)
  {
    gpu::PipelineInfo pipeInfo;
    TEST_SUCCESS(gpu::micromeshGpuRemeshingGetPipeline(m_remesher, i, &pipeInfo));
    createPipeline(context, pipeInfo, i);
  }


  m_taskData.readData.resources         = m_readResourceInfos.data();
  m_taskData.readData.resourceDataSizes = m_readResourceSizes.data();
  m_taskData.readData.resourceDatas     = m_readResourceDatas.data();


  m_taskData.allResourceHandles.resize(gpu::eGpuRemeshingScratchStart + m_remesherSetupInfo.scratchPersistentCount
                                       + m_remesherSetupInfo.scratchTaskCount);

  for(uint32_t i = 0; i < m_remesherSetupInfo.scratchPersistentCount; i++)
  {
    m_taskData.allResourceHandles[i + gpu::eGpuRemeshingScratchStart] = {m_scratchPersistentResources[i].buffer,
                                                                         scratchPersistentAllocs[i].buffer.size};
  }

  m_vertexCopy.addBufferBinding(eGpuRemeshingMeshVertexHashBuffer);

  m_vertexCopy.addBufferBinding(eModifiedVertexPositionNormalBuffer);
  m_vertexCopy.addBufferBinding(eModifiedVertexTangentSpaceBuffer);
  m_vertexCopy.addBufferBinding(eModifiedVertexTexcoordBuffer);
  m_vertexCopy.addBufferBinding(eModifiedVertexDirectionsBuffer);
  m_vertexCopy.addBufferBinding(eModifiedVertexDirectionBoundsBuffer);
  m_vertexCopy.addBufferBinding(eModifiedVertexImportanceBuffer);

  m_vertexCopy.setCode(ctx->m_device, (void*)vertex_copy_comp, sizeof(vertex_copy_comp));
  m_vertexCopy.finalizePipeline(ctx->m_device);


  //m_vertexMerge.addBufferBinding(eGpuRemeshingMeshVertexPositionsBuffer);
  m_vertexMerge.addBufferBinding(eGpuRemeshingMeshVertexHashBuffer);
  m_vertexMerge.addBufferBinding(eGpuRemeshingMeshVertexMergeBuffer);
  m_vertexMerge.addBufferBinding(eGpuRemeshingCurrentStateBuffer);

  m_vertexMerge.addBufferBinding(eModifiedVertexPositionNormalBuffer);
  m_vertexMerge.addBufferBinding(eModifiedVertexTangentSpaceBuffer);
  m_vertexMerge.addBufferBinding(eModifiedVertexTexcoordBuffer);
  m_vertexMerge.addBufferBinding(eModifiedVertexDirectionsBuffer);
  m_vertexMerge.addBufferBinding(eModifiedVertexDirectionBoundsBuffer);
  m_vertexMerge.addBufferBinding(eModifiedVertexImportanceBuffer);


  m_vertexMerge.setCode(ctx->m_device, (void*)vertex_merge_comp, sizeof(vertex_merge_comp));
  m_vertexMerge.finalizePipeline(ctx->m_device);

  return true;
}


bool RemeshingOperator_c::destroy(Context context)
{
  nvvk::Context* ctx = getContext(context);
  if(!ctx)
  {
    LOGE("Could not destroy remesher - Vulkan context unavailable\n");
    return false;
  }

  m_vertexCopy.destroy(ctx->m_device);

  m_vertexMerge.destroy(ctx->m_device);

  for(size_t i = 0; i < m_pipelines.size(); i++)
  {
    vkDestroyPipeline(ctx->m_device, m_pipelines[i], nullptr);
  }
  for(size_t i = 0; i < m_pipelineLayouts.size(); i++)
  {
    vkDestroyPipelineLayout(ctx->m_device, m_pipelineLayouts[i].layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx->m_device, m_pipelineLayouts[i].descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(ctx->m_device, m_pipelineLayouts[i].descriptorPool, nullptr);
  }
  if(Result::eSuccess != (gpu::micromeshGpuRemeshingDestroy(m_remesher)))
  {
    LOGE("Could not destroy remesher - micromeshGpuRemeshingDestroy failed\n");
    return false;
  }


  return true;
}


bool RemeshingOperator_c::beginRemeshTask(Context context)
{
  std::vector<gpu::ResourceAllocInfo> scratchTaskResources(m_remesherSetupInfo.scratchTaskCount);
  m_output.scratchTaskCount  = m_remesherSetupInfo.scratchTaskCount;
  m_output.scratchTaskAllocs = scratchTaskResources.data();

  m_taskData.hostReadBuffers.clear();
  m_taskData.hostReadBuffers.clear();

  for(auto& d : m_readResourceDatas)
    d = nullptr;

  TEST_SUCCESS(gpu::micromeshGpuRemeshingBeginTask(m_remesher, &m_remesherParams, &m_input, &m_output, &m_task));

  // prepare task specific scratch resources
  m_taskData.scratchTaskResources.resize(m_output.scratchTaskCount);
  assert(m_remesherSetupInfo.scratchTaskCount == m_output.scratchTaskCount);
  for(uint32_t i = 0; i < m_remesherSetupInfo.scratchTaskCount; i++)
  {
    // allocate
    m_taskData.scratchTaskResources[i] = allocateRemesherBuffer(m_output.scratchTaskAllocs[i], &context->m_vk->m_resourceAllocator);
    // update task table for easier resolving
    m_taskData.allResourceHandles[i + gpu::eGpuRemeshingScratchStart + m_remesherSetupInfo.scratchPersistentCount] = {
        m_taskData.scratchTaskResources[i].buffer, scratchTaskResources[i].buffer.size};
  }


  m_seq.previousReadData = nullptr;
  m_seq.userData         = &m_taskData;
  m_taskData.sysData     = this;

  auto fnCommandGenerator = [](gpu::CommandType cmdType, const void* cmdData, void* userData) {
    TaskData* data = reinterpret_cast<TaskData*>(userData);
    auto      ctx  = getContext(data->context);
    switch(cmdType)
    {
      case gpu::CommandType::eBindPipeline: {
        const auto* bindPipeline = reinterpret_cast<const gpu::CmdBindPipeline*>(cmdData);
        VkPipeline  pipeline     = data->sysData->m_pipelines[bindPipeline->pipelineIndex];
        vkCmdBindPipeline(data->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      }
      break;
      case gpu::CommandType::eBindUserPipeline: {
        const auto* bindUserPipeline = reinterpret_cast<const gpu::CmdBindUserPipeline<gpu::GpuRemeshingUserPipeline>*>(cmdData);
        if(bindUserPipeline->userPipelineEnum == gpu::eGpuRemeshingUserMergeVertices)
        {
          data->sysData->m_vertexMerge.bind(data->cmd);
          VertexMergeConstants vmc;
          vmc.useDirection =
              (data->sysData->m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit);
          vmc.useNormal = (data->sysData->m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexNormalBit);
          vmc.useTangent = (data->sysData->m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexTangentBit);
          vmc.useTexCoord = (data->sysData->m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexTexcoordBit);
          vmc.texcoordIndex = data->sysData->m_heightmapTextureCoord;
          vmc.texcoordCount = data->sysData->m_texcoordCount;

          vmc.fitToOriginalSurface = data->sysData->m_remesherParams.fitToOriginalSurface ? 1u : 0u;

          vkCmdPushConstants(data->cmd, data->sysData->m_vertexMerge.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(VertexMergeConstants), &vmc);
        }
      }
      break;
      case gpu::CommandType::eBindResources: {
        const auto* bindResources = reinterpret_cast<const gpu::CmdBindResources<gpu::GpuRemeshingResource>*>(cmdData);

        std::vector<VkDescriptorBufferInfo> bi;
        for(uint32_t i = 0; i < bindResources->resourceCount; i++)
        {
          bi.emplace_back(VkDescriptorBufferInfo{data->allResourceHandles[bindResources->resources[i].resourceIndex].deviceBuffer,
                                                 0, VK_WHOLE_SIZE});
        }

        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorSet dset = data->sysData->m_pipelineLayouts[bindResources->pipelineLayoutIndex].getNextDescriptorSet();
        for(uint32_t i = 0; i < bindResources->resourceCount; i++)
        {
          VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
          writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_MAX_ENUM;
          writeSet.descriptorCount      = 1;
          writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          writeSet.dstBinding           = bindResources->resources[i].resourceIndex;
          writeSet.dstSet               = dset;
          writeSet.dstArrayElement      = 0;
          writeSet.pBufferInfo          = &bi[i];
          writes.emplace_back(writeSet);
        }
        vkUpdateDescriptorSets(ctx->m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);


        vkCmdBindDescriptorSets(data->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                data->sysData->m_pipelineLayouts[bindResources->pipelineLayoutIndex].layout, 0, 1,
                                &dset, 0, nullptr);

        // prepare descriptor set and bind it
        // there will be maximum of setup.descriptorSetAllocationInfo.setMaxCount
        // many eBindResources per sequence.
      }
      break;
      case gpu::CommandType::eClearResources: {
        const auto* clearResources = reinterpret_cast<const gpu::CmdClearResources<gpu::GpuRemeshingResource>*>(cmdData);
        for(uint32_t i = 0; i < clearResources->resourceCount; i++)
        {
          vkCmdFillBuffer(data->cmd, data->allResourceHandles[clearResources->resources[i].resourceIndex].deviceBuffer,
                          0, VK_WHOLE_SIZE, clearResources->clearValue);
        }
      }
      break;
      case gpu::CommandType::eReadResources: {
        const auto* readResources = reinterpret_cast<const gpu::CmdReadResources<gpu::GpuRemeshingResource>*>(cmdData);
        for(uint32_t i = 0; i < readResources->resourceCount; i++)
        {
          auto& h = data->allResourceHandles[readResources->resources[i].resourceIndex];
          if(h.hostVisibleBuffer.buffer == VK_NULL_HANDLE)
          {
            data->localTaskResources.push_back(data->context->m_vk->m_resourceAllocator.createBuffer(
                h.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
            h.hostVisibleBuffer = data->localTaskResources.back();
          }
          VkBufferCopy c{0ull, 0ull, h.size};
          vkCmdCopyBuffer(data->cmd, h.deviceBuffer, h.hostVisibleBuffer.buffer, 1, &c);
          data->readData.resources[i] = readResources->resources[i];
        }
        data->hadRead                = true;
        data->readData.resourceCount = readResources->resourceCount;
      }

      break;
      case gpu::CommandType::eGlobalConstants: {
        //const auto* globalConstant = reinterpret_cast<const gpu::CmdGlobalConstants*>(cmdData);
        // FIXME : to implement, how? - Not needed by current remesher anyway
      }
      break;
      case gpu::CommandType::eLocalConstants: {
        const auto* localConstant = reinterpret_cast<const gpu::CmdLocalConstants*>(cmdData);
        vkCmdPushConstants(data->cmd, data->sysData->m_pipelineLayouts[localConstant->pipelineLayoutIndex].layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, localConstant->byteSize, localConstant->data);
      }
      break;
      case gpu::CommandType::eBarrier: {
        // FIXME: take into account all combinations
        //if (barrier->readBits & gpu::BarrierBits::eBarrierBufferBit)
        {
          VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
          mb.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
          vkCmdPipelineBarrier(data->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0 /*VK_DEPENDENCY_DEVICE_GROUP_BIT*/, 1, &mb, 0, nullptr, 0, nullptr);
        }
      }
      break;
      case gpu::CommandType::eDispatch: {
        const auto* dispatch = reinterpret_cast<const gpu::CmdDispatch*>(cmdData);
        vkCmdDispatch(data->cmd, dispatch->gridX, dispatch->gridY, dispatch->gridZ);
      }
      break;
      case gpu::CommandType::eDispatchIndirect: {
        const auto* dispatch = reinterpret_cast<const gpu::CmdDispatchIndirect<uint32_t>*>(cmdData);

        vkCmdDispatchIndirect(data->cmd, data->allResourceHandles[dispatch->indirectBuffer.resourceIndex].deviceBuffer,
                              dispatch->indirectBufferOffset);
      }
      break;
      case gpu::CommandType::eBeginLabel: {
        const auto* label = reinterpret_cast<const gpu::CmdBeginLabel*>(cmdData);
        VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label->labelName, {1.0f, 1.0f, 1.0f, 1.0f}};
        vkCmdBeginDebugUtilsLabelEXT(data->cmd, &s);
      }
      break;
      case gpu::CommandType::eEndLabel: {
        vkCmdEndDebugUtilsLabelEXT(data->cmd);
      }
      break;
    }
  };
  m_seq.pfnGenerateGpuCommand = fnCommandGenerator;
  return true;
}

bool RemeshingOperator_c::endRemeshTask(Context context)
{
  auto ctx = getContext(context);
  vkQueueWaitIdle(ctx->m_queueC);

  TEST_SUCCESS(gpu::micromeshGpuRemeshingEndTask(m_remesher, m_task, &m_output));

  assert(static_cast<size_t>(m_remesherSetupInfo.scratchTaskCount) == m_taskData.scratchTaskResources.size());
  for(uint32_t i = 0; i < m_remesherSetupInfo.scratchTaskCount; i++)
  {
    context->m_vk->m_resourceAllocator.destroy(m_taskData.scratchTaskResources[i]);
  }

  for(uint32_t i = 0; i < m_taskData.localTaskResources.size(); i++)
  {
    context->m_vk->m_resourceAllocator.destroy(m_taskData.localTaskResources[i]);
  }

  return true;
}


void RemeshingOperator_c::createRemesherResources(Context context, const OpRemesh_input& input, OpRemesh_modified& modified, DeviceMesh modifiedMesh)
{
  auto ctx = getContext(context);

  m_texcoordCount = modifiedMesh->getDeviceMeshVk()->vertexTexcoordCount;
  //m_texcoordIndex = input.importanceTextureCoord;

  VkPhysicalDeviceMemoryProperties2 memProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  VkPhysicalDeviceMemoryBudgetPropertiesEXT memBudgetProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  memProps.pNext = &memBudgetProps;
  vkGetPhysicalDeviceMemoryProperties2(ctx->m_physicalDevice, &memProps);
  size_t   maxDeviceLocalHeapSize = 0ull;
  uint32_t maxHeapId              = ~0u;
  for(uint32_t i = 0; i < memProps.memoryProperties.memoryHeapCount; i++)
  {
    if((memProps.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0)
      continue;
    if(maxDeviceLocalHeapSize < memProps.memoryProperties.memoryHeaps[i].size)
    {
      maxDeviceLocalHeapSize = memProps.memoryProperties.memoryHeaps[i].size;
      maxHeapId              = i;
    }
  }


  m_input.maxDisplacementSubdivLevel = 5;
  m_input.meshTriangleCount          = uint32_t(modified.meshView->triangleCount());
  m_input.meshVertexCount            = uint32_t(modified.meshView->vertexCount());
  if(maxHeapId < VK_MAX_MEMORY_HEAPS)
  {
    m_input.deviceMemoryBudgetMegaBytes = uint32_t(memBudgetProps.heapBudget[maxHeapId] / (1024ull * 1024ull));
  }
  else
  {
    m_input.deviceMemoryBudgetMegaBytes = 0;
  }


  VkBufferUsageFlags commonUsageFlags =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexPositionsBuffer] = {
      modifiedMesh->getDeviceMeshVk()->vertexPositionNormalBuffer.buffer, sizeof(float) * 4 * m_input.meshVertexCount};


  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexTexcoordsBuffer] = {
      modifiedMesh->getDeviceMeshVk()->vertexTexcoordBuffer.buffer, 2 * m_input.meshVertexCount * m_texcoordCount};


  m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
      sizeof(uint32_t) * 2 * m_input.meshVertexCount, commonUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexHashBuffer] = {m_taskData.localTaskResources.back().buffer,
                                                                           sizeof(uint32_t) * 2 * m_input.meshVertexCount};

  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshTrianglesBuffer] = {
      modifiedMesh->getDeviceMeshVk()->triangleVertexIndexBuffer.buffer, sizeof(uint32_t) * 3 * m_input.meshTriangleCount};

  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexImportanceBuffer] = {
      modifiedMesh->getDeviceMeshVk()->vertexImportanceBuffer.buffer, sizeof(uint16_t) * m_input.meshVertexCount};
  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexDirectionsBuffer] = {
      modifiedMesh->getDeviceMeshVk()->vertexDirectionsBuffer.buffer, 4 * sizeof(uint16_t) * m_input.meshVertexCount};

  // FIXME support through options
  //    // 1 x uint per-triangle (e.g. per-triangle component/material assignments etc.)
  //    // (optional `GpuRemeshing_config::useTriangleUserIDs`)
  m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
      1 * sizeof(uint32_t) * m_input.meshTriangleCount, commonUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshTriangleUserIDsBuffer] = {m_taskData.localTaskResources.back().buffer,
                                                                                1 * sizeof(uint32_t) * m_input.meshTriangleCount};

  //    ,
  //
  //
  //    // output buffers
  //    // -------------------------
  //    //
  //    // 1 x uint { uint16 subdivlevel, uint16 edgeflags} per-triangle
  //    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
  //    eGpuRemeshingMeshTriangleSubdivisionInfoBuffer,
  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshTriangleSubdivisionInfoBuffer] = {
      modifiedMesh->getDeviceMeshVk()->triangleAttributesBuffer.buffer, 1 * sizeof(uint32_t) * m_input.meshTriangleCount};
  //    // 2 x float per-vertex
  //    // (optional `OpRemeshing_settings::generateDisplacementInfo`, only in eDecimate mode)
  //    eGpuRemeshingMeshVertexDirectionBoundsBuffer,
  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexDirectionBoundsBuffer] = {
      modifiedMesh->getDeviceMeshVk()->vertexDirectionBoundsBuffer.buffer, 2 * sizeof(float) * m_input.meshVertexCount};

  // intermediate buffers used during process
  // ----------------------------------------
  // 3 x uint per-vertex as below
  // RemeshingVertexMergeInfo {
  //  uint32_t vertexIndexA;
  //  uint32_t vertexIndexB;
  //  float    blendAtoB;
  // }

  m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
      sizeof(uint32_t) * 3 * m_input.meshVertexCount, commonUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexMergeBuffer] = {m_taskData.localTaskResources.back().buffer,
                                                                            sizeof(uint32_t) * 3 * m_input.meshVertexCount};


  m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
      sizeof(uint32_t) * 1 * m_input.meshVertexCount, commonUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  m_taskData.allResourceHandles[gpu::eGpuRemeshingDebugVertexBuffer] = {m_taskData.localTaskResources.back().buffer,
                                                                        sizeof(uint32_t) * 1 * m_input.meshVertexCount};

  m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
      sizeof(uint32_t) * 1 * m_input.meshTriangleCount, commonUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  m_taskData.allResourceHandles[gpu::eGpuRemeshingDebugTriangleBuffer] = {m_taskData.localTaskResources.back().buffer,
                                                                          sizeof(uint32_t) * 1 * m_input.meshTriangleCount};

  m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
      sizeof(RemeshingCurrentState), commonUsageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

  m_taskData.allResourceHandles[gpu::eGpuRemeshingCurrentStateBuffer] = {m_taskData.localTaskResources.back().buffer,
                                                                         sizeof(RemeshingCurrentState)};

  {
    m_vertexCopy.updateBufferBinding(eGpuRemeshingMeshVertexHashBuffer,
                                     m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexHashBuffer].deviceBuffer);

    m_vertexCopy.updateBufferBinding(eModifiedVertexPositionNormalBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexPositionNormalBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexTangentSpaceBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexTangentSpaceBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexTexcoordBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexTexcoordBuffer.buffer);


    m_vertexCopy.updateBufferBinding(eModifiedVertexPositionNormalBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexPositionNormalBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexTangentSpaceBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexTangentSpaceBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexTexcoordBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexTexcoordBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexDirectionsBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexDirectionsBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexDirectionBoundsBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexDirectionBoundsBuffer.buffer);
    m_vertexCopy.updateBufferBinding(eModifiedVertexImportanceBuffer,
                                     modifiedMesh->getDeviceMeshVk()->vertexImportanceBuffer.buffer);


    m_vertexMerge.updateBufferBinding(eGpuRemeshingMeshVertexHashBuffer,
                                      m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexHashBuffer].deviceBuffer);
    m_vertexMerge.updateBufferBinding(eGpuRemeshingMeshVertexMergeBuffer,
                                      m_taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexMergeBuffer].deviceBuffer);
    m_vertexMerge.updateBufferBinding(eGpuRemeshingCurrentStateBuffer,
                                      m_taskData.allResourceHandles[gpu::eGpuRemeshingCurrentStateBuffer].deviceBuffer);

    m_vertexMerge.updateBufferBinding(eModifiedVertexPositionNormalBuffer,
                                      modifiedMesh->getDeviceMeshVk()->vertexPositionNormalBuffer.buffer);
    m_vertexMerge.updateBufferBinding(eModifiedVertexTangentSpaceBuffer,
                                      modifiedMesh->getDeviceMeshVk()->vertexTangentSpaceBuffer.buffer);
    m_vertexMerge.updateBufferBinding(eModifiedVertexTexcoordBuffer,
                                      modifiedMesh->getDeviceMeshVk()->vertexTexcoordBuffer.buffer);
    m_vertexMerge.updateBufferBinding(eModifiedVertexDirectionsBuffer,
                                      modifiedMesh->getDeviceMeshVk()->vertexDirectionsBuffer.buffer);
    m_vertexMerge.updateBufferBinding(eModifiedVertexDirectionBoundsBuffer,
                                      modifiedMesh->getDeviceMeshVk()->vertexDirectionBoundsBuffer.buffer);
    m_vertexMerge.updateBufferBinding(eModifiedVertexImportanceBuffer,
                                      modifiedMesh->getDeviceMeshVk()->vertexImportanceBuffer.buffer);
  }
}

void RemeshingOperator_c::freeRemesherBuffers(Context context)
{
  auto ctx = getContext(context);
  vkQueueWaitIdle(ctx->m_queueC);

  for(size_t i = 0; i < m_taskData.localTaskResources.size(); i++)
  {
    context->m_vk->m_resourceAllocator.destroy(m_taskData.localTaskResources[i]);
  }
}


void RemeshingOperator_c::copyMeshToRemesher(VkCommandBuffer cmd, const OpRemesh_input& inputs)
{

  VertexCopyConstants vcc;
  vcc.itemCount     = m_input.meshVertexCount;
  vcc.texcoordCount = m_texcoordCount;

  vcc.texcoordIndex = inputs.heightmapTextureCoord;
  vcc.useDirection  = (m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit);
  vcc.useNormal     = (m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexNormalBit);
  vcc.useTangent    = (m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexTangentBit);
  vcc.useTexCoord   = (m_preservedAttributes & meshops::MeshAttributeFlagBits::eMeshAttributeVertexTexcoordBit);

  m_vertexCopy.dispatchThreads(cmd, vcc.itemCount, &vcc, nvvk::DispatcherBarrier::eNone);

  // Copy from input to modified has already been done externally
  //TriangleCopyConstants tcc;
  //tcc.itemCount = m_input.meshTriangleCount * 3;
  //m_triangleCopy.dispatchThreads(cmd, tcc.itemCount, &tcc, nvvk::DispatcherBarrier::eCompute);
}


micromesh::Result RemeshingOperator_c::remesh(Context               context,
                                              const OpRemesh_input& input,
                                              OpRemesh_modified&    modified,
                                              DeviceMesh            modifiedMesh,
                                              uint32_t*             outputTriangleCount,
                                              uint32_t*             outputVertexCount)
{
  *outputTriangleCount = 0u;
  *outputVertexCount   = 0u;

  m_remesherParams.clampDecimationLevel   = input.maxSubdivLevel;
  m_remesherParams.dispMapResolution.x    = input.heightmapTextureWidth;
  m_remesherParams.dispMapResolution.y    = input.heightmapTextureHeight;
  m_remesherParams.errorThreshold         = input.errorThreshold;
  m_remesherParams.fitToOriginalSurface   = input.fitToOriginalSurface;
  m_remesherParams.generateMicromeshInfo  = input.generateMicromeshInfo;
  m_remesherParams.maxTriangleCount       = input.maxOutputTriangleCount;
  m_remesherParams.maxVertexImportance    = input.importanceThreshold;
  m_remesherParams.maxVertexValence       = input.maxVertexValence;
  m_remesherParams.mode                   = RemeshingMode::eDecimate;
  m_remesherParams.texcoordCount          = 1;
  m_remesherParams.texcoordIndex          = input.heightmapTextureCoord;
  m_remesherParams.vertexImportanceWeight = input.importanceWeight;
  m_remesherParams.directionBoundsFactor  = input.directionBoundsFactor;


  m_heightmapTextureCoord = input.heightmapTextureCoord;

  m_preservedAttributes = input.preservedVertexAttributeFlags;
  auto ctx              = getContext(context);
  createRemesherResources(context, input, modified, modifiedMesh);
  bool     done                           = false;
  bool     first                          = true;

  beginRemeshTask(context);
  nvh::Stopwatch    sw;
  nvvk::CommandPool cmdPool(ctx->m_device, ctx->m_queueC);

  Result result = Result::eContinue;

  float    progress       = 0.f;
  uint32_t iterationIndex = 0u;

  nvh::Stopwatch timer;
  while(!done)
  {
    VkCommandBuffer cmd = cmdPool.createCommandBuffer();

    if(first)
    {
      first                  = false;
      m_seq.previousReadData = nullptr;
      copyMeshToRemesher(cmd, input);
    }

    m_taskData.cmd     = cmd;
    m_taskData.context = context;

    m_taskData.hadRead = false;

    result = gpu::micromeshGpuRemeshingContinueTask(m_remesher, m_task, &m_seq);

    if(result == Result::eSuccess || result == Result::eContinue)
    {
      // submit generated command buffer
      if(m_taskData.hadRead)
      {
        for(uint32_t i = 0; i < m_taskData.readData.resourceCount; i++)
        {

          auto& h               = m_taskData.allResourceHandles[m_taskData.readData.resources[i].resourceIndex];
          bool  isFirstReadback = false;

          if(m_taskData.readData.resourceDatas[i] == nullptr)
          {
            isFirstReadback = true;
            m_taskData.hostReadBuffers.push_back({});
            m_taskData.hostReadBuffers.back().resize(h.size);
            m_taskData.readData.resourceDatas[i]     = m_taskData.hostReadBuffers.back().data();
            m_taskData.readData.resourceDataSizes[i] = h.size;
          }

          // At first readback return a buffer filled with 0
          if(isFirstReadback)
          {
            memset(m_taskData.readData.resourceDatas[i], 0u, h.size);
          }
          else
          {
            if(h.hostVisibleBuffer.buffer == VK_NULL_HANDLE)
            {
              m_taskData.localTaskResources.push_back(context->m_vk->m_resourceAllocator.createBuffer(
                  h.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
              h.hostVisibleBuffer = m_taskData.localTaskResources.back();
            }

            // The obtained data will always be 1 frame late
            auto* d = reinterpret_cast<uint8_t*>(context->m_vk->m_resourceAllocator.map(h.hostVisibleBuffer));
            memcpy(m_taskData.readData.resourceDatas[i], d, h.size);
            if(m_taskData.readData.resources[i].resourceEnum == gpu::eGpuRemeshingCurrentStateBuffer)
            {
              memcpy(&m_currentState, d, h.size);
            }

            context->m_vk->m_resourceAllocator.unmap(h.hostVisibleBuffer);
          }
        }

        // readback and setup for next
        m_seq.previousReadData = &m_taskData.readData;
      }
      else
      {
        m_seq.previousReadData = nullptr;
      }
      if(result == Result::eSuccess || (result == Result::eContinue && input.progressiveRemeshing))
      {
        done = true;
      }
    }
    else
    {
      LOGE("Failed to remesh\n");
      endRemeshTask(context);  // free resources
      return result;
    }
    cmdPool.submitAndWait(cmd);
    float currentProgress = 0.f;
    if(m_currentState.triangleCount > 0)
      currentProgress = static_cast<float>(modified.meshView->triangleCount() - m_currentState.triangleCount)
                        / static_cast<float>(modified.meshView->triangleCount() - m_remesherParams.maxTriangleCount);
    if(m_remesherParams.maxTriangleCount > 0 && iterationIndex == 0)
    {
      LOGI("Remeshing started %d -> %d triangles max\n", static_cast<int32_t>(modified.meshView->triangleCount()),
           static_cast<int32_t>(m_remesherParams.maxTriangleCount));
    }

    if(m_remesherParams.maxTriangleCount > 0 && currentProgress - progress > 0.05f)
    {
      progress = currentProgress;
      if(m_currentState.triangleCount > 0)
      {
        LOGI("Remeshing in progress %d -> %d triangles - %.1f%% (%.2f ms)\n",
             static_cast<int32_t>(modified.meshView->triangleCount()),
             static_cast<int32_t>(m_currentState.triangleCount), std::min(99.f, progress * 99.f), timer.elapsed());
      }
    }

    if(m_remesherParams.maxTriangleCount < 0 && iterationIndex % 50U == 0U)
    {
      if(m_currentState.triangleCount > 0)
      {
        LOGI("Remeshing in progress %d -> %d triangles (%.2f ms)\n", static_cast<int32_t>(modified.meshView->triangleCount()),
             static_cast<int32_t>(m_currentState.triangleCount), timer.elapsed());
      }
    }


    iterationIndex++;
  }
  endRemeshTask(context);
  *outputTriangleCount = m_currentState.triangleCount;
  *outputVertexCount   = m_currentState.vertexCount;

  return result;
}


VkDescriptorType toDescriptorType(gpu::DescriptorType t)
{
  switch(t)
  {
    case gpu::DescriptorType::eBufferRead:
    case gpu::DescriptorType::eBufferReadWrite:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case gpu::DescriptorType::eConstantBuffer:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    default:
      LOGE("Unsupported descriptor type %u - defaulting to storage buffer\n", unsigned(t));
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
}

VkShaderStageFlags toShaderStages(uint32_t pipelineTypesUsed)
{
  VkShaderStageFlags result = 0;
  if(pipelineTypesUsed & (1 << gpu::ShaderType::eShaderCompute))
    result |= VK_SHADER_STAGE_COMPUTE_BIT;
  // No other shader type supported for now by gpu::ShaderType
  return result;
}


void RemeshingOperator_c::createPipelineLayout(Context context, const micromesh::gpu::PipelineLayoutInfo& info, size_t index)
{
  auto            ctx            = getContext(context);
  PipelineLayout& pipelineLayout = m_pipelineLayouts[index];
  pipelineLayout                 = {};

  for(size_t i = 0; i < info.descriptorRangeCount; i++)
  {
    pipelineLayout.bindings.addBinding(VkDescriptorSetLayoutBinding{
        info.descriptorRanges[i].baseRegisterIndex, toDescriptorType(info.descriptorRanges[i].descriptorType),
        info.descriptorRanges[i].descriptorCount, toShaderStages(info.pipelineTypesUsed)});
  }


  pipelineLayout.descriptorSetLayout = pipelineLayout.bindings.createLayout(ctx->m_device);
  pipelineLayout.descriptorPool =
      pipelineLayout.bindings.createPool(ctx->m_device, uint32_t(pipelineLayout.descriptorSets.size()));
  for(size_t i = 0; i < pipelineLayout.descriptorSets.size(); i++)
  {
    pipelineLayout.descriptorSets[i] =
        nvvk::allocateDescriptorSet(ctx->m_device, pipelineLayout.descriptorPool, pipelineLayout.descriptorSetLayout);
  }

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pSetLayouts    = &pipelineLayout.descriptorSetLayout;
  pipelineLayoutCreateInfo.setLayoutCount = 1;


  VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, info.localPushConstantSize};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstantRange;

  VkResult r = vkCreatePipelineLayout(ctx->m_device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout.layout);

  if(r != VK_SUCCESS || pipelineLayout.layout == VK_NULL_HANDLE)
    assert(0 && "Could not create pipeline layout");
}


VkPipeline createSinglePipeline(VkDevice device, VkPipelineLayout layout, const void* shaderCode, uint32_t codeSize)
{
  VkPipelineShaderStageCreateInfo stageCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
  stageCreateInfo.pName                           = "main";

  VkShaderModuleCreateInfo moduleCreateInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  moduleCreateInfo.codeSize = codeSize;
  moduleCreateInfo.pCode    = reinterpret_cast<const uint32_t*>(shaderCode);
  VkShaderModule computeShaderModule;
  VkResult       r = vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &computeShaderModule);
  if(r != VK_SUCCESS || computeShaderModule == VK_NULL_HANDLE)
    assert(0 && "Could not create shader module");


  stageCreateInfo.module = computeShaderModule;

  VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  createInfo.stage  = stageCreateInfo;
  createInfo.layout = layout;

  VkPipeline pipeline;

  r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline);
  if(r != VK_SUCCESS || pipeline == VK_NULL_HANDLE)
    assert(0 && "Could not create pipeline");

  vkDestroyShaderModule(device, computeShaderModule, nullptr);
  return pipeline;
}

void RemeshingOperator_c::createPipeline(Context context, const gpu::PipelineInfo& info, size_t index)
{
  auto ctx = getContext(context);
  if(info.pipelineLayoutIndex >= m_pipelineLayouts.size())
  {
    LOGE("Trying to access invalid pipeline layout index %zu (max: %zu)\n", index, m_pipelineLayouts.size());
    return;
  }

  if(gpu::PipelineType::eCompute != info.type)
  {
    LOGE("Only compute pipelines supported\n");
    return;
  }

  VkPipelineLayout layout = m_pipelineLayouts[info.pipelineLayoutIndex].layout;
  if(info.sourceCount != 1)
  {
    LOGE("Unsupported multiple sources for a single shader\n");
    return;
  }

  const gpu::ShaderCode& code = info.sources[0];
  if(code.codeType != gpu::ShaderCodeType::eShaderCodeSPIRV)
  {
    LOGE("Unsupported shader code type - only SPIR-V is supported\n");
    return;
  }
  VkPipeline p       = createSinglePipeline(ctx->m_device, layout, code.data, uint32_t(code.size));
  m_pipelines[index] = p;
}


}  // namespace meshops
