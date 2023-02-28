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

#include <meshops_internal/meshops_context.h>
#include <nvh/alignment.hpp>
#include <cstddef>

namespace meshops {

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsContextCreate(const ContextConfig& config, Context* pContext)
{
  *pContext = new Context_c(config, {});
  return micromesh::Result::eSuccess;
}

MESHOPS_API void MESHOPS_CALL meshopsContextDestroy(Context context)
{
  delete context;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsContextGetConfig(Context context, ContextConfig* config)
{
  if(context != nullptr)
  {
    *config = context->m_config;
    return micromesh::Result::eSuccess;
  }
  return micromesh::Result::eInvalidValue;
}

MESHOPS_API micromesh::Result meshopsContextCreateVK(const ContextConfig& config, const ContextVK& sharedContextVK, Context* pContext)
{
  *pContext = new Context_c(config, sharedContextVK);
  return micromesh::Result::eSuccess;
}

MESHOPS_API ContextVK* MESHOPS_CALL meshopsContextGetVK(Context context)
{
  assert(context->m_vk);
  return context->m_vk ? &context->m_vk->m_ptrs : nullptr;
}

MESHOPS_API void MESHOPS_CALL meshopsGetContextRequirements(const ContextConfig&     config,
                                                            nvvk::ContextCreateInfo& createInfo,
                                                            std::vector<uint8_t>&    createInfoData)
{
  createInfo.setVersion(1, 3);  // Using Vulkan 1.3

  struct RequiredFeatureStructs
  {
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
#ifdef _DEBUG
    VkValidationFeaturesEXT      validationFeatures{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    VkValidationFeatureEnableEXT validationFeatureEnables[1]{VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};

    //VkValidationFeatureEnableEXT validationFeatureDisables[1];
#endif  // _DEBUG
  };

  RequiredFeatureStructs* features;
  {
    assert(createInfoData.empty());
    createInfoData.resize(sizeof(RequiredFeatureStructs) + alignof(RequiredFeatureStructs));
    features = reinterpret_cast<RequiredFeatureStructs*>(
        nvh::align_up(reinterpret_cast<uintptr_t>(createInfoData.data()), alignof(RequiredFeatureStructs)));
    *features = RequiredFeatureStructs();
  }

  createInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &features->accelFeature);
  createInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &features->rtPipelineFeature);
  createInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &features->rayQueryFeatures);
  createInfo.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  createInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  createInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  createInfo.addDeviceExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

  createInfo.addDeviceExtension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, false, &features->floatFeatures);


#ifdef _DEBUG
  // #debug_printf
  createInfo.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  features->validationFeatures.enabledValidationFeatureCount = uint32_t(std::size(features->validationFeatureEnables));
  features->validationFeatures.pEnabledValidationFeatures    = features->validationFeatureEnables;
  //features->validationFeatures.disabledValidationFeatureCount = std::size(features->validationFeatureDisables);
  //features->validationFeatures.pDisabledValidationFeatures    = features->validationFeatureDisables;
  createInfo.instanceCreateInfoExt = &features->validationFeatures;
#endif  // _DEBUG
}

}  // namespace meshops
