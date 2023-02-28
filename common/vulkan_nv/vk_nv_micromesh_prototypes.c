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
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//
// WARNING: VK_NV_displacement_micromap is in beta and subject to future changes.
//          Do not use these headers in production code.
//
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include "vk_nv_micromesh_prototypes.h"

#ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap

// for name in test:gmatch("[%w_]+") do print("static "..name.." pfn"..name:sub(4,-1).." = 0;") end

#ifndef VK_NO_PROTOTYPES
static PFN_vkCreateMicromapEXT                 pfn_vkCreateMicromapEXT                 = 0;
static PFN_vkDestroyMicromapEXT                pfn_vkDestroyMicromapEXT                = 0;
static PFN_vkCmdBuildMicromapsEXT              pfn_vkCmdBuildMicromapsEXT              = 0;
static PFN_vkBuildMicromapsEXT                 pfn_vkBuildMicromapsEXT                 = 0;
static PFN_vkCopyMicromapEXT                   pfn_vkCopyMicromapEXT                   = 0;
static PFN_vkCopyMicromapToMemoryEXT           pfn_vkCopyMicromapToMemoryEXT           = 0;
static PFN_vkCopyMemoryToMicromapEXT           pfn_vkCopyMemoryToMicromapEXT           = 0;
static PFN_vkWriteMicromapsPropertiesEXT       pfn_vkWriteMicromapsPropertiesEXT       = 0;
static PFN_vkCmdCopyMicromapEXT                pfn_vkCmdCopyMicromapEXT                = 0;
static PFN_vkCmdCopyMicromapToMemoryEXT        pfn_vkCmdCopyMicromapToMemoryEXT        = 0;
static PFN_vkCmdCopyMemoryToMicromapEXT        pfn_vkCmdCopyMemoryToMicromapEXT        = 0;
static PFN_vkCmdWriteMicromapsPropertiesEXT    pfn_vkCmdWriteMicromapsPropertiesEXT    = 0;
static PFN_vkGetDeviceMicromapCompatibilityEXT pfn_vkGetDeviceMicromapCompatibilityEXT = 0;
static PFN_vkGetMicromapBuildSizesEXT          pfn_vkGetMicromapBuildSizesEXT          = 0;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateMicromapEXT(VkDevice                       device,
                                                   const VkMicromapCreateInfoEXT* pCreateInfo,
                                                   const VkAllocationCallbacks*   pAllocator,
                                                   VkMicromapEXT*                 pMicromap)
{
  return pfn_vkCreateMicromapEXT(device, pCreateInfo, pAllocator, pMicromap);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyMicromapEXT(VkDevice device, VkMicromapEXT Micromap, const VkAllocationCallbacks* pAllocator)
{
  pfn_vkDestroyMicromapEXT(device, Micromap, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vkCmdBuildMicromapsEXT(VkCommandBuffer commandBuffer, uint32_t infoCount, const VkMicromapBuildInfoEXT* pInfos)
{
  pfn_vkCmdBuildMicromapsEXT(commandBuffer, infoCount, pInfos);
}

VKAPI_ATTR VkResult VKAPI_CALL vkBuildMicromapsEXT(VkDevice                      device,
                                                   VkDeferredOperationKHR        deferredOperation,
                                                   uint32_t                      infoCount,
                                                   const VkMicromapBuildInfoEXT* pInfos)
{
  return pfn_vkBuildMicromapsEXT(device, deferredOperation, infoCount, pInfos);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCopyMicromapEXT(VkDevice device, VkDeferredOperationKHR deferredOperation, const VkCopyMicromapInfoEXT* pInfo)
{
  return pfn_vkCopyMicromapEXT(device, deferredOperation, pInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCopyMicromapToMemoryEXT(VkDevice                             device,
                                                         VkDeferredOperationKHR               deferredOperation,
                                                         const VkCopyMicromapToMemoryInfoEXT* pInfo)
{
  return pfn_vkCopyMicromapToMemoryEXT(device, deferredOperation, pInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCopyMemoryToMicromapEXT(VkDevice                             device,
                                                         VkDeferredOperationKHR               deferredOperation,
                                                         const VkCopyMemoryToMicromapInfoEXT* pInfo)
{
  return pfn_vkCopyMemoryToMicromapEXT(device, deferredOperation, pInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL vkWriteMicromapsPropertiesEXT(VkDevice             device,
                                                             uint32_t             MicromapCount,
                                                             const VkMicromapEXT* pMicromaps,
                                                             VkQueryType          queryType,
                                                             size_t               dataSize,
                                                             void*                pData,
                                                             size_t               stride)
{
  return pfn_vkWriteMicromapsPropertiesEXT(device, MicromapCount, pMicromaps, queryType, dataSize, pData, stride);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyMicromapEXT(VkCommandBuffer commandBuffer, const VkCopyMicromapInfoEXT* pInfo)
{
  pfn_vkCmdCopyMicromapEXT(commandBuffer, pInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyMicromapToMemoryEXT(VkCommandBuffer commandBuffer, const VkCopyMicromapToMemoryInfoEXT* pInfo)
{
  pfn_vkCmdCopyMicromapToMemoryEXT(commandBuffer, pInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyMemoryToMicromapEXT(VkCommandBuffer commandBuffer, const VkCopyMemoryToMicromapInfoEXT* pInfo)
{
  pfn_vkCmdCopyMemoryToMicromapEXT(commandBuffer, pInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdWriteMicromapsPropertiesEXT(VkCommandBuffer      commandBuffer,
                                                            uint32_t             MicromapCount,
                                                            const VkMicromapEXT* pMicromaps,
                                                            VkQueryType          queryType,
                                                            VkQueryPool          queryPool,
                                                            uint32_t             firstQuery)
{
  pfn_vkCmdWriteMicromapsPropertiesEXT(commandBuffer, MicromapCount, pMicromaps, queryType, queryPool, firstQuery);
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceMicromapCompatibilityEXT(VkDevice                                 device,
                                                               const VkMicromapVersionInfoEXT*          pVersionInfo,
                                                               VkAccelerationStructureCompatibilityKHR* pCompatibility)
{
  pfn_vkGetDeviceMicromapCompatibilityEXT(device, pVersionInfo, pCompatibility);
}

VKAPI_ATTR void VKAPI_CALL vkGetMicromapBuildSizesEXT(VkDevice                            device,
                                                      VkAccelerationStructureBuildTypeKHR buildType,
                                                      const VkMicromapBuildInfoEXT*       pBuildInfo,
                                                      VkMicromapBuildSizesInfoEXT*        pSizeInfo)
{
  pfn_vkGetMicromapBuildSizesEXT(device, buildType, pBuildInfo, pSizeInfo);
}

void load_VK_EXT_opacity_micromap_prototypes(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr)
{
  // for name in test:gmatch("[%w_]+") do print("    pfn"..name:sub(4,-1).." = ("..name..')getDeviceProcAddr(device, "'..name:sub(5,-1)..'");') end
  pfn_vkCreateMicromapEXT       = (PFN_vkCreateMicromapEXT)getDeviceProcAddr(device, "vkCreateMicromapEXT");
  pfn_vkDestroyMicromapEXT      = (PFN_vkDestroyMicromapEXT)getDeviceProcAddr(device, "vkDestroyMicromapEXT");
  pfn_vkCmdBuildMicromapsEXT    = (PFN_vkCmdBuildMicromapsEXT)getDeviceProcAddr(device, "vkCmdBuildMicromapsEXT");
  pfn_vkBuildMicromapsEXT       = (PFN_vkBuildMicromapsEXT)getDeviceProcAddr(device, "vkBuildMicromapsEXT");
  pfn_vkCopyMicromapEXT         = (PFN_vkCopyMicromapEXT)getDeviceProcAddr(device, "vkCopyMicromapEXT");
  pfn_vkCopyMicromapToMemoryEXT = (PFN_vkCopyMicromapToMemoryEXT)getDeviceProcAddr(device, "vkCopyMicromapToMemoryEXT");
  pfn_vkCopyMemoryToMicromapEXT = (PFN_vkCopyMemoryToMicromapEXT)getDeviceProcAddr(device, "vkCopyMemoryToMicromapEXT");
  pfn_vkWriteMicromapsPropertiesEXT =
      (PFN_vkWriteMicromapsPropertiesEXT)getDeviceProcAddr(device, "vkWriteMicromapsPropertiesEXT");
  pfn_vkCmdCopyMicromapEXT = (PFN_vkCmdCopyMicromapEXT)getDeviceProcAddr(device, "vkCmdCopyMicromapEXT");
  pfn_vkCmdCopyMicromapToMemoryEXT =
      (PFN_vkCmdCopyMicromapToMemoryEXT)getDeviceProcAddr(device, "vkCmdCopyMicromapToMemoryEXT");
  pfn_vkCmdCopyMemoryToMicromapEXT =
      (PFN_vkCmdCopyMemoryToMicromapEXT)getDeviceProcAddr(device, "vkCmdCopyMemoryToMicromapEXT");
  pfn_vkCmdWriteMicromapsPropertiesEXT =
      (PFN_vkCmdWriteMicromapsPropertiesEXT)getDeviceProcAddr(device, "vkCmdWriteMicromapsPropertiesEXT");
  pfn_vkGetDeviceMicromapCompatibilityEXT =
      (PFN_vkGetDeviceMicromapCompatibilityEXT)getDeviceProcAddr(device, "vkGetDeviceMicromapCompatibilityEXT");
  pfn_vkGetMicromapBuildSizesEXT =
      (PFN_vkGetMicromapBuildSizesEXT)getDeviceProcAddr(device, "vkGetMicromapBuildSizesEXT");
  // for name in test:gmatch("[%w_]+") do print("    assert(pfn"..name:sub(4,-1)..");") end
  assert(pfn_vkCreateMicromapEXT);
  assert(pfn_vkDestroyMicromapEXT);
  assert(pfn_vkCmdBuildMicromapsEXT);
  assert(pfn_vkBuildMicromapsEXT);
  assert(pfn_vkCopyMicromapEXT);
  assert(pfn_vkCopyMicromapToMemoryEXT);
  assert(pfn_vkCopyMemoryToMicromapEXT);
  assert(pfn_vkWriteMicromapsPropertiesEXT);
  assert(pfn_vkCmdCopyMicromapEXT);
  assert(pfn_vkCmdCopyMicromapToMemoryEXT);
  assert(pfn_vkCmdCopyMemoryToMicromapEXT);
  assert(pfn_vkCmdWriteMicromapsPropertiesEXT);
  assert(pfn_vkGetDeviceMicromapCompatibilityEXT);
  assert(pfn_vkGetMicromapBuildSizesEXT);
}
#else
void load_VK_EXT_opacity_micromap_functions(VK_EXT_opacity_micromap_functions* fns, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr)
{
  // for name in test:gmatch("[%w_]+") do print("    fns->"..name:sub(4,-1).." = ("..name..')getDeviceProcAddr(device, "'..name:sub(5,-1)..'");') end
  fns->pfn_vkCreateMicromapEXT    = (PFN_vkCreateMicromapEXT)getDeviceProcAddr(device, "vkCreateMicromapEXT");
  fns->pfn_vkDestroyMicromapEXT   = (PFN_vkDestroyMicromapEXT)getDeviceProcAddr(device, "vkDestroyMicromapEXT");
  fns->pfn_vkCmdBuildMicromapsEXT = (PFN_vkCmdBuildMicromapsEXT)getDeviceProcAddr(device, "vkCmdBuildMicromapsEXT");

  fns->pfn_vkBuildMicromapsEXT = (PFN_vkBuildMicromapsEXT)getDeviceProcAddr(device, "vkBuildMicromapsEXT");
  fns->pfn_vkCopyMicromapEXT   = (PFN_vkCopyMicromapEXT)getDeviceProcAddr(device, "vkCopyMicromapEXT");
  fns->pfn_vkCopyMicromapToMemoryEXT =
      (PFN_vkCopyMicromapToMemoryEXT)getDeviceProcAddr(device, "vkCopyMicromapToMemoryEXT");
  fns->pfn_vkCopyMemoryToMicromapEXT =
      (PFN_vkCopyMemoryToMicromapEXT)getDeviceProcAddr(device, "vkCopyMemoryToMicromapEXT");
  fns->pfn_vkWriteMicromapsPropertiesEXT =
      (PFN_vkWriteMicromapsPropertiesEXT)getDeviceProcAddr(device, "vkWriteMicromapsPropertiesEXT");
  fns->pfn_vkCmdCopyMicromapEXT = (PFN_vkCmdCopyMicromapEXT)getDeviceProcAddr(device, "vkCmdCopyMicromapEXT");
  fns->pfn_vkCmdCopyMicromapToMemoryEXT =
      (PFN_vkCmdCopyMicromapToMemoryEXT)getDeviceProcAddr(device, "vkCmdCopyMicromapToMemoryEXT");
  fns->pfn_vkCmdCopyMemoryToMicromapEXT =
      (PFN_vkCmdCopyMemoryToMicromapEXT)getDeviceProcAddr(device, "vkCmdCopyMemoryToMicromapEXT");
  fns->pfn_vkCmdWriteMicromapsPropertiesEXT =
      (PFN_vkCmdWriteMicromapsPropertiesEXT)getDeviceProcAddr(device, "vkCmdWriteMicromapsPropertiesEXT");
  fns->pfn_vkGetDeviceMicromapCompatibilityEXT =
      (PFN_vkGetDeviceMicromapCompatibilityEXT)getDeviceProcAddr(device, "vkGetDeviceMicromapCompatibilityEXT");
  fns->pfn_vkGetMicromapBuildSizesEXT =
      (PFN_vkGetMicromapBuildSizesEXT)getDeviceProcAddr(device, "vkGetMicromapBuildSizesEXT");
  // for name in test:gmatch("[%w_]+") do print("    assert(fns->"..name:sub(4,-1)..");") end
  assert(fns->pfn_vkCreateMicromapEXT);
  assert(fns->pfn_vkDestroyMicromapEXT);
  assert(fns->pfn_vkCmdBuildMicromapsEXT);
  assert(fns->pfn_vkBuildMicromapsEXT);
  assert(fns->pfn_vkCopyMicromapEXT);
  assert(fns->pfn_vkCopyMicromapToMemoryEXT);
  assert(fns->pfn_vkCopyMemoryToMicromapEXT);
  assert(fns->pfn_vkWriteMicromapsPropertiesEXT);
  assert(fns->pfn_vkCmdCopyMicromapEXT);
  assert(fns->pfn_vkCmdCopyMicromapToMemoryEXT);
  assert(fns->pfn_vkCmdCopyMemoryToMicromapEXT);
  assert(fns->pfn_vkCmdWriteMicromapsPropertiesEXT);
  assert(fns->pfn_vkGetDeviceMicromapCompatibilityEXT);
  assert(fns->pfn_vkGetMicromapBuildSizesEXT);
}
#endif

#endif  // #ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap
