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

#pragma once

#include "vk_nv_micromesh.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap
#ifndef VK_NO_PROTOTYPES
void load_VK_EXT_opacity_micromap_prototypes(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
#else
typedef struct VK_EXT_opacity_micromap_functions
{
  PFN_vkCreateMicromapEXT                 pfn_vkCreateMicromapEXT;
  PFN_vkDestroyMicromapEXT                pfn_vkDestroyMicromapEXT;
  PFN_vkCmdBuildMicromapsEXT              pfn_vkCmdBuildMicromapsEXT;
  PFN_vkBuildMicromapsEXT                 pfn_vkBuildMicromapsEXT;
  PFN_vkCopyMicromapEXT                   pfn_vkCopyMicromapEXT;
  PFN_vkCopyMicromapToMemoryEXT           pfn_vkCopyMicromapToMemoryEXT;
  PFN_vkCopyMemoryToMicromapEXT           pfn_vkCopyMemoryToMicromapEXT;
  PFN_vkWriteMicromapsPropertiesEXT       pfn_vkWriteMicromapsPropertiesEXT;
  PFN_vkCmdCopyMicromapEXT                pfn_vkCmdCopyMicromapEXT;
  PFN_vkCmdCopyMicromapToMemoryEXT        pfn_vkCmdCopyMicromapToMemoryEXT;
  PFN_vkCmdCopyMemoryToMicromapEXT        pfn_vkCmdCopyMemoryToMicromapEXT;
  PFN_vkCmdWriteMicromapsPropertiesEXT    pfn_vkCmdWriteMicromapsPropertiesEXT;
  PFN_vkGetDeviceMicromapCompatibilityEXT pfn_vkGetDeviceMicromapCompatibilityEXT;
  PFN_vkGetMicromapBuildSizesEXT          pfn_vkGetMicromapBuildSizesEXT;
} VK_EXT_opacity_micromap_functions;

void load_VK_EXT_opacity_micromap_functions(VK_EXT_opacity_micromap_functions* fns, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
#endif
#else   // ^^^ #ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap
// When the Vulkan SDK provides VK_EXT_opacity_micromap, extensions_vk.cpp loads it for us.
inline void load_VK_EXT_opacity_micromap_prototypes(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr){};
#endif  // #ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap

#ifdef __cplusplus
}
#endif

// there are no extra function prototypes for displacement