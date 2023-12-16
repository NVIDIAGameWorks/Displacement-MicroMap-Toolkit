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

#pragma once

#include <meshops/meshops_vk.h>
#include <micromesh/micromesh_operations.h>
#include <memory>
#include <cstdarg>
#include <cerrno>
#include <system_error>
#include <nvvk/context_vk.hpp>
#include <nvvk/memallocator_vma_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>

namespace meshops {

class ContextVKData
{
public:
  ContextVKData(const ContextConfig& config, const ContextVK& sharedContextVK)
      : m_ptrs(sharedContextVK)
  {
    if(!m_ptrs.context)
    {
      nvvk::ContextCreateInfo createInfo;
      std::vector<uint8_t>    blob;
      meshopsGetContextRequirements(config, createInfo, blob);
      m_ownedCtx = std::make_unique<nvvk::Context>();
      m_ownedCtx->init(createInfo);
      m_ptrs.context = m_ownedCtx.get();

      // Enable glsl debugPrintfEXT(fmt, ...) in debug builds
#ifdef _DEBUG
      // Vulkan message callback - for receiving the printf in the shader
      // Note: there is already a callback in nvvk::Context, but by defaut it is not printing INFO severity
      //       this callback will catch the message and will make it clean for display.
      auto dbg_messenger_callback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                       VkDebugUtilsMessageTypeFlagsEXT        messageType,
                                       const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData) -> VkBool32 {
        // Get rid of all the extra message we don't need
        std::string clean_msg = callbackData->pMessage;
        clean_msg             = clean_msg.substr(clean_msg.find_last_of('|') + 2);
        nvprintfLevel(LOGLEVEL_DEBUG, "Debug: %s", clean_msg.c_str());  // <- This will end up in the Logger
        return VK_FALSE;                                                // to continue
      };

      // #debug_printf : Creating the callback
      VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
      dbg_messenger_create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
      dbg_messenger_create_info.pfnUserCallback = dbg_messenger_callback;
      NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(m_ownedCtx->m_instance, &dbg_messenger_create_info, nullptr, &m_dbgMessenger));
#endif
    }

    // Fill in the optional queue overrides with queues on the nvvk::Context if
    // they're not set. Meshops should always use these instead.
    if(m_ptrs.queueGCT.queue == VK_NULL_HANDLE)
    {
      m_ptrs.queueGCT = m_ptrs.context->m_queueGCT;
    }
    if(m_ptrs.queueT.queue == VK_NULL_HANDLE)
    {
      m_ptrs.queueT = m_ptrs.context->m_queueT;
    }
    if(m_ptrs.queueC.queue == VK_NULL_HANDLE)
    {
      m_ptrs.queueC = m_ptrs.context->m_queueC;
    }

    if(!m_ptrs.vma)
    {
      VmaAllocatorCreateInfo allocatorInfo = {};
      allocatorInfo.physicalDevice         = m_ptrs.context->m_physicalDevice;
      allocatorInfo.device                 = m_ptrs.context->m_device;
      allocatorInfo.instance               = m_ptrs.context->m_instance;
      allocatorInfo.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
      vmaCreateAllocator(&allocatorInfo, &m_ownedVMA);
      m_ptrs.vma = m_ownedVMA;
    }

    {
      m_vmaMemoryAllocator.init(m_ptrs.context->m_device, m_ptrs.context->m_physicalDevice, (VmaAllocator)m_ptrs.vma);
      m_resourceAllocator.init(m_ptrs.context->m_device, m_ptrs.context->m_physicalDevice, &m_vmaMemoryAllocator);

      m_ptrs.resAllocator = &m_resourceAllocator;
    }

    {
      m_cmdPoolGCT.init(m_ptrs.context->m_device, m_ptrs.context->m_queueGCT.familyIndex,
                        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_ptrs.context->m_queueGCT.queue);
    }
  }

  ~ContextVKData()
  {
    m_resourceAllocator.deinit();
    m_vmaMemoryAllocator.deinit();
    m_cmdPoolGCT.deinit();

    if(m_ownedVMA)
    {
      vmaDestroyAllocator(m_ownedVMA);
    }
    if(m_ownedCtx)
    {
#ifdef _DEBUG
      if(m_dbgMessenger)
      {
        vkDestroyDebugUtilsMessengerEXT(m_ownedCtx->m_instance, m_dbgMessenger, nullptr);
      }
#endif
      m_ownedCtx->deinit();
    }
  }

  std::unique_ptr<nvvk::Context> m_ownedCtx;
  VkDebugUtilsMessengerEXT       m_dbgMessenger = VK_NULL_HANDLE;
  VmaAllocator                   m_ownedVMA = nullptr;
  nvvk::VMAMemoryAllocator       m_vmaMemoryAllocator;
  nvvk::ResourceAllocator        m_resourceAllocator;
  nvvk::CommandPool              m_cmdPoolGCT;
  ContextVK                      m_ptrs;

  // creates a primary command buffer
  VkCommandBuffer createTempCmdBufferGCT() { return m_cmdPoolGCT.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY); }

  // also calls vkEndCommandBuffer
  void submitAndWaitGCT(VkCommandBuffer cmd) { m_cmdPoolGCT.submitAndWait(cmd); }
};

class Context_c
{
public:
  Context_c(ContextConfig config, const ContextVK& sharedContextVK)
      : m_config(config)
      , m_micromeshContext(micromesh::OpConfig{micromesh::OpContextType::eImmediateAutomaticThreading, m_config.threadCount},
                           m_config.messageCallback)
      , m_vk(sharedContextVK.context || m_config.requiresDeviceContext ? std::make_unique<ContextVKData>(config, sharedContextVK) : nullptr)
  {
    if(m_vk)
    {
      m_vkDevice = m_vk->m_ptrs.context->m_device;
    }
  }
  ~Context_c() {}
  void message(const micromesh::MessageSeverity severity,
#ifdef _MSC_VER
               _Printf_format_string_
#endif
               const char* fmt,
               ...)
  {
    std::va_list vlist;
    va_start(vlist, fmt);
    const std::string msg = vformat(fmt, vlist);
    va_end(vlist);
    if(m_config.messageCallback.pfnCallback != nullptr)
      m_config.messageCallback.pfnCallback(severity, msg.c_str(), 0, m_config.messageCallback.userData);
  }

  ContextConfig                  m_config;
  micromesh::ScopedOpContext     m_micromeshContext;
  std::unique_ptr<ContextVKData> m_vk;
  VkDevice                       m_vkDevice = VK_NULL_HANDLE;

private:
  inline std::string vformat(const char* fmt, va_list vlist)
  {
    va_list vlistCopy;
    va_copy(vlistCopy, vlist);
    int len = std::vsnprintf(nullptr, 0, fmt, vlistCopy);
    va_end(vlistCopy);

    assert(len >= 0);
    if(len < 0)
    {
      const auto err = errno;
      const auto ec  = std::error_code(err, std::generic_category());
      throw std::system_error(ec);
    }

    std::string s(static_cast<size_t>(len) + 1, '\0');
    std::vsnprintf(s.data(), s.size(), fmt, vlist);
    s.resize(static_cast<size_t>(len));
    return s;
  }
};

// let's put the simple operators here until the grow

class TopologyOperator_c
{
  uint32_t dummy;
};

class SubdivisionLevelOperator_c
{
  uint32_t dummy;
};

class DisplacementMicromapOperator_c
{
  uint32_t dummy;
};

}  // namespace meshops
#define MESHOPS_LOGI(ctx, msg, ...)                                                                                    \
  (ctx)->message(micromesh::MessageSeverity::eInfo, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define MESHOPS_LOGW(ctx, msg, ...)                                                                                    \
  (ctx)->message(micromesh::MessageSeverity::eWarning, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__);
#define MESHOPS_LOGE(ctx, msg, ...)                                                                                    \
  (ctx)->message(micromesh::MessageSeverity::eError, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__)

// Functions for reducing lines of code when checking if arguments are null.
#define MESHOPS_CHECK_NONNULL(ctx, argument)                                                                           \
  {                                                                                                                    \
    if(!(argument))                                                                                                    \
    {                                                                                                                  \
      MESHOPS_LOGE((ctx), "Non-optional argument `" #argument "` was null.");                                          \
      return micromesh::Result::eInvalidValue;                                                                         \
    }                                                                                                                  \
  }
#define MESHOPS_CHECK_CTX_NONNULL(ctx)                                                                                 \
  if(!(ctx))                                                                                                           \
  {                                                                                                                    \
    return micromesh::Result::eInvalidValue;                                                                           \
  };
