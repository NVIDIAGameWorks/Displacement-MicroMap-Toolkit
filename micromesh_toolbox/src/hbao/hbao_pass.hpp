/*
*  Copyright (c) 2014-2023, NVIDIA CORPORATION.All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met :
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and / or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
*  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
*  PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR
*  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
*  EXEMPLARY, OR CONSEQUENTIAL DAMAGES( INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
*  PROFITS; OR BUSINESS INTERRUPTION ) HOWEVER CAUSED AND ON ANY THEORY
*  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  ( INCLUDING NEGLIGENCE OR OTHERWISE ) ARISING IN ANY WAY OUT OF THE USE
*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef HBAOPASS_H__
#define HBAOPASS_H__

#include <vulkan/vulkan_core.h>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvh/trangeallocator.hpp>

#include <assert.h>
#include <nvmath/nvmath.h>

//////////////////////////////////////////////////////////////////////////

/// HbaoSystem implements a screen-space
/// ambient occlusion effect using
/// horizon-based ambient occlusion.
/// See https://github.com/nvpro-samples/gl_ssao
/// for more details

class HbaoPass
{
public:
  static const int RANDOM_SIZE     = 4;
  static const int RANDOM_ELEMENTS = RANDOM_SIZE * RANDOM_SIZE;

  struct Config
  {
    VkFormat targetFormat;
    uint32_t maxFrames;
  };
  HbaoPass(VkDevice device, nvvk::ResourceAllocator* allocator, const Config& config)
  {
    init(device, allocator, nullptr, config);
  }
  ~HbaoPass() { deinit(); }

  void init(VkDevice device, nvvk::ResourceAllocator* allocator, nvvk::ShaderModuleManager* shaderManager, const Config& config);
  void reloadShaders();
  void deinit();

  struct FrameConfig
  {
    bool blend;

    uint32_t sourceWidthScale;
    uint32_t sourceHeightScale;

    uint32_t targetWidth;
    uint32_t targetHeight;

    VkDescriptorImageInfo sourceDepth;
    VkDescriptorImageInfo targetColor;
  };

  struct FrameIMGs
  {
    nvvk::Texture depthlinear, viewnormal, result, blur, resultarray, deptharray;
  };

  struct Frame
  {
    uint32_t slot = ~0u;

    FrameIMGs images;
    int       width;
    int       height;

    FrameConfig config;
  };

  bool initFrame(Frame& frame, const FrameConfig& config, VkCommandBuffer cmd);
  void deinitFrame(Frame& frame);


  struct View
  {
    bool          isOrtho;
    float         nearPlane;
    float         farPlane;
    float         halfFovyTan;
    nvmath::mat4f projectionMatrix;
  };

  struct Settings
  {
    View view;

    float unit2viewspace = 1.0f;
    float intensity      = 1.0f;
    float radius         = 1.0f;
    float bias           = 0.1f;
    float blurSharpness  = 40.0f;
  };

  // before: must do appropriate barriers for color write access and depth read access
  // after:  from compute write to whatever output image needs
  void cmdCompute(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;

private:
  struct Shaders
  {
    //nvvk::ShaderModuleID depth_linearize, viewnormal, blur, blur_apply, deinterleave, calc, reinterleave;
    VkShaderModule depth_linearize, viewnormal, blur, blur_apply, deinterleave, calc, reinterleave;
  };

  struct Pipelines
  {
    VkPipeline depth_linearize = VK_NULL_HANDLE;
    VkPipeline viewnormal      = VK_NULL_HANDLE;
    VkPipeline blur            = VK_NULL_HANDLE;
    VkPipeline blur_apply      = VK_NULL_HANDLE;
    VkPipeline deinterleave    = VK_NULL_HANDLE;
    VkPipeline calc            = VK_NULL_HANDLE;
    VkPipeline reinterleave    = VK_NULL_HANDLE;
  };

  VkDevice                   m_device;
  nvvk::ResourceAllocator*   m_allocator;
  nvvk::ShaderModuleManager* m_shaderManager;
  nvh::TRangeAllocator<1>    m_slots;

  nvvk::DescriptorSetContainer m_setup;

  nvvk::Buffer           m_ubo;
  VkDescriptorBufferInfo m_uboInfo;

  VkSampler m_linearSampler;

  Shaders   m_shaders;
  Pipelines m_pipelines;

  nvmath::vec4f m_hbaoRandom[RANDOM_ELEMENTS];

  void           updatePipelines();
  void           updateUbo(VkCommandBuffer cmd, const Frame& frame, const Settings& settings) const;
  VkShaderModule createShaderModule(const std::vector<uint32_t>& code);
};

#endif
