/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once
#include <micromesh/micromesh_displacement_remeshing.h>

#ifdef __cplusplus  // GLSL Type
#include <glm/glm.hpp>
using namespace glm;
#endif
#include "shaders/generate_importance_host_device.h"
#include "micromesh/micromesh_gpu.h"


#include "nvvk/images_vk.hpp"
#include "nvh/timesampler.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "vk_mem_alloc.h"
#include "nvvk/descriptorsets_vk.hpp"

#include "nvvk/compute_vk.hpp"
#include "meshops/meshops_operations.h"
#include <array>

namespace meshops {

class GenerateImportanceOperator_c
{
public:
  bool create(Context context);
  bool destroy(Context context);


  bool generateImportance(Context context, size_t inputCount, OpGenerateImportance_modified* inputs);

private:
  bool m_isInitialized{false};

  nvvk::PushComputeDispatcher<GenerateImportanceConstants, GenerateImportanceBindings> m_generateImportance;

  nvvk::Image   m_dummyMap;
  nvvk::Texture m_dummyTex;

  GenerateImportanceConstants m_constants;
};
}  // namespace meshops