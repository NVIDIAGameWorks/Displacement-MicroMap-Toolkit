/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "micromesh_compressed_vk.hpp"

namespace microdisp {

class MicroSplitParts;

class MicromeshSubTriangleDecoderVK
{
public:
  MicromeshSubTriangleDecoderVK(MicromeshSetCompressedVK& microSet)
      : m_micro(microSet)
  {
  }

  void init(ResourcesVK&             res,
            const bary::ContentView& bary,
            uint8_t*                 decimateEdgeFlags,
            uint32_t                 maxSubdivLevel,
            bool                     useBaseTriangles,
            bool                     withAttributes,
            uint32_t                 numThreads);

private:
  friend class MicromeshMicroTriangleDecoderVK;

  MicromeshSetCompressedVK& m_micro;

  void uploadMicroSubTriangles(nvvk::StagingMemoryManager* staging,
                               VkCommandBuffer             cmd,
                               const bary::ContentView&    bary,
                               uint8_t*                    decimateEdgeFlags,
                               uint32_t                    maxSubdivLevel,
                               uint32_t                    numThreads);

  void uploadMicroBaseTriangles(nvvk::StagingMemoryManager* staging,
                                VkCommandBuffer             cmd,
                                const bary::ContentView&    bary,
                                uint8_t*                    decimateEdgeFlags,
                                uint32_t                    maxSubdivLevel,
                                uint32_t                    numThreads);

  void uploadDescends(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits);

  void uploadVertices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits);
};

}  // namespace microdisp
