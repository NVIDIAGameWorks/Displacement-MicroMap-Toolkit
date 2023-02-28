/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <baryutils/baryutils.h>
#include <nvvk/stagingmemorymanager_vk.hpp>
#include "microdisp_shim.hpp"

namespace microdisp {

struct MicromeshCombinedData;

struct MicromeshSetCompressedVK
{
  // TODO: create these only once
  RBuffer umajor2bmap;

  RBuffer vertices;
  RBuffer descends;
  RBuffer triangleIndices;

  struct MeshData
  {
    // TODO: merge with BaryInfo struct
    RBuffer                binding;
    MicromeshCombinedData* combinedData = nullptr;

    // Either base- or sub-triangle data is used and
    // never both. We kept separate variables
    // for clarity

    RBuffer baseTriangles;
    RBuffer baseSpheres;

    RBuffer subTriangles;
    RBuffer subSpheres;

    RBuffer distances;  // compressed values buffer
    RBuffer mipDistances;

    RBuffer attrNormals;
    RBuffer attrTriangles;

    // just for visualization purposes not for rendering
    RBuffer baseTriangleMinMaxs;

    // either sub or base triangle count
    uint32_t microTriangleCount;

    // TODO: Make another struct, or merge just the following into DeviceMicromap
    //RBuffer baseTriangles;
    //RBuffer baseSpheres;
    //RBuffer baseTriangleMinMaxs;
    // Binding struct will also need a ref to
    //RBuffer distances;
  };

  std::vector<MeshData> meshDatas;

  bool hasBaseTriangles = false;
  bool usedFormats[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024) + 1];

  // creates buffers & uploads typical data that is agnostic of the specific
  // rasterization decoder chosen.
  // see various `micromesh_decoder_...` files for the full init sequence

  void initBasics(ResourcesVK& res, const bary::ContentView& bary, bool useBaseTriangles, bool useMips);

  // creates buffers & uploads micro vertex attribute normals
  void initAttributeNormals(ResourcesVK& res, const bary::ContentView& bary, uint32_t numThreads = 0);

  void deinit(ResourcesVK& res);

  // updates the state of `MeshData::combinedData` to retrieve most buffer addresses and store them
  // in the binding buffer
  void uploadMeshDatasBinding(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd);
};

}  // namespace microdisp