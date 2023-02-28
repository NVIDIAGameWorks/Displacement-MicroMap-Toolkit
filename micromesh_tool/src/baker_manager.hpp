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
// The meshops_bake BakerManager schedules texture resampling operations,
// and moves textures between different storage locations during baking.

#pragma once

#include <list>
#include <stdint.h>
#include <unordered_set>

#include <nvvk/resourceallocator_vk.hpp>

#include "nvvk/context_vk.hpp"
#include "tool_bake.hpp"

#include "tool_scene.hpp"
#include "tool_meshops_objects.hpp"
#include "pullpush_filter.hpp"

namespace tool_bake {

using GPUTextureContainer = micromesh_tool::GPUTextureContainer;

constexpr VkFormat RESAMPLE_COLOR_FORMAT      = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat RESAMPLE_DISTANCE_FORMAT   = VK_FORMAT_R32_SFLOAT;
constexpr VkFormat RESAMPLE_QUATERNION_FORMAT = RESAMPLE_COLOR_FORMAT;
constexpr VkFormat RESAMPLE_OFFSET_FORMAT     = VK_FORMAT_R16G16B16A16_UNORM;
constexpr VkFormat RESAMPLE_HEIGHT_FORMAT     = VK_FORMAT_R16_UNORM;

// Represents an index into one of the BakerManager's vectors.
struct GPUTextureIndex
{
  enum class VectorIndex
  {
    eInput,
    eOutput,
    eOutputAux,
    eDistance
  } vec{};
  size_t idx = 0;

  struct Hash
  {
    size_t operator()(const GPUTextureIndex& i) const { return std::hash<uint64_t>()(uint64_t(i.vec) << 62 | i.idx); }
  };

  struct Equal
  {
    bool operator()(const GPUTextureIndex& a, const GPUTextureIndex& b) const
    {
      return (a.vec == b.vec) && (a.idx == b.idx);
    }
  };

  using UnorderedSet = std::unordered_set<GPUTextureIndex, GPUTextureIndex::Hash, GPUTextureIndex::Equal>;
};

constexpr size_t NoInputIndex         = ~size_t(0);
constexpr size_t OutputAuxIndex       = ~size_t(0);
constexpr size_t InvalidDistanceIndex = ~size_t(0);

// This is used to encode ResampleTextureContainer more compactly.
struct ResampleInstruction
{
  meshops::TextureType texelContent = meshops::TextureType::eGeneric;
  size_t inputIndex = NoInputIndex;  // Into m_resamplingInputStorage; may be NoInputIndex if texelContent is not eGeneric
  size_t outputIndex = OutputAuxIndex;  // Into m_resamplingOutputStorage unless OutputAuxIndex, which redirects to outputAuxIndex
  size_t outputAuxIndex{};              // Into m_resamplingOutputStorageAux
  size_t distanceIndex = InvalidDistanceIndex;  // Into m_resamplingDistanceStorage
};

struct FinalUse
{
  GPUTextureIndex index;
  // When filling empty spaces in normal and quaternion textures, we should
  // normalize interpolated texels. We can't do this if parts of the texture
  // contain different things, though.
  bool onlyContainsNormals     = false;
  bool onlyContainsQuaternions = false;
};

// The BakerManager emits one of these "frames" of resampling info per mesh.
struct ResampleMeshInstructions
{
  // A set of resampling instructions to perform from the hi-res to
  // the lo-res mesh.
  std::vector<ResampleInstruction> instructions;

  // The set of images we'll write for the last time. This lets us know when
  // we should export information to an image file.
  std::vector<FinalUse> finalUses;
};


// This describes the input and output textures to the resampler. It doesn't
// correspond to a glTF output, since a texture might contain both normal and
// color information in different areas.
struct ResampleTextureContainer
{
  meshops::TextureType texelContent = meshops::TextureType::eGeneric;

  // Highres texture. Might point to the BakerManager's
  // 1x1 (128, 128, 255, 255)/255 texture if the texelContent is not eColor.
  GPUTextureContainer input;

  // Output texture to be filled by sampling from the highres mesh
  GPUTextureContainer output;

  // Distance texture to keep the closest highres mesh hits between passes
  // The op is less than or equal so the same depth texture can be used by multiple textures in separate passes.
  GPUTextureContainer distance;
};

struct BakerManagerConfig
{
  std::string                       outTextureStem;  // Output filename stem for generated textures
  std::vector<ResampleExtraTexture> resampleExtraTextures;
  std::string                       quaternionTexturesStem;
  std::string                       heightTexturesStem;
  std::string                       offsetTexturesStem;
  TexturesToResample                texturesToResample = TexturesToResample::eNone;
  int                               resampleResolution{0};
};

class BakerManager
{
public:
  BakerManager() = delete;
  BakerManager(VkDevice device, nvvk::ResourceAllocator& alloc);
  ~BakerManager() { destroy(); }

  // All functions returning bool return false and print a message on error.

  // Initializes the BakerManager for a hi->lo-res bake. This will store
  // pointers to the hi- and lo-res glTF scenes. It'll also analyze the
  // files and output a vector of sets of resampling instructions, one per
  // hi-res mesh. The hi- and lo-res scenes must have the same number of
  // meshes.
  //
  // Note that this may modify `lowMesh`'s glTF to set up the map from
  // materials to images, and may add new textures (e.g. if the high-res mesh
  // has textures the low-res mesh doesn't, and if we're resampling all
  // textures).
  //
  // Ideally, we'd like to also change the materials. Here's why: Suppose in
  // the lo-res scene, primitives 0 and 1 have the same material. However, in
  // the high-res scene, primitives 0 and 1 have different materials: primitive
  // 0 has albedo in image 0, but primitive 1 has both albedo in image 0 and a
  // normal map in image 1. In the output (when resampling all materials),
  // primitive 0's material should have only resampled image 0, while primitive
  // 1 should have both resampled image 0 and resampled image 1.
  //
  // At the moment, though, I don't think we can quite do that, because earlier
  // parts of the code introduce a dependency on the materials in the low-res
  // mesh. We'll just check for situations like the above and produce an error
  // for now.
  //
  // Since this can modify the lo-res mesh images, this means that lowMesh's
  // materials aren't reliable after this, which isn't fantastic.
  bool generateInstructions(const BakerManagerConfig&                   info,
                            const micromesh_tool::ToolScene*            highMesh,
                            std::unique_ptr<micromesh_tool::ToolScene>& lowMesh,
                            std::vector<ResampleMeshInstructions>&      instructions);
  // Returns the minimum VRAM limit required for the BakerManager to be sure
  // to not run out of VRAM, and the ideal amount of VRAM to let the
  // BakerManager use.
  void getTextureMemoryRequirements(uint64_t&                                    minimumRequiredBytes,
                                    uint64_t&                                    idealBytes,
                                    const std::vector<ResampleMeshInstructions>& instructions) const;
  // The BakerManager will use at most this number of bytes in VRAM.
  // 0 == no limit. (Internally, 0 sets m_memLimit to a large number.)
  void setMemoryLimit(uint64_t limitBytes)
  {
    if(limitBytes == 0)
    {
      m_memLimit = 1ULL << 56;
    }
    else
    {
      m_memLimit = limitBytes;
    }
  }
  // These two functions ensure that textures are loaded into VRAM when the
  // resampler needs them, and that they're all saved to image files once
  // the last mesh has been processed. Outside of a prepare...finish block,
  // the BakerManager can move image data between a number of places; to
  // minimize memory usage; see GPUTextureContainer::Storage for the
  // full list.
  bool prepareTexturesForMesh(nvvk::Context::Queue queueGCT, nvvk::Context::Queue queueT, const ResampleMeshInstructions& meshInstructions);
  void finishTexturesForMesh(nvvk::Context::Queue queueGCT, const ResampleMeshInstructions& meshInstructions);
  // Unpacks a ResampleMeshInstructions object to a set of lower-level resample containers.
  std::vector<ResampleTextureContainer> getResampleTextures(const ResampleMeshInstructions& meshInstructions) const;
  // Destroys all resources.
  void destroy();

private:
  // Returns all the textures used by a particular high/low-res mesh pair.
  GPUTextureIndex::UnorderedSet getTexturesForMesh(const ResampleMeshInstructions& meshInstructions) const;
  // Load all the textures in the given list into VRAM; assumes they aren't
  // already in VRAM.
  bool loadResamplingTextures(nvvk::Context::Queue queueGCT, const std::vector<GPUTextureIndex>& textureIndices);
  // Returns the URI for a newly generated texture.
  std::string getNewTextureURI(const BakerManagerConfig& info, size_t outputTextureIndex) const;
  // Returns the filename (relative to the working directory) that a given
  // texture would have if it was cached.
  std::string getCacheFilename(GPUTextureIndex textureIndex) const;
  // Moves the given textures from VRAM to cache files.
  bool cacheResamplingTexturesToDisk(nvvk::Context::Queue queueT, const std::vector<GPUTextureIndex>& textureIndices);
  // Call this function if we need to exit early and cached files might exist;
  // this will attempt to remove any existing ones so that we don't leave
  // temporary files on the user's file system.
  void deleteCachedFilesEarly();
  // Single interface for getting an element of the m_resampling arrays.
  GPUTextureContainer&       getResamplingTexture(GPUTextureIndex idx);
  const GPUTextureContainer& getResamplingTexture(GPUTextureIndex idx) const;


  // --- Generic
  // default constructor sets this to std::thread::hardware_concurrency()
  uint32_t m_numThreads = 1;


  // --- Non-owning pointers
  // Stored from constructor
  VkDevice                 m_device = nullptr;
  nvvk::ResourceAllocator* m_alloc  = nullptr;
  // Stored from `generateInstructions`
  micromesh_tool::ToolScene*       m_lowMesh  = nullptr;
  const micromesh_tool::ToolScene* m_highMesh = nullptr;

  // --- Texture storage
  // One image - possibly null - for each element of the `images` array in the
  // glTF input, plus resampling extra textures.
  std::vector<GPUTextureContainer> m_resamplingInputStorage;
  // One image - possibly null - for each element of the `images` array in the
  // glTF output, plus resampling extra textures.
  std::vector<GPUTextureContainer> m_resamplingOutputStorage;
  // Generated output images that are not linked to the gltf scene.
  std::vector<GPUTextureContainer> m_resamplingDistanceStorage;
  // Extra indexing information: Each output uses only one distance texture,
  // but multiple outputs can use the same one. m_outputToDistanceTextureMap[i]
  // is the index in m_resamplingDistanceStorage of the distance texture used
  // by m_resamplingOutputStorage[i].
  std::vector<size_t> m_outputToDistanceTextureMap;

  // --- Caching
  // Our BakerManager can cache images to disk to save VRAM. This stores a
  // random cache file name prefix to avoid I/O-level collisions with other
  // app instances.
  std::string m_cacheFilePrefix;
  // Stores images that may be in VRAM in the order they were loaded. Could be a ring buffer.
  std::list<GPUTextureIndex> m_textureCacheFIFO;
  // Memory limit in bytes. As we store it, 0 means "0 bytes" instead of "no limit".
  uint64_t m_memLimit = 0;
  // Total of texture memory for all images currently in VRAM.
  uint64_t m_currentTextureMemoryUsage = 0;

  // --- Filtering
  // We use this to fill in empty spaces when exporting textures.
  PullPushFilter m_pullPushFilter;
  // We generate three specializations:
  // - General, for filling without normalization
  // - Normals, which normalizes normals in the RGB channels after filling
  // - Quaternons, which normalizes quaternions in the RGBA channels after filling.
  // Note: We currently create and destroy these each time we initialize the
  // orchestrator. It would be nice if we could do this just once per app run.
  PullPushFilter::Pipes m_pullPushFilterPipesGeneral;
  PullPushFilter::Pipes m_pullPushFilterPipesNormals;
  PullPushFilter::Pipes m_pullPushFilterPipesQuaternions;
};

}  // namespace tool_bake
