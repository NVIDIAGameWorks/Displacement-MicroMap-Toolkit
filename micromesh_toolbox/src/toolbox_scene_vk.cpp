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

#include <cinttypes>
#include "tiny_gltf.h"
#include "toolbox_scene_vk.hpp"
#include "meshops/meshops_vk.h"
#include "nvh/parallel_work.hpp"
#include "nvh/timesampler.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvk/images_vk.hpp"
#include "shaders/dh_scn_desc.h"
#include "toolbox_version.h"
#include "vulkan_mutex.h"

#if defined(NVP_SUPPORTS_NVML)  // #!584
#include "nvml.h"
#endif

ToolboxSceneVk::ToolboxSceneVk(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, meshops::Context context, nvvk::Context::Queue extraQueue)
    : m_ctx(ctx)
    , m_alloc(alloc)
    , m_context(context)
    , m_qGCT1(extraQueue)
{
  m_dutil = std::make_unique<nvvk::DebugUtil>(ctx->m_device);  // Debug utility
}

ToolboxSceneVk::~ToolboxSceneVk()
{
  // The destructor wasn't called
  assert(m_deviceMeshes.empty());
}

static bool computeDmmExtEnabled(nvvk::Context& ctx, std::string& reason)
{
  if(!ctx.hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME))
  {
    return false;
  }

  // WORKAROUND: Avoid a known crash by disabling micromesh for GPUs older
  // than Ada with these specific drivers.
#if 1  // #!584

  // If the driver is not one of these, enable displacement micromap. Otherwise,
  // go on to check the gpu arch.
  LOGI("Driver version: %u\n", ctx.m_physicalInfo.properties10.driverVersion);
  if(ctx.m_physicalInfo.properties10.driverVersion != 2227896320 && ctx.m_physicalInfo.properties10.driverVersion != 2202780544)
  {
    return true;
  }

  // Complex NVML-based logic to check to avoid errors in the beta driver
#if !defined(NVP_SUPPORTS_NVML)
  reason = "the Toolbox was built without NVML.";
  return false;
#else

  static bool        adaCheckPassed = false;
  static std::string adaCheckReason;
  static bool        adaCheckStarted = false;  // Ensures we only run this code once
  if(adaCheckStarted)
  {
    reason = adaCheckReason;
    return adaCheckPassed;
  }

  adaCheckStarted = true;
  // This doesn't interfere with the NVML monitor because nvmlInit()
  // and nvmlShutdown() count the number of times they have been called.
  struct ScopedNvml
  {
    bool valid = false;
    ScopedNvml() { valid = (NVML_SUCCESS == nvmlInit()); }
    ~ScopedNvml()
    {
      if(valid)
        nvmlShutdown();
    }
  } scopedNvml = ScopedNvml();

  if(!scopedNvml.valid)
  {
    reason = adaCheckReason = "nvmlInit() failed.";
    return (adaCheckPassed = false);
  }

  unsigned int physicalGpuCount = 0;
  if(NVML_SUCCESS != nvmlDeviceGetCount(&physicalGpuCount))
  {
    reason = adaCheckReason = "nvmlDeviceGetCount() failed.";
    return (adaCheckPassed = false);
  }

  for(unsigned int i = 0; i < physicalGpuCount; i++)
  {
    nvmlDevice_t device{};
    if(NVML_SUCCESS != nvmlDeviceGetHandleByIndex(i, &device))
      continue;

    std::array<char, 96> name{};
    if(NVML_SUCCESS != nvmlDeviceGetName(device, name.data(), (unsigned int)name.size()))
      continue;

    if(strcmp(name.data(), ctx.m_physicalInfo.properties10.deviceName) == 0)
    {
      // This is the device we're rendering with! Is it an Ada Lovelace or newer GPU?
      nvmlDeviceArchitecture_t architecture = 0;
      if(NVML_SUCCESS != nvmlDeviceGetArchitecture(device, &architecture))
      {
        reason = adaCheckReason = "nvmlDeviceGetArchitecture() failed.";
        return (adaCheckPassed = false);
      }

      adaCheckPassed = (architecture > NVML_DEVICE_ARCH_AMPERE);
      if(!adaCheckPassed)
      {
        reason = adaCheckReason = std::string(
            "not enabled because of a known issue in the first beta driver with pre-Ada GPUs and "
            "version " MICROMESH_TOOLBOX_VERSION_STRING
            " of the Toolbox. The dmm_displacement sample will ray trace correctly on this GPU, however, and "
            "the Toolbox will ray trace on Ada GPUs correctly.");
      }
      return adaCheckPassed;
    }
  }
  reason = adaCheckReason = "the GPU names returned by NVML did not match the Vulkan GPU names.";
  return (adaCheckPassed = false);
#endif
#endif  // #!584
}

//--------------------------------------------------------------------------------------------------
// Create all Vulkan resources to hold a nvvkhl::Scene
//
void ToolboxSceneVk::create(VkCommandBuffer cmd, micromesh_tool::ToolScene& scn)
{
  destroy();  // Make sure not to leave allocated buffers

  m_hasDisplacementMicromeshExt = computeDmmExtEnabled(*m_ctx, m_hasRtxMicromeshReason);

  createMaterialBuffer(cmd, scn);
  createInstanceInfoBuffer(cmd, scn);
  if(!createDeviceMeshBuffer(cmd, scn))
  {
    destroy();  // Makes sure that external code does not need to call destroy() if it is not fully constructed
    return;
  }
  createTextureImages(cmd, scn.textures(), scn.images());
  if(!createDeviceBaryBuffer(cmd, m_qGCT1, scn))
  {
    destroy();  // Makes sure that external code does not need to call destroy() if it is not fully constructed
    return;
  }


  // Buffer references
  SceneDescription scene_desc{};
  scene_desc.materialAddress       = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bMaterial.buffer);
  scene_desc.deviceMeshInfoAddress = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bDeviceMeshInfo.buffer);
  scene_desc.deviceBaryInfoAddress = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bDeviceBaryInfo.buffer);
  scene_desc.instInfoAddress       = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bInstances.buffer);

  auto lock    = GetVkQueueOrAllocatorLock();
  m_bSceneDesc = m_alloc->createBuffer(cmd, sizeof(SceneDescription), &scene_desc,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bSceneDesc.buffer);
}

//--------------------------------------------------------------------------------------------------
// Create a buffer of all materials, with only the elements we need
//
void ToolboxSceneVk::createMaterialBuffer(VkCommandBuffer cmd, const micromesh_tool::ToolScene& scn)
{
  nvh::ScopedTimer _st("- Create Material Buffer");

  // Importing the tinygltf::material in a struct with all value resolved
  nvh::GltfScene scene_materials;
  scene_materials.importMaterials(scn.model());

  // The material on the GPU is slightly different/smaller
  std::vector<GltfShadeMaterial> shade_materials;
  shade_materials.reserve(scene_materials.m_materials.size());

  // Lambda to convert from nvh::GltfMaterial to the GPU version
  auto convertMaterial = [](const nvh::GltfMaterial& m) {
    GltfShadeMaterial s{};
    s.emissiveFactor               = m.emissiveFactor;
    s.emissiveTexture              = m.emissiveTexture;
    s.khrDiffuseFactor             = m.specularGlossiness.diffuseFactor;
    s.khrDiffuseTexture            = m.specularGlossiness.diffuseTexture;
    s.khrSpecularFactor            = m.specularGlossiness.specularFactor;
    s.khrGlossinessFactor          = m.specularGlossiness.glossinessFactor;
    s.khrSpecularGlossinessTexture = m.specularGlossiness.specularGlossinessTexture;
    s.normalTexture                = m.normalTexture;
    s.normalTextureScale           = m.normalTextureScale;
    s.pbrBaseColorFactor           = m.baseColorFactor;
    s.pbrBaseColorTexture          = m.baseColorTexture;
    s.pbrMetallicFactor            = m.metallicFactor;
    s.pbrMetallicRoughnessTexture  = m.metallicRoughnessTexture;
    s.pbrRoughnessFactor           = m.roughnessFactor;
    s.shadingModel                 = m.shadingModel;
    s.alphaMode                    = m.alphaMode;
    s.alphaCutoff                  = m.alphaCutoff;
    s.khrDisplacementTexture       = m.displacement.displacementGeometryTexture;
    s.khrDisplacementFactor        = m.displacement.displacementGeometryFactor;
    s.khrDisplacementOffset        = m.displacement.displacementGeometryOffset;

    return s;
  };

  // Converting all materials
  for(const auto& m : scene_materials.m_materials)
  {
    GltfShadeMaterial s = convertMaterial(m);
    shade_materials.push_back(s);
  }

  // In case the scene material was empty, create a default one
  if(scene_materials.m_materials.empty())
  {
    nvh::GltfMaterial defaultMat;
    GltfShadeMaterial s = convertMaterial(defaultMat);
    shade_materials.push_back(s);
  }

  // Create the buffer of all scene materials
  auto lock   = GetVkQueueOrAllocatorLock();
  m_bMaterial = m_alloc->createBuffer(cmd, shade_materials,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bMaterial.buffer);
}

//--------------------------------------------------------------------------------------------------
// Array of instance information
// - Use by the vertex shader to retrieve the position of the instance
void ToolboxSceneVk::createInstanceInfoBuffer(VkCommandBuffer cmd, const micromesh_tool::ToolScene& scn)
{
  assert(scn.model().scenes.size() > 0);
  nvh::ScopedTimer _st("- Create Instance Buffer");

  const std::vector<micromesh_tool::ToolScene::PrimitiveInstance>& prim_instances = scn.getPrimitiveInstances();

  std::vector<InstanceInfo> inst_info;
  inst_info.reserve(prim_instances.size());
  for(const auto& prim_inst : prim_instances)
  {
    InstanceInfo info{};
    info.objectToWorld = prim_inst.worldMatrix;
    info.worldToObject = nvmath::invert(prim_inst.worldMatrix);
    info.materialID    = prim_inst.material;
    inst_info.push_back(info);
  }

  auto lock = GetVkQueueOrAllocatorLock();
  m_bInstances = m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bInstances.buffer);
}

//--------------------------------------------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
bool ToolboxSceneVk::createDeviceMeshBuffer(VkCommandBuffer cmd, micromesh_tool::ToolScene& scn)
{
  nvh::ScopedTimer _st("- Create Vertex Buffer");

  const std::vector<std::unique_ptr<micromesh_tool::ToolMesh>>& meshes = scn.meshes();

  auto lock = GetVkQueueOrAllocatorLock();
  for(size_t meshIndex = 0; meshIndex < meshes.size(); meshIndex++)
  {
    micromesh_tool::ToolMesh&   mesh     = *meshes[meshIndex];
    meshops::ResizableMeshView& meshView = mesh.view();

    // Create the buffers of the attributes that exist
    meshops::DeviceMeshSettings settings{};
    settings.attribFlags = meshView.getMeshAttributeFlags();
    if(mesh.relations().bary != -1 && !scn.barys().empty())
    {
      // Forcing the direction vector to be present (will use normal if was not provided)
      // as direction vectors are mandatory for displacement
      settings.attribFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit;
      settings.attribFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit;

      // Provide defaults to initialize the device vertexDirectionBounds with
      // the bary group's bias and scale in case ToolMesh has no direction
      // bounds. They should be mutually exclusive.
      const bary::BasicView& basic  = scn.barys()[mesh.relations().bary]->groups()[mesh.relations().group].basic;
      settings.directionBoundsBias  = basic.groups[0].floatBias.r;
      settings.directionBoundsScale = basic.groups[0].floatScale.r;
    }

    meshops::DeviceMesh     d{nullptr};
    const micromesh::Result result = meshops::meshopsDeviceMeshCreate(m_context, meshView, settings, &d);
    if(micromesh::Result::eSuccess != result)
    {
      LOGE("Error: Could not create device mesh %zu\n", meshIndex);
      return false;
    }
    m_deviceMeshes.push_back(d);
  }

  std::vector<DeviceMeshInfo> device_mesh_infos;
  device_mesh_infos.reserve(m_deviceMeshes.size());
  for(auto& device : m_deviceMeshes)
  {
    meshops::DeviceMeshVK* vk = meshops::meshopsDeviceMeshGetVK(device);
    DeviceMeshInfo         info{};
    info.triangleVertexIndexBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->triangleVertexIndexBuffer.buffer);
    info.triangleAttributesBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->triangleAttributesBuffer.buffer);
    info.vertexPositionNormalBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->vertexPositionNormalBuffer.buffer);
    info.vertexTangentSpaceBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->vertexTangentSpaceBuffer.buffer);
    info.vertexTexcoordBuffer     = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->vertexTexcoordBuffer.buffer);
    info.vertexDirectionsBuffer   = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->vertexDirectionsBuffer.buffer);
    info.vertexDirectionBoundsBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->vertexDirectionBoundsBuffer.buffer);
    info.vertexImportanceBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, vk->vertexImportanceBuffer.buffer);
    info.deviceAttribFlags      = vk->deviceAttribFlags;
    info.sourceAttribFlags      = vk->sourceAttribFlags;
    device_mesh_infos.push_back(info);
  }

  auto usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  // Creating the buffer of all device mesh information
  m_bDeviceMeshInfo = m_alloc->createBuffer(cmd, device_mesh_infos, usage_flag);
  m_dutil->DBG_NAME(m_bDeviceMeshInfo.buffer);
  return true;
}


//--------------------------------------------------------------------------------------------------
// Creating the resources holding the Barycentric data for micromeshes
//
bool ToolboxSceneVk::createDeviceBaryBuffer(VkCommandBuffer cmd, nvvk::Context::Queue extraQueue, micromesh_tool::ToolScene& scn)
{
  if(scn.barys().empty())
  {
    return true;
  }

  std::map<std::pair<int32_t, int32_t>, const micromesh_tool::ToolMesh*> baryToMeshMap;
  for(auto& mesh : scn.meshes())
  {
    baryToMeshMap[{mesh->relations().bary, mesh->relations().group}] = mesh.get();
  }

  uint64_t usageFlags = eDeviceMicromeshUsageRasterizingBit;

  if(m_hasDisplacementMicromeshExt)
  {
    usageFlags |= eDeviceMicromeshUsageRaytracingBit;
  }

  std::vector<DeviceBaryInfo> deviceBaryInfos;
  deviceBaryInfos.reserve(m_barys.size());
  for(int32_t baryIndex = 0; baryIndex < static_cast<int32_t>(scn.barys().size()); ++baryIndex)
  {
    // Create a DeviceBary from a ToolBary
    const std::unique_ptr<micromesh_tool::ToolBary>& toolBary = scn.barys()[baryIndex];
    m_barys.push_back(std::make_unique<DeviceBary>());
    std::unique_ptr<DeviceBary>& deviceBary = m_barys.back();
    for(int32_t baryGroup = 0; baryGroup < static_cast<int32_t>(toolBary->groups().size()); ++baryGroup)
    {
      // Build a structure of addresses to reference the ToolMicromap data in
      // shaders. These are linearized, so m_deviceBaryInfoMap is created to
      // refer back to them given a bary and group index.
      DeviceBaryInfo info{};

      // Add a DeviceMicromap to the DeviceBary for every ToolBary's group
      const bary::ContentView& groupView     = toolBary->groups()[baryGroup];
      auto                     displacedMesh = baryToMeshMap.find({baryIndex, baryGroup});
      if(displacedMesh == baryToMeshMap.end())
      {
        LOGI("Skipping unused micromap %i group %i\n", baryIndex, baryGroup);
        m_barys.back()->addEmpty();
      }
      else if(groupView.basic.valuesInfo->valueFormat != bary::Format::eDispC1_r11_unorm_block)
      {
        LOGW("Warning: cannot render uncompressed micromap %i group %i\n", baryIndex, baryGroup);
        m_barys.back()->addEmpty();
      }
      else
      {
        m_barys.back()->addMicromap(m_context, *m_alloc, extraQueue.queue, extraQueue.familyIndex, cmd, usageFlags,
                                    groupView, *displacedMesh->second);
        const DeviceMicromap& micromap = deviceBary->micromaps().back();

        info.baryValuesBuffer    = micromap.valuesAddress();
        info.baryTrianglesBuffer = micromap.trianglesAddress();
        if(micromap.raster())
        {
          const microdisp::MicromeshSetCompressedVK& rasterData = micromap.raster()->micromeshSet;
          assert(rasterData.meshDatas.size() == 1);
          info.rasterMeshDataBindingBuffer =
              nvvk::getBufferDeviceAddress(m_ctx->m_device, rasterData.meshDatas[0].binding.buffer);
        }
      }

      // Add info even if it's empty so that baryInfoIndex() can always return a valid index
      // TODO: use same index as the mesh
      m_deviceBaryInfoMap[{baryIndex, baryGroup}] = static_cast<int32_t>(deviceBaryInfos.size());
      deviceBaryInfos.push_back(info);
    }
  }

  auto usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  // Creating the buffer of all device bary information
  m_bDeviceBaryInfo = m_alloc->createBuffer(cmd, deviceBaryInfos, usage_flag);
  m_dutil->DBG_NAME(m_bDeviceBaryInfo.buffer);

  return true;
}

void ToolboxSceneVk::createTextureImages(VkCommandBuffer cmd, const std::vector<tinygltf::Texture>& textures, const ToolImageVector& images)
{
  nvh::ScopedTimer _st("- Create Textures\n");

  VkSamplerCreateInfo sampler_create_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_create_info.minFilter  = VK_FILTER_LINEAR;
  sampler_create_info.magFilter  = VK_FILTER_LINEAR;
  sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_create_info.maxLod     = FLT_MAX;

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [&](uint32_t idx, const std::array<uint8_t, 4>& color) {
    VkImageCreateInfo image_create_info = nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1});
    nvvk::Image       image             = m_alloc->createImage(cmd, 4, color.data(), image_create_info);
    assert(idx < m_images.size());
    m_images[idx] = {image, image_create_info};
    m_dutil->setObjectName(m_images[idx].nvvkImage.image, "Dummy");
  };

  // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [&]() {
    assert(!m_images.empty());
    SceneImage&           scn_image = m_images[0];
    VkImageViewCreateInfo iv_info   = nvvk::makeImageViewCreateInfo(scn_image.nvvkImage.image, scn_image.createInfo);
    m_textures.emplace_back(m_alloc->createTexture(scn_image.nvvkImage, iv_info, sampler_create_info));
  };

  // Load images in parallel
  m_images.resize(images.size());
  uint32_t num_threads = std::min((uint32_t)images.size(), std::thread::hardware_concurrency());
  nvh::parallel_batches<1>(  // Not batching
      images.size(),
      [&](uint64_t i) {
        auto& image = *images[i];
        LOGI("  - (%" PRIu64 ") %s \n", i, image.relativePath().string().c_str());
        loadImage(image, m_images[i]);
      },
      num_threads);

  // Create Vulkan images
  auto lock = GetVkQueueOrAllocatorLock();
  for(size_t i = 0; i < m_images.size(); i++)
  {
    if(!createImage(cmd, m_images[i]))
    {
      addDefaultImage((uint32_t)i, {255, 0, 255, 255});  // Image not present or incorrectly loaded (image.empty)
    }
  }

  // Add default image if nothing was loaded
  if(images.empty())
  {
    m_images.resize(1);
    addDefaultImage(0, {255, 255, 255, 255});
  }

  // Creating the textures using the above images
  m_textures.reserve(textures.size());
  for(size_t i = 0; i < textures.size(); i++)
  {
    int source_image = textures[i].source;
    if(static_cast<size_t>(source_image) >= images.size() || source_image < 0)
    {
      addDefaultTexture();  // Incorrect source image
      continue;
    }

    SceneImage&           scn_image = m_images[source_image];
    VkImageViewCreateInfo iv_info   = nvvk::makeImageViewCreateInfo(scn_image.nvvkImage.image, scn_image.createInfo);
    m_textures.emplace_back(m_alloc->createTexture(scn_image.nvvkImage, iv_info, sampler_create_info));
  }

  // Add a default texture, cannot work with empty descriptor set
  if(textures.empty())
  {
    addDefaultTexture();
  }
}

//--------------------------------------------------------------------------------------------------
// Loading images from disk
//
void ToolboxSceneVk::loadImage(micromesh_tool::ToolImage& toolImage, SceneImage& image)
{
  if(!toolImage.info().valid())
  {
    // Image failed to load, e.g. file not found.
    return;
  }

  VkFormat vkFormat = toolImage.info().vkFormat();
  if(vkFormat == VK_FORMAT_UNDEFINED)
  {
    // Unsupported image format
    return;
  }

  auto rawData = static_cast<uint8_t*>(toolImage.raw());
  if(!rawData)
  {
    return;
  }

  image.size   = {static_cast<uint32_t>(toolImage.info().width), static_cast<uint32_t>(toolImage.info().height)};
  image.format = vkFormat;
  image.mipData.emplace_back(rawData, rawData + toolImage.info().totalBytes());
}

bool ToolboxSceneVk::createImage(const VkCommandBuffer& cmd, SceneImage& image)
{
  if(image.size.width == 0 || image.size.height == 0)
    return false;

  VkFormat          format            = image.format;
  VkExtent2D        img_size          = image.size;
  VkImageCreateInfo image_create_info = nvvk::makeImage2DCreateInfo(img_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

  // Check if we can generate mipmap with the the incoming image
  bool               can_generate_mipmaps = false;
  VkFormatProperties format_properties;
  vkGetPhysicalDeviceFormatProperties(m_ctx->m_physicalDevice, format, &format_properties);
  if((format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == VK_FORMAT_FEATURE_BLIT_DST_BIT)
    can_generate_mipmaps = true;
  if(image.mipData.size() > 1)  // Use only the number of levels defined
    image_create_info.mipLevels = (uint32_t)image.mipData.size();
  if(image.mipData.size() == 1 && can_generate_mipmaps == false)
    image_create_info.mipLevels = 1;  // Cannot use cmdGenerateMipmaps

  // Keep info for the creation of the texture
  image.createInfo = image_create_info;

  VkDeviceSize buffer_size  = image.mipData[0].size();
  nvvk::Image  result_image = m_alloc->createImage(cmd, buffer_size, image.mipData[0].data(), image_create_info);

  if(image.mipData.size() == 1 && can_generate_mipmaps)
  {
    nvvk::cmdGenerateMipmaps(cmd, result_image.image, format, img_size, image_create_info.mipLevels);
  }
  else
  {
    // Create all mip-levels
    nvvk::cmdBarrierImageLayout(cmd, result_image.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    auto staging = m_alloc->getStaging();
    for(uint32_t mip = 1; mip < (uint32_t)image.mipData.size(); mip++)
    {
      image_create_info.extent.width  = std::max(1u, image.size.width >> mip);
      image_create_info.extent.height = std::max(1u, image.size.height >> mip);

      VkOffset3D               offset{};
      VkImageSubresourceLayers subresource{};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      subresource.mipLevel   = mip;

      std::vector<uint8_t>& mipresource = image.mipData[mip];
      VkDeviceSize          bufferSize  = mipresource.size();
      if(image_create_info.extent.width > 0 && image_create_info.extent.height > 0)
      {
        staging->cmdToImage(cmd, result_image.image, offset, image_create_info.extent, subresource, bufferSize,
                            mipresource.data());
      }
    }
    nvvk::cmdBarrierImageLayout(cmd, result_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  if(!image.imgName.empty())
  {
    m_dutil->setObjectName(result_image.image, image.imgName);
  }
  else
  {
    m_dutil->DBG_NAME(result_image.image);
  }

  // Clear image.mipData as it is no longer needed
  image = {result_image, image_create_info, image.srgb, image.imgName};

  return true;
}

void ToolboxSceneVk::destroy()
{
  for(auto& device : m_deviceMeshes)
  {
    meshops::meshopsDeviceMeshDestroy(m_context, device);
  }
  m_deviceMeshes = {};

  auto lock = GetVkQueueOrAllocatorLock();
  m_alloc->destroy(m_bMaterial);
  m_alloc->destroy(m_bDeviceMeshInfo);
  m_alloc->destroy(m_bDeviceBaryInfo);
  m_alloc->destroy(m_bInstances);
  m_alloc->destroy(m_bSceneDesc);

  for(auto& bary : m_barys)
  {
    bary->deinit(*m_alloc);
  }
  m_barys.clear();

  for(auto& i : m_images)
  {
    m_alloc->destroy(i.nvvkImage);
  }
  m_images.clear();

  for(auto& t : m_textures)
  {
    vkDestroyImageView(m_ctx->m_device, t.descriptor.imageView, nullptr);
  }
  m_textures.clear();
}
