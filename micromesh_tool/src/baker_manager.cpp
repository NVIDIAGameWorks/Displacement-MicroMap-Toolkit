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
#include "baker_manager.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <sstream>
#include <mutex>

#include "nvvk/images_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvh/gltfscene.hpp"

#include "imageio/imageio.hpp"
#include <nvh/parallel_work.hpp>
#include <unordered_set>

#include "_autogen/pullpush.comp.h"
#include "tool_image.hpp"

namespace tool_bake {

namespace fs = std::filesystem;

// This gives a way to iterate over the resampleable texture members of an
// nvh::GltfMaterial. We use an order where the first textures are always
// normal textures. Note that we don't iterate over or resample
// KHR_materials_displacement.
enum class GltfTextureField : uint32_t
{
  eStart  = 0,
  eNormal = eStart,
  eClearcoatNormal,
  eNormalFieldsEnd,
  eEmissive = eNormalFieldsEnd,
  eOcclusion,
  ePBRBaseColor,
  ePBRMetallicRoughness,
  eSpecularGlossinessDiffuse,
  eSpecularGlossinessSpecularGlossiness,
  eSpecularTexture,
  eSpecularColorTexture,
  eClearcoat,
  eClearcoatRoughness,
  eTransmission,
  eAnisotropy,
  eVolumeThickness,
  eAllFieldsEnd,
  eInvalid
};

GltfTextureField& operator++(GltfTextureField& v)
{
  v = static_cast<GltfTextureField>(static_cast<uint32_t>(v) + 1);
  return v;
}

GltfTextureField getResampleableFieldEnd(TexturesToResample set)
{
  if(set == TexturesToResample::eNone)
  {
    return GltfTextureField::eStart;
  }
  else if(set == TexturesToResample::eNormals)
  {
    return GltfTextureField::eNormalFieldsEnd;
  }
  else
  {
    return GltfTextureField::eAllFieldsEnd;
  }
}

// Returns the fieldIndex'th possibly-filled texture in a material, which is
// -1 if no texture was set. The code below depends on how GltfScene's
// materials are the same as tinygltf's materials, with a default material at
// the end.
// Note that texture indexing may not be the same as image indexing! Look at
// the textures[i].source property to get the index of the image.
int getTextureField(const nvh::GltfMaterial& mat, GltfTextureField fieldIndex)
{
  switch(fieldIndex)
  {
    case GltfTextureField::eNormal:
      return mat.normalTexture;
    case GltfTextureField::eClearcoatNormal:
      return mat.clearcoat.normalTexture;
    case GltfTextureField::eEmissive:
      return mat.emissiveTexture;
    case GltfTextureField::eOcclusion:
      return mat.occlusionTexture;
    case GltfTextureField::ePBRBaseColor:
      return mat.baseColorTexture;
    case GltfTextureField::ePBRMetallicRoughness:
      return mat.metallicRoughnessTexture;
    case GltfTextureField::eSpecularGlossinessDiffuse:
      return mat.specularGlossiness.diffuseTexture;
    case GltfTextureField::eSpecularGlossinessSpecularGlossiness:
      return mat.specularGlossiness.specularGlossinessTexture;
    case GltfTextureField::eSpecularTexture:
      return mat.specular.specularTexture;
    case GltfTextureField::eSpecularColorTexture:
      return mat.specular.specularColorTexture;
    case GltfTextureField::eClearcoat:
      return mat.clearcoat.texture;
    case GltfTextureField::eClearcoatRoughness:
      return mat.clearcoat.roughnessTexture;
    case GltfTextureField::eTransmission:
      return mat.transmission.texture;
    case GltfTextureField::eAnisotropy:
      return mat.anisotropy.texture;
    case GltfTextureField::eVolumeThickness:
      return mat.volume.thicknessTexture;
  }
  return -1;
}

const char* getTextureFieldName(GltfTextureField fieldIndex)
{
  switch(fieldIndex)
  {
    case GltfTextureField::eNormal:
      return "normal";
    case GltfTextureField::eClearcoatNormal:
      return "clearcoatNormal";
    case GltfTextureField::eEmissive:
      return "emissive";
    case GltfTextureField::eOcclusion:
      return "occlusion";
    case GltfTextureField::ePBRBaseColor:
      return "color";
    case GltfTextureField::ePBRMetallicRoughness:
      return "metallicRoughness";
    case GltfTextureField::eSpecularGlossinessDiffuse:
      return "diffuse";
    case GltfTextureField::eSpecularGlossinessSpecularGlossiness:
      return "specularGlossiness";
    case GltfTextureField::eSpecularTexture:
      return "specular";
    case GltfTextureField::eSpecularColorTexture:
      return "specularColor";
    case GltfTextureField::eClearcoat:
      return "clearcoat";
    case GltfTextureField::eClearcoatRoughness:
      return "clearcoatRoughness";
    case GltfTextureField::eTransmission:
      return "transmission";
    case GltfTextureField::eAnisotropy:
      return "anisotropy";
    case GltfTextureField::eVolumeThickness:
      return "thickness";
  }
  return "unknown";
}

static std::string getResampledTextureFilename(const BakerManagerConfig& info, const fs::path& source, size_t outputTextureIndex)
{
  assert(!source.empty());
  fs::path newFilename = source.stem().string() + "_resampled" + source.extension().string();
  return (source.parent_path() / newFilename).string();
}

struct NewImageSource
{
  const tinygltf::Material* material = nullptr;
  GltfTextureField          field    = GltfTextureField::eInvalid;
};

// Replace spaces and characters that are often illegal in filenames with
// underscores.
static std::string sanitizeFilename(std::string filename)
{
  static const std::unordered_set<char> illegalChars = {'/', '<', '>', ':', '"', '\\', '|', '?', '*'};
  for(auto& c : filename)
  {
    // Non-printable characters, space and illegal characters.
    if(c < ' ' || c == ' ' || illegalChars.count(c))
    {
      c = '_';
    }
  }
  return filename;
}

static std::string getNewTextureFilename(const BakerManagerConfig& info, const NewImageSource& source, size_t outputTextureIndex)
{
  // Use the explicitly given output stem if it exists (typically from
  // micromesh_tool CLI). Else, generate one based on the material name.
  std::string stem = info.outTextureStem;
  if(stem.empty() && source.material != nullptr)
  {
    stem = sanitizeFilename(source.material->name);
  }

  // Add the texture field to the stem, e.g. "color", if it exists
  if(source.field != GltfTextureField::eInvalid)
  {
    stem = stem + (stem.empty() ? "" : "_") + getTextureFieldName(source.field);
  }

  return stem + "_resampled_new_" + std::to_string(outputTextureIndex) + ".png";
}

void setTextureFieldBase(tinygltf::Material&       tinygltfMaterial,
                         const char*               extensionName,
                         const char*               fieldName,
                         int                       index,
                         const tinygltf::Material& defaultMaterialIfNull)
{
  tinygltf::Value* extension;
  const auto&      it = tinygltfMaterial.extensions.find(extensionName);
  if(it != tinygltfMaterial.extensions.end())
  {
    extension = &(it->second);
  }
  else
  {
    // tinygltfMaterial didn't have this extension. Find what this extension
    // should look like by default, and add it to tinygltfMaterial.
    const auto& findDefaultExtension = defaultMaterialIfNull.extensions.find(extensionName);
    if(findDefaultExtension == defaultMaterialIfNull.extensions.end())
    {
      // Nothing we can do without additional logic for setting default
      // textures, which probably isn't what we want in this context.
      // This case should never be reached in this code.
      assert(false);
      return;
    }
    else
    {
      // Insert it into the ExtensionMap, then get it
      extension = &(tinygltfMaterial.extensions.insert({extensionName, findDefaultExtension->second}).first->second);
    }
  }

  // This is a bit unusual: we're setting v.fieldName.index by asserting
  // that v is a JSON Object, accessing `fieldName`, asserting that that
  // field is an Object, accessing `index`, asserting that that is an int,
  // and setting it.
  if(!extension->IsObject())
    return;
  tinygltf::Value::Object& obj   = extension->Get<tinygltf::Value::Object>();
  tinygltf::Value&         field = obj[fieldName];
  if(!field.IsObject())
    return;
  tinygltf::Value& baseObj = field.Get<tinygltf::Value::Object>()["index"];
  baseObj                  = tinygltf::Value(index);
}

// Sets a texture field, adding it if it didn't already exist.
void setTextureField(tinygltf::Material&       tgMat,
                     nvh::GltfMaterial&        nvhMat,
                     GltfTextureField          fieldIndex,
                     int                       textureIndex,
                     const tinygltf::Material& defaultMaterialIfNull)
{
  switch(fieldIndex)
  {
    case GltfTextureField::eNormal:
      nvhMat.normalTexture      = textureIndex;
      tgMat.normalTexture.index = textureIndex;
      break;
    case GltfTextureField::eClearcoatNormal:
      nvhMat.clearcoat.normalTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME, "clearcoatNormalTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eEmissive:
      nvhMat.emissiveTexture      = textureIndex;
      tgMat.emissiveTexture.index = textureIndex;
      break;
    case GltfTextureField::eOcclusion:
      nvhMat.occlusionTexture      = textureIndex;
      tgMat.occlusionTexture.index = textureIndex;
      break;
    case GltfTextureField::ePBRBaseColor:
      nvhMat.baseColorTexture                           = textureIndex;
      tgMat.pbrMetallicRoughness.baseColorTexture.index = textureIndex;
      break;
    case GltfTextureField::ePBRMetallicRoughness:
      nvhMat.metallicRoughnessTexture                           = textureIndex;
      tgMat.pbrMetallicRoughness.metallicRoughnessTexture.index = textureIndex;
      break;
    case GltfTextureField::eSpecularGlossinessDiffuse:
      nvhMat.specularGlossiness.diffuseTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_PBRSPECULARGLOSSINESS_EXTENSION_NAME, "diffuseTexture", textureIndex,
                          defaultMaterialIfNull);
      break;
    case GltfTextureField::eSpecularGlossinessSpecularGlossiness:
      nvhMat.specularGlossiness.specularGlossinessTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_PBRSPECULARGLOSSINESS_EXTENSION_NAME, "specularGlossinessTexture",
                          textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eSpecularTexture:
      nvhMat.specular.specularTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_SPECULAR_EXTENSION_NAME, "specularTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eSpecularColorTexture:
      nvhMat.specular.specularColorTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_SPECULAR_EXTENSION_NAME, "specularColorTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eClearcoat:
      nvhMat.clearcoat.texture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME, "clearcoatTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eClearcoatRoughness:
      nvhMat.clearcoat.roughnessTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME, "clearcoatRoughnessTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eTransmission:
      nvhMat.transmission.texture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME, "transmissionTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eAnisotropy:
      nvhMat.anisotropy.texture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME, "anisotropyTexture", textureIndex, defaultMaterialIfNull);
      break;
    case GltfTextureField::eVolumeThickness:
      nvhMat.volume.thicknessTexture = textureIndex;
      setTextureFieldBase(tgMat, KHR_MATERIALS_VOLUME_EXTENSION_NAME, "thicknessTexture", textureIndex, defaultMaterialIfNull);
      break;
  }
}

// Must sync before using using the pointer and use before freeing the allocator's staging buffer
const void* downloadImage(nvvk::ResourceAllocator& alloc, VkCommandBuffer cmdBuf, const GPUTextureContainer& tex)
{
  VkDeviceSize             size        = tex.mipSizeInBytes(0);
  VkOffset3D               offset      = {0, 0, 0};
  VkImageSubresourceLayers subresource = {0};
  subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
  subresource.mipLevel                 = 0;
  subresource.layerCount               = 1;
  subresource.baseArrayLayer           = 0;
  const void* mapped = alloc.getStaging()->cmdFromImage(cmdBuf, tex.texture.image, offset, tex.info.extent, subresource,
                                                        size, tex.texture.descriptor.imageLayout);
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  return mapped;
}

bool isPowerOfTwo(uint32_t n)
{
  return (n != 0) && ((n & (n - 1)) == 0);
}

BakerManager::BakerManager(VkDevice device, nvvk::ResourceAllocator& alloc)
    : m_device(device)
    , m_alloc(&alloc)
{
  m_numThreads = std::thread::hardware_concurrency();
}

//-----------------------------------------------------------------------------
// Member functions

bool BakerManager::generateInstructions(const BakerManagerConfig&                   info,
                                        const micromesh_tool::ToolScene*            highMesh,
                                        std::unique_ptr<micromesh_tool::ToolScene>& lowMesh,
                                        std::vector<ResampleMeshInstructions>&      instructions)
{
  m_highMesh = highMesh;
  m_lowMesh  = lowMesh.get();

  nvh::GltfScene lowSceneMaterials;
  nvh::GltfScene highSceneMaterials;
  lowSceneMaterials.importMaterials(lowMesh->model());
  highSceneMaterials.importMaterials(highMesh->model());

  if(info.texturesToResample == TexturesToResample::eNone  //
     && info.quaternionTexturesStem.empty()                //
     && info.offsetTexturesStem.empty()                    //
     && info.heightTexturesStem.empty())
  {
    // We're not resampling! No need to do anything - just resize
    // `instructions` so we preserve the "one `ResampleMeshInstructions` per
    // mesh" invariant, but include 0 resampling instructions per mesh.
    instructions.resize(highMesh->meshes().size(), {});
    return true;
  }

  instructions.resize(highMesh->meshes().size());

  tinygltf::Material defaultMaterial = tinygltf::Material();
  defaultMaterial.emissiveFactor     = {0.0, 0.0, 0.0};

  // If the lo-res scene had no materials, the GltfScene process will create
  // a default one. Let's match that in the glTF file.
  if(lowMesh->materials().size() == 0)
  {
    // Only produce a warning if we wanted to resample something
    if(info.texturesToResample != TexturesToResample::eNone)
    {
      LOGW(
          "Warning: The lo-res file had no materials. The lo-res file's material setup defines which meshes should "
          "write to which textures. That means this will work if there's only one input material, but if there are "
          "multiple input meshes and multiple input materials, you may wish to ensure the output has (untexured) "
          "materials.\n")
    }
    lowMesh->materials().push_back(defaultMaterial);
    for(auto& mesh : lowMesh->meshes())
      mesh->relations().material = 0;
  }

  if(highMesh->materials().size() == 0)
  {
    // Produce an error if we wanted to resample something. If we're only
    // generating offset or quaternion textures, this is OK.
    if(info.texturesToResample != TexturesToResample::eNone)
    {
      LOGE(
          "Error: The hi-res file had no materials! This means we have no information about how to resample the input. "
          "Please make sure the exporter had material exporting enabled.\n");
      return false;
    }
  }

  if(highMesh->meshes().size() != lowMesh->meshes().size())
  {
    LOGE(
        "Error: The hi-res and lo-res scene must have the same number of primitive meshes! The hi-res scene had %zu, "
        "while the lo-res scene had %zu.\n",
        highMesh->meshes().size(), lowMesh->meshes().size());
    return false;
  }

  const GltfTextureField resampleableFieldEnd = getResampleableFieldEnd(info.texturesToResample);

  const size_t NO_INVERSE = ~size_t(0);
  // Used to check the condition in the comments above.
  // materialToFirstPrimitive[material idx] gives the first low-res GltfPrimMesh
  // encountered so far that used that material.
  std::vector<size_t> materialToFirstPrimitive(lowMesh->materials().size(), NO_INVERSE);

  // Tracks which hi-res images we'll read.
  std::unordered_set<int> hiImagesToLoadSet;

  std::unordered_set<int> loImagesToWrite;    // All images we'll write to
  std::unordered_set<int> loImagesToReplace;  // All images we'll write to that aren't new images.

  // Record textures that are to be resampled into each output image so it can be sized to the maximum
  std::map<int, std::vector<int>> loImageSources;

  // Record sources for generating new filenames
  std::unordered_map<int, NewImageSource> newImageSources;

  for(size_t meshIdx = 0; meshIdx < lowMesh->meshes().size(); meshIdx++)
  {
    const std::unique_ptr<micromesh_tool::ToolMesh>& loMesh        = lowMesh->meshes()[meshIdx];
    const int                                        loMaterialIdx = loMesh->relations().material;
    const int                                        hiMaterialIdx = highMesh->meshes()[meshIdx]->relations().material;

    // Skip meshes without materials
    if(hiMaterialIdx == -1)
    {
      LOGI("Resampler skipping mesh pair %zu as the reference mesh has no material\n", meshIdx);
      continue;
    }
    if(loMaterialIdx == -1)
    {
      LOGI("Resampler skipping mesh pair %zu as the base mesh has no material\n", meshIdx);
      continue;
    }

    nvh::GltfMaterial&       loMat = lowSceneMaterials.m_materials[loMaterialIdx];
    const nvh::GltfMaterial& hiMat = highSceneMaterials.m_materials[hiMaterialIdx];

    // Check the above condition
    {
      const size_t otherMeshIdx = materialToFirstPrimitive[loMaterialIdx];
      if(otherMeshIdx == NO_INVERSE)
      {
        materialToFirstPrimitive[loMaterialIdx] = meshIdx;
      }
      else
      {
        const nvh::GltfMaterial& otherHiMat =
            highSceneMaterials.m_materials[highMesh->meshes()[otherMeshIdx]->relations().material];

        bool ok = true;
        for(GltfTextureField field = GltfTextureField::eStart; field < resampleableFieldEnd; ++field)
        {
          ok = ok && ((getTextureField(hiMat, field) == -1) == (getTextureField(otherHiMat, field) == -1));
        }

        if(!ok)
        {
          LOGE(
              "This scene has material configurations that prevents resampling from working correctly at the moment: "
              "lo-res GltfUMeshes %zu and %zu had the same material index, but hi-res GltfUmeshes %zu and %zu had "
              "different materials, and specifically a different set of textures to resample in their materials.\n",
              meshIdx, otherMeshIdx, meshIdx, otherMeshIdx);
          return false;
        }
      }
    }

    // Track the resample operations that will occur. Set the output paths for
    // new low-res textures, and track referenced low-res textures that already
    // existed.

    tinygltf::Material&       loMatTG = lowMesh->materials()[loMaterialIdx];
    const tinygltf::Material& hiMatTG = highMesh->materials()[highMesh->meshes()[meshIdx]->relations().material];

    // For each hi image idx, contains the list of lo images it will be
    // resampled to, but only for this mesh!
    std::vector<std::unordered_set<int>> meshHiLoList(highMesh->images().size());

    for(GltfTextureField field = GltfTextureField::eStart; field < resampleableFieldEnd; ++field)
    {
      const int hiResTextureIdx = getTextureField(hiMat, field);

      if(hiResTextureIdx == -1)
        continue;

      if(static_cast<size_t>(hiResTextureIdx) >= highMesh->textures().size())
      {
        // Turns out TinyGLTF doesn't check for this!
        LOGE(
            "Error: Mesh %zu had a material that referenced texture %i, but the hi-res glTF file only has %zu "
            "textures!\n",
            meshIdx, hiResTextureIdx, highMesh->textures().size());
        return false;
      }

      const int hiResImageIdx   = highMesh->textures()[hiResTextureIdx].source;
      int       loResTextureIdx = getTextureField(loMat, field);
      int       loResImageIdx;

      if(loResTextureIdx == -1)
      {
        // The material of the hi-res mesh had an image that was missing from
        // the material of the lo-res mesh. To handle the use case where we're
        // given a textured hi-res mesh, a blank lo-res mesh, and asked to
        // transfer textures from the hi-res mesh to the lo-res mesh, we add
        // a new lo-res texture pointing to a new lo-res image.
        loResTextureIdx = int(lowMesh->textures().size());
        loResImageIdx   = lowMesh->createImage();

        tinygltf::Texture newTexture;
        newTexture.source = loResImageIdx;
        lowMesh->textures().push_back(newTexture);

        setTextureField(loMatTG, loMat, field, loResTextureIdx, hiMatTG);
        newImageSources[loResImageIdx] = {&hiMatTG, field};
      }
      else
      {
        loResImageIdx = lowMesh->textures()[loResTextureIdx].source;
        // Mark it as an image to replace, if we haven't added it as an image
        // to write (e.g. it must not be a new image).
        if(loImagesToWrite.find(loResImageIdx) == loImagesToWrite.end())
        {
          loImagesToReplace.insert(loResImageIdx);
        }
      }

      loImagesToWrite.insert(loResImageIdx);
      hiImagesToLoadSet.insert(hiResImageIdx);
      loImageSources[loResImageIdx].push_back(hiResImageIdx);

      // Make sure we don't write instructions for the same {hi, lo} pair
      // twice. That can happen when we have Opacity, Roughness, Metallic
      // textures (where the same texture is used for both the opacity and
      // roughness and metallic fields)
      const auto& hiLoIterator = meshHiLoList[hiResImageIdx].find(loResImageIdx);
      if(hiLoIterator == meshHiLoList[hiResImageIdx].end())
      {
        meshHiLoList[hiResImageIdx].insert(loResImageIdx);

        ResampleInstruction instruction;
        instruction.texelContent =
            ((field < GltfTextureField::eNormalFieldsEnd) ? meshops::TextureType::eNormalMap : meshops::TextureType::eGeneric);
        instruction.inputIndex  = hiResImageIdx;
        instruction.outputIndex = loResImageIdx;
        instructions[meshIdx].instructions.push_back(instruction);
      }
    }
  }

  // Prepare input storage
  m_resamplingInputStorage.resize(highMesh->images().size());
  for(int hiImgIdx : hiImagesToLoadSet)
  {
    GPUTextureContainer& container = m_resamplingInputStorage[hiImgIdx];
    // Get the input filename and file info
    const micromesh_tool::ToolImage& toolImage = *highMesh->images()[hiImgIdx];
    container.storageLocation                  = GPUTextureContainer::Storage::eToolImage;
    container.info.extent.width                = uint32_t(toolImage.info().width);
    container.info.extent.height               = uint32_t(toolImage.info().height);
    container.info.mipLevels                   = 1;
    container.info.format                      = RESAMPLE_COLOR_FORMAT;
    container.filePath                         = toolImage.relativePath().string();
    assert(!container.filePath.empty());
  }

  // Prepare output storage
  m_resamplingOutputStorage.resize(lowMesh->images().size());
  for(const int loImgIdx : loImagesToWrite)
  {
    // Use the global output resolution, or the maximum resolution of all
    // contributing textures, e.g. when a low resolution material references
    // textures used by separate high resolution materials.
    glm::uvec2 size{info.resampleResolution, info.resampleResolution};
    if(info.resampleResolution == 0)
    {
      for(int hiImgIdx : loImageSources[loImgIdx])
      {
        auto& input = m_resamplingInputStorage[hiImgIdx];
        size        = glm::max(size, {input.info.extent.width, input.info.extent.height});
      }
    }

    GPUTextureContainer& result = m_resamplingOutputStorage[loImgIdx];
    if(loImagesToReplace.find(loImgIdx) != loImagesToReplace.end())
    {
      // Rename lo-resolution textures we're writing that aren't new
      micromesh_tool::ToolImage& image = *lowMesh->images()[loImgIdx];
      result.filePath                  = getResampledTextureFilename(info, image.relativePath(), loImgIdx);
    }
    else
    {
      result.filePath = getNewTextureFilename(info, newImageSources[loImgIdx], loImgIdx);
    }
    result.info.extent.width    = size.x;
    result.info.extent.height   = size.y;
    result.info.mipLevels       = nvvk::mipLevels(result.info.extent);
    result.info.format          = RESAMPLE_COLOR_FORMAT;
    result.storageLocation      = GPUTextureContainer::Storage::eCreateOnFirstUse;
    assert(!result.filePath.empty());
  }
  const size_t glTFOutputEnd = m_resamplingOutputStorage.size();

  // Add extra resampling textures and instructions
  for(size_t i = 0; i < info.resampleExtraTextures.size(); i++)
  {
    const ResampleExtraTexture& extraTexture = info.resampleExtraTextures[i];

    if(static_cast<size_t>(extraTexture.meshIdx) > lowMesh->meshes().size())
    {
      LOGE(
          "Error: The mesh index (%i) for resample extra texture %zu was out of bounds (it was greater than the "
          "number of glTF primitives in the scene, %zu)!\n",
          extraTexture.meshIdx, i, lowMesh->meshes().size());
      return false;
    }
    if(extraTexture.inURI.empty())
    {
      LOGE(
          "The high image URI for resample extra texture %zu was empty! Without this, we don't know what to sample "
          "from.\n",
          i);
      return false;
    }

    // Input textures typically come from the scene. This is the only source
    // that is a file and not on the scene object.
    GPUTextureContainer& hiTex = m_resamplingInputStorage.emplace_back(GPUTextureContainer{});
    hiTex.filePath             = extraTexture.inURI;
    size_t w = 0, h = 0, comp = 0;
    if(!imageio::info(hiTex.filePath.c_str(), &w, &h, &comp))
    {
      LOGE("Error: imageio::info could not read %s.\n", hiTex.filePath.c_str());
      return false;
    }
    hiTex.info.extent.width  = static_cast<uint32_t>(w);
    hiTex.info.extent.height = static_cast<uint32_t>(h);
    hiTex.info.mipLevels     = 1;
    hiTex.info.format        = RESAMPLE_COLOR_FORMAT;
    hiTex.storageLocation    = GPUTextureContainer::Storage::eImageFile;

    GPUTextureContainer& outTex = m_resamplingOutputStorage.emplace_back(GPUTextureContainer{});
    if(extraTexture.outURI.empty())
    {
      if(extraTexture.inURI.empty())
      {
        outTex.filePath = getNewTextureFilename(info, {}, m_resamplingOutputStorage.size() - 1);
      }
      else
      {
        fs::path path        = extraTexture.inURI;
        fs::path newFilename = path.stem().string() + "_resampled" + path.extension().string();
        outTex.filePath = getResampledTextureFilename(info, extraTexture.inURI, m_resamplingOutputStorage.size() - 1);
      }
    }
    else
    {
      outTex.filePath = extraTexture.outURI;
    }
    if(info.resampleResolution == 0)
    {
      outTex.info.extent = hiTex.info.extent;
    }
    else
    {
      outTex.info.extent.width  = static_cast<uint32_t>(info.resampleResolution);
      outTex.info.extent.height = static_cast<uint32_t>(info.resampleResolution);
    }
    outTex.info.mipLevels  = nvvk::mipLevels(outTex.info.extent);
    outTex.info.format     = RESAMPLE_COLOR_FORMAT;
    outTex.storageLocation = GPUTextureContainer::Storage::eCreateOnFirstUse;
    assert(!outTex.filePath.empty());

    ResampleInstruction instruction;
    instruction.texelContent = (extraTexture.isNormalMap ? meshops::TextureType::eNormalMap : meshops::TextureType::eGeneric);
    instruction.inputIndex  = m_resamplingInputStorage.size() - 1;
    instruction.outputIndex = m_resamplingOutputStorage.size() - 1;
    instructions[extraTexture.meshIdx].instructions.push_back(instruction);
  }

  // Add quaternion and offset textures. These are special because they don't
  // have an input texture; they depend only on the hi-res and lo-res mesh.
  {
    // Determine what resolution they should be, since they have no
    // corresponding input. If resampleResolution was specified, we use that:
    uint32_t noInputResolution = info.resampleResolution;
    if(noInputResolution == 0)
    {
      // Otherwise, we use the maximum resolution of our inputs:
      for(const GPUTextureContainer& inputTexture : m_resamplingInputStorage)
      {
        noInputResolution = std::max(inputTexture.info.extent.width, noInputResolution);
        noInputResolution = std::max(inputTexture.info.extent.height, noInputResolution);
      }
    }
    if(noInputResolution == 0)
    {
      // Otherwise, we could silently choose, say, 4096 x 4096. We'll warn
      // about this, though.
      noInputResolution = 4096;
      if((!info.quaternionTexturesStem.empty()) || (!info.offsetTexturesStem.empty()) || (!info.heightTexturesStem.empty()))
      {
        LOGW(
            "Warning: Quaternion textures or offset textures were requested, but their resolution was unspecified, "
            "since there were no other input textures and --resample-resolution was 0. Choosing a resolution of %u x "
            "%u.",
            noInputResolution, noInputResolution);
      }
    }
    std::array<std::pair<const std::string*, meshops::TextureType>, 3> cases;
    cases[0] = {&info.quaternionTexturesStem, meshops::TextureType::eQuaternionMap};
    cases[1] = {&info.offsetTexturesStem, meshops::TextureType::eOffsetMap};
    cases[2] = {&info.heightTexturesStem, meshops::TextureType::eHeightMap};
    for(const auto& stemContent : cases)
    {
      const std::string& stem = *stemContent.first;
      if(stem.empty())
      {
        continue;
      }

      for(size_t meshIdx = 0; meshIdx < lowMesh->meshes().size(); meshIdx++)
      {
        GPUTextureContainer& outTex = m_resamplingOutputStorage.emplace_back(GPUTextureContainer{});
        outTex.filePath             = stem + "." + std::to_string(meshIdx) + ".png";
        outTex.info.extent.width    = noInputResolution;
        outTex.info.extent.height   = noInputResolution;
        outTex.info.mipLevels       = nvvk::mipLevels(outTex.info.extent);
        switch(stemContent.second)
        {
          default:
            assert(false);
          case meshops::TextureType::eQuaternionMap:
            outTex.info.format = RESAMPLE_QUATERNION_FORMAT;
            break;
          case meshops::TextureType::eOffsetMap:
            outTex.info.format = RESAMPLE_OFFSET_FORMAT;
            break;
          case meshops::TextureType::eHeightMap:
            outTex.info.format = RESAMPLE_HEIGHT_FORMAT;
            break;
        }
        outTex.storageLocation = GPUTextureContainer::Storage::eCreateOnFirstUse;
        assert(!outTex.filePath.empty());

        ResampleInstruction instruction;
        instruction.texelContent = stemContent.second;
        instruction.inputIndex   = NoInputIndex;
        instruction.outputIndex  = m_resamplingOutputStorage.size() - 1;
        instructions[meshIdx].instructions.push_back(instruction);
      }
    }
  }

  // Compute required distance textures. Output textures can share distance
  // textures if they have the same size, and the distances written to them
  // in places where mesh islands overlap are exactly the same. Here, we only
  // group distance textures under stricter requirements: in addition to having
  // the same size, textures only share distance textures if they appear in
  // exactly the same set of lo-res meshes*, and glTF textures don't share
  // distances with non-glTF textures (in case KHR_texture_transform affects
  // things in the future).
  //
  // * This implies that the textures are written by the same materials; we
  // assume that if two lo-res meshes use the same materials and we're baking
  // them, then their islands either coincide or don't overlap at all.
  std::vector<glm::ivec2> distanceTextureSizes;
  m_outputToDistanceTextureMap.resize(m_resamplingOutputStorage.size(), InvalidDistanceIndex);
  // Create a map of [output image index] -> [lo-res meshes that use it]
  std::vector<std::set<size_t>> outputImageMeshes(m_resamplingOutputStorage.size());
  for(size_t meshIdx = 0; meshIdx < lowMesh->meshes().size(); meshIdx++)
  {
    for(const ResampleInstruction& instruction : instructions[meshIdx].instructions)
    {
      outputImageMeshes[instruction.outputIndex].insert(meshIdx);
    }
  }
  // Now determine distance textures. Note that this can be quadratic-time!
  for(size_t outputImageIdx = 0; outputImageIdx < m_resamplingOutputStorage.size(); outputImageIdx++)
  {
    if(outputImageMeshes[outputImageIdx].empty())
    {
      continue;
    }

    const VkExtent3D thisSize    = m_resamplingOutputStorage[outputImageIdx].info.extent;
    const bool       isGLTFImage = (outputImageIdx < glTFOutputEnd);
    const size_t     searchStart = isGLTFImage ? 0 : glTFOutputEnd;

    bool sharePrevious = false;
    for(size_t otherOutputIdx = searchStart; otherOutputIdx < outputImageIdx; otherOutputIdx++)
    {
      const VkExtent3D otherSize      = m_resamplingOutputStorage[otherOutputIdx].info.extent;
      const bool       matchingSize   = (thisSize.width == otherSize.width) && (thisSize.height == otherSize.height);
      const bool       matchingMeshes = (outputImageMeshes[outputImageIdx] == outputImageMeshes[otherOutputIdx]);
      if(matchingSize && matchingMeshes)
      {
        m_outputToDistanceTextureMap[outputImageIdx] = m_outputToDistanceTextureMap[otherOutputIdx];
        sharePrevious                                = true;
        break;
      }
    }

    if(!sharePrevious)
    {
      m_outputToDistanceTextureMap[outputImageIdx] = distanceTextureSizes.size();
      distanceTextureSizes.push_back(glm::ivec2(thisSize.width, thisSize.height));
    }
  }

  // Fill instructions:
  for(auto& meshInstructions : instructions)
  {
    for(auto& instruction : meshInstructions.instructions)
    {
      instruction.distanceIndex = m_outputToDistanceTextureMap[instruction.outputIndex];
      assert(instruction.distanceIndex != InvalidDistanceIndex);
    }
  }

  // Prepare distance buffers
  m_resamplingDistanceStorage.resize(distanceTextureSizes.size());
  for(size_t i = 0; i < distanceTextureSizes.size(); ++i)
  {
    GPUTextureContainer& result = m_resamplingDistanceStorage[i];
    result.storageLocation      = GPUTextureContainer::Storage::eCreateOnFirstUse;
    result.info.extent.width    = uint32_t(distanceTextureSizes[i].x);
    result.info.extent.height   = uint32_t(distanceTextureSizes[i].y);
    // Distance images have allocated mips, because we use this space for
    // pull-push filtering.
    result.info.mipLevels = nvvk::mipLevels(result.info.extent);
    result.info.format    = RESAMPLE_DISTANCE_FORMAT;
  }

  // Some output textures might contain both colors and normal maps, in which
  // case we shouldn't use normalization in the pull/push filter. So, mark
  // which ones only had normal map-style writes.
  // We need to know what kind of data the texels in a texture contain so we
  // can apply the correct form of normalization in the pull/push filter. If
  // a texture contains texels with different kinds of data, we must not
  // renormalize. So, keep track of which ones only had certain types of writes.
  std::vector<bool> outputsWrittenOnlyWithNormals(m_resamplingOutputStorage.size(), true);
  std::vector<bool> outputsWrittenOnlyWithQuaternions(m_resamplingOutputStorage.size(), true);
  for(const auto& meshInstructions : instructions)
  {
    for(const auto& instruction : meshInstructions.instructions)
    {
      if(instruction.texelContent != meshops::TextureType::eNormalMap)
      {
        outputsWrittenOnlyWithNormals[instruction.outputIndex] = false;
      }
      if(instruction.texelContent != meshops::TextureType::eQuaternionMap)
      {
        outputsWrittenOnlyWithQuaternions[instruction.outputIndex] = false;
      }
    }
  }

  // Finally, go through the instructions and mark when we use each texture for
  // the last time.
  {
    GPUTextureIndex::UnorderedSet usedTextures;
    auto                          markUse = [&](std::vector<FinalUse>& finalUses, GPUTextureIndex idx) {
      if(usedTextures.find(idx) == usedTextures.end())
      {
        FinalUse finalUse{};
        finalUse.index = idx;
        if(idx.vec == GPUTextureIndex::VectorIndex::eOutput)
        {
          finalUse.onlyContainsNormals     = outputsWrittenOnlyWithNormals[idx.idx];
          finalUse.onlyContainsQuaternions = outputsWrittenOnlyWithQuaternions[idx.idx];
        }
        finalUses.push_back(finalUse);
        usedTextures.insert(idx);
      }
    };
    for(size_t reverseI = 0; reverseI < instructions.size(); reverseI++)
    {
      auto& meshInstructions = instructions[instructions.size() - 1 - reverseI];
      auto& finalUses        = meshInstructions.finalUses;
      for(auto& instruction : meshInstructions.instructions)
      {
        if(instruction.inputIndex != NoInputIndex)
        {
          markUse(finalUses, GPUTextureIndex{GPUTextureIndex::VectorIndex::eInput, instruction.inputIndex});
        }
        markUse(finalUses, GPUTextureIndex{GPUTextureIndex::VectorIndex::eOutput, instruction.outputIndex});
        markUse(finalUses, GPUTextureIndex{GPUTextureIndex::VectorIndex::eDistance, instruction.distanceIndex});
      }
    }
  }

  // Initialize cache data.
  {
    // Generate a random cache file name prefix.
    std::random_device dev;  // Uses true RNG if possible here
    std::stringstream  stream;
    stream << std::hex << ((uint64_t(dev()) << 32) | uint64_t(dev()));
    m_cacheFilePrefix = stream.str();
  }
  m_textureCacheFIFO.clear();
  m_currentTextureMemoryUsage = 0;

  return true;
}

void BakerManager::getTextureMemoryRequirements(uint64_t&                                    minimumRequiredBytes,
                                                uint64_t&                                    idealBytes,
                                                const std::vector<ResampleMeshInstructions>& instructions) const
{
  uint64_t allTexturesBytes  = 0;  // Peak VRAM usage to keep all textures in memory.
  uint64_t maxFrameSizeBytes = 0;  // Maximum VRAM usage for any single hi/lo res pair.
  for(uint32_t meshIndex = 0; meshIndex < instructions.size(); meshIndex++)
  {
    uint64_t                             thisFrameSizeBytes = 0;
    const GPUTextureIndex::UnorderedSet& textures           = getTexturesForMesh(instructions[meshIndex]);
    for(const GPUTextureIndex texIdx : textures)
    {
      thisFrameSizeBytes += getResamplingTexture(texIdx).fullSizeInBytes();
    }

    allTexturesBytes += thisFrameSizeBytes;
    maxFrameSizeBytes = std::max(maxFrameSizeBytes, thisFrameSizeBytes);
  }

  // When loading textures, we need to temporarily have memory for
  // staging buffers.
  minimumRequiredBytes = 2 * maxFrameSizeBytes;
  idealBytes           = allTexturesBytes + maxFrameSizeBytes;
}

bool BakerManager::prepareTexturesForMesh(nvvk::Context::Queue queueGCT, nvvk::Context::Queue queueT, const ResampleMeshInstructions& meshInstructions)
{
  // Start by getting the textures we'll need, the textures to load, and how
  // many bytes we'd use if we loaded them all.
  GPUTextureIndex::UnorderedSet texturesNeeded = getTexturesForMesh(meshInstructions);
  GPUTextureIndex::UnorderedSet texturesToLoad;
  uint64_t                      postLoadBytes       = m_currentTextureMemoryUsage;
  uint64_t                      texturesToLoadBytes = 0;
  for(GPUTextureIndex texIdx : texturesNeeded)
  {
    if(getResamplingTexture(texIdx).storageLocation != GPUTextureContainer::Storage::eVRAM)
    {
      texturesToLoad.insert(texIdx);
      const uint64_t thisSizeBytes = getResamplingTexture(texIdx).fullSizeInBytes();
      texturesToLoadBytes += thisSizeBytes;
      postLoadBytes += thisSizeBytes;
    }
  }
  // Disk cache unneeded textures until we've met our ideal of
  // 2*texturesToLoadBytes free, or we've cached all we can, whichever comes
  // first. In the latter case, we let the loader know it should really try
  // to minimize staging memory usage.
  // We use a FIFO caching policy -- we have the information to implement
  // Belady's optimal algorithm, but this requires less code.
  std::vector<GPUTextureIndex> texturesToDiskCache;
  {
    auto it = m_textureCacheFIFO.begin();
    while((postLoadBytes + 2 * texturesToLoadBytes > m_memLimit) && (it != m_textureCacheFIFO.end()))
    {
      const GPUTextureIndex textureToTry = *it;
      const auto&           tex          = getResamplingTexture(textureToTry);
      if(tex.storageLocation != GPUTextureContainer::Storage::eVRAM)
      {
        it = m_textureCacheFIFO.erase(it);
      }
      else if(texturesNeeded.find(textureToTry) == texturesNeeded.end())
      {
        texturesToDiskCache.push_back(textureToTry);
        postLoadBytes -= tex.fullSizeInBytes();
        it = m_textureCacheFIFO.erase(it);
      }
      else
      {
        ++it;
      }
    }
  }

  if(!cacheResamplingTexturesToDisk(queueT, texturesToDiskCache))
  {
    deleteCachedFilesEarly();
    return false;
  }
  std::vector<GPUTextureIndex> texturesToLoadList;
  texturesToLoadList.insert(texturesToLoadList.end(), texturesToLoad.begin(), texturesToLoad.end());
  if(!loadResamplingTextures(queueGCT, texturesToLoadList))
  {
    deleteCachedFilesEarly();
    return false;
  }

  return true;
}

void BakerManager::finishTexturesForMesh(nvvk::Context::Queue queueGCT, const ResampleMeshInstructions& meshInstructions)
{
  const std::vector<FinalUse>& finalUses = meshInstructions.finalUses;
  if(finalUses.empty())
    return;

  // Initialize the pull-push filter if it's not already initialized.
  if(!m_pullPushFilter.initialized())
  {
    m_pullPushFilter.init(m_device);
    VkShaderModule pullPushShaderModule = nvvk::createShaderModule(m_device, pullpush_comp, sizeof(pullpush_comp));
    m_pullPushFilter.initPipes(m_pullPushFilterPipesGeneral, PullPushFilter::Variant::eStandard, pullPushShaderModule, false);
    m_pullPushFilter.initPipes(m_pullPushFilterPipesNormals, PullPushFilter::Variant::eNormals, pullPushShaderModule, false);
    m_pullPushFilter.initPipes(m_pullPushFilterPipesQuaternions, PullPushFilter::Variant::eQuaternions, pullPushShaderModule, false);
    vkDestroyShaderModule(m_device, pullPushShaderModule, nullptr);
  }

  nvvk::CommandPool cmdPool(m_device, queueGCT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queueGCT.queue);

  std::mutex        gpuAccess;
  std::atomic<bool> allThreadsOk = true;

  // We process these in two steps: writing eOutput textures, and destroying
  // all textures.
  // This is because to correctly fill in gaps in a texture, we need to know
  // which texels are gaps -- i.e. we need to read the distance texture. So we
  // must avoid destroying distance textures before pull-push filtering images
  // that rely on them.

  std::vector<size_t> outputTextureIndices;
  for(size_t i = 0; i < finalUses.size(); i++)
  {
    if(finalUses[i].index.vec == GPUTextureIndex::VectorIndex::eOutput)
    {
      outputTextureIndices.push_back(i);
    }
  }

  nvh::parallel_batches<1>(
      outputTextureIndices.size(),
      [&](uint64_t arrayIdx) {
        const FinalUse        finalUse     = finalUses[outputTextureIndices[arrayIdx]];
        const GPUTextureIndex textureIndex = finalUse.index;
        GPUTextureContainer&  tex          = getResamplingTexture(textureIndex);
        assert(tex.storageLocation == GPUTextureContainer::Storage::eVRAM);
        assert(textureIndex.vec == GPUTextureIndex::VectorIndex::eOutput);

        {
          const uint32_t w = tex.info.extent.width;
          const uint32_t h = tex.info.extent.height;

          // Apply pull-push filtering if possible, then download its data.
          const void* dataMapped;
          {
            // Make sure only one thread enters this section at a time
            std::lock_guard<std::mutex> lock(gpuAccess);

            PullPushFilter::Views pullPushViews;
            VkCommandBuffer       cmdBuf = cmdPool.createCommandBuffer();

            PullPushFilter::ImageInfo pullPushRGBAInfo;
            pullPushRGBAInfo.width       = w;
            pullPushRGBAInfo.height      = h;
            pullPushRGBAInfo.levelCount  = tex.info.mipLevels;
            pullPushRGBAInfo.image       = tex.texture.image;
            pullPushRGBAInfo.imageFormat = tex.info.format;

            const GPUTextureContainer& distanceTex =
                m_resamplingDistanceStorage[m_outputToDistanceTextureMap[textureIndex.idx]];
            PullPushFilter::ImageInfo pullPushDistanceWeightInfo = pullPushRGBAInfo;
            pullPushDistanceWeightInfo.image                     = distanceTex.texture.image;
            pullPushDistanceWeightInfo.imageFormat               = distanceTex.info.format;

            m_pullPushFilter.initViews(pullPushViews, pullPushRGBAInfo, pullPushDistanceWeightInfo);
            const PullPushFilter::Pipes& pipelines = (finalUse.onlyContainsNormals ? m_pullPushFilterPipesNormals :  //
                                                          (finalUse.onlyContainsQuaternions ? m_pullPushFilterPipesQuaternions :  //
                                                               m_pullPushFilterPipesGeneral));
            // Note that this returns true instead of false on failure.
            if(m_pullPushFilter.process(cmdBuf, pipelines, pullPushRGBAInfo, pullPushDistanceWeightInfo, pullPushViews))
            {
              LOGW("Warning: Pull-push filtering %s (%u x %u) failed.", tex.filePath.c_str(), w, h);
            }

            nvvk::cmdBarrierImageLayout(cmdBuf, tex.texture.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            tex.texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            dataMapped                         = downloadImage(*m_alloc, cmdBuf, tex);
            cmdPool.submitAndWait(cmdBuf);

            m_pullPushFilter.deinitViews(pullPushViews);
          }

          // dataMapped is an indirect pointer into GPU memory, which means
          // accesses to it will go over the PCIe bus, which makes our image
          // writer slow. By allocating this buffer, we only need to access
          // each element over PCIe once.
          assert(tex.info.format == VK_FORMAT_R8G8B8A8_UNORM || tex.info.format == VK_FORMAT_R16_UNORM
                 || tex.info.format == VK_FORMAT_R16G16B16A16_UNORM);
          const size_t         mip0SizeBytes = tex.mipSizeInBytes(0);

          // Initialize the ToolImage with the data.
          micromesh_tool::ToolImage::Info toolImageInfo;
          toolImageInfo.width             = tex.info.extent.width;
          toolImageInfo.height            = tex.info.extent.height;
          toolImageInfo.components        = tex.bytesPerPixel() / tex.bytesPerComponent();
          toolImageInfo.componentBitDepth = tex.bytesPerComponent() * 8;
          auto toolImage                  = micromesh_tool::ToolImage::create(toolImageInfo, tex.filePath);
          if(toolImage)
          {
            assert(toolImage->info().totalBytes() == mip0SizeBytes);
            memcpy(toolImage->raw(), dataMapped, mip0SizeBytes);

            // Insert the image into the scene. Aux images are alwaus new and
            // are simply appended to the scene to be saved in the same location
            // as the gltf. All other images have been resampled so we need to
            // replace existing images.
            bool isAuxImage = static_cast<size_t>(textureIndex.idx) >= m_lowMesh->images().size();
            if(isAuxImage)
            {
              m_lowMesh->appendAuxImage(std::move(toolImage));
            }
            else
            {
              // Replace the ToolImage
              assert(textureIndex.idx < m_lowMesh->images().size());
              m_lowMesh->setImage(textureIndex.idx, std::move(toolImage));
            }
          }
          else
          {
            LOGE("Error: Failed to allocate auxiliary image output for %s\n", tex.filePath.c_str());
          }
        }
      },  // If outputTextureIndices.size() is smaller than the number of threads here,
      // nvh::parallel_batches will process them one by one. We want to avoid that.
      std::min(m_numThreads, uint32_t(outputTextureIndices.size())));

  // Destroy all textures
  for(const FinalUse& finalUse : finalUses)
  {
    const GPUTextureIndex textureIndex = finalUse.index;
    GPUTextureContainer&  tex          = getResamplingTexture(textureIndex);
    assert(tex.storageLocation == GPUTextureContainer::Storage::eVRAM);

    m_alloc->destroy(tex.texture);
    m_currentTextureMemoryUsage -= tex.fullSizeInBytes();

    if(textureIndex.vec == GPUTextureIndex::VectorIndex::eOutput)
    {
      tex.storageLocation = GPUTextureContainer::Storage::eToolImage;
    }
    else
    {
      tex.storageLocation = GPUTextureContainer::Storage::eUnknownOrUnused;
    }
  }

  m_alloc->finalizeAndReleaseStaging();

  if(!allThreadsOk.load())
  {
    LOGW("Some resampled images failed to save!\n");
    // But this may be okay; carry on for now.
  }
}

std::vector<ResampleTextureContainer> BakerManager::getResampleTextures(const ResampleMeshInstructions& meshInstructions) const
{
  std::vector<ResampleTextureContainer> resampleTextures;
  for(const ResampleInstruction& instr : meshInstructions.instructions)
  {
    ResampleTextureContainer container;
    container.texelContent = instr.texelContent;
    container.input = (instr.inputIndex == NoInputIndex) ? GPUTextureContainer{} : m_resamplingInputStorage[instr.inputIndex];
    container.output = m_resamplingOutputStorage[instr.outputIndex];
    container.distance = m_resamplingDistanceStorage[instr.distanceIndex];
    resampleTextures.push_back(container);
  }
  return resampleTextures;
}

void BakerManager::destroy()
{
  if(!m_alloc)
    return;
  for(GPUTextureContainer& tex : m_resamplingInputStorage)
  {
    m_alloc->destroy(tex.texture);
  }
  for(GPUTextureContainer& tex : m_resamplingOutputStorage)
  {
    m_alloc->destroy(tex.texture);
  }
  for(GPUTextureContainer& tex : m_resamplingDistanceStorage)
  {
    m_alloc->destroy(tex.texture);
  }

  if(m_pullPushFilter.initialized())
  {
    m_pullPushFilter.deinitPipes(m_pullPushFilterPipesGeneral);
    m_pullPushFilter.deinitPipes(m_pullPushFilterPipesNormals);
    m_pullPushFilter.deinitPipes(m_pullPushFilterPipesQuaternions);
    m_pullPushFilter.deinit();
  }
}

GPUTextureIndex::UnorderedSet BakerManager::getTexturesForMesh(const ResampleMeshInstructions& meshInstructions) const
{
  GPUTextureIndex::UnorderedSet result;
  for(const ResampleInstruction& instr : meshInstructions.instructions)
  {
    if(instr.inputIndex != NoInputIndex)
    {
      result.insert({GPUTextureIndex::VectorIndex::eInput, instr.inputIndex});
    }
    result.insert({GPUTextureIndex::VectorIndex::eOutput, instr.outputIndex});
    result.insert({GPUTextureIndex::VectorIndex::eDistance, instr.distanceIndex});
  }
  return result;
}

bool BakerManager::loadResamplingTextures(nvvk::Context::Queue queueGCT, const std::vector<GPUTextureIndex>& textureIndices)
{
  if(textureIndices.empty())
    return true;

  nvvk::CommandPool cmdPool(m_device, queueGCT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queueGCT.queue);
  VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();

  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  // Load the given texture indices.
  LOGI("Loading or creating %zu images\n", textureIndices.size());

  std::mutex        gpuAccess;
  std::atomic<bool> allThreadsOk = true;

  nvh::parallel_batches<1>(
      textureIndices.size(),
      [&](uint64_t i) {
        const GPUTextureIndex textureIndex = textureIndices[i];
        GPUTextureContainer&  tex          = getResamplingTexture(textureIndex);
        bool                  thisThreadOk = true;

        // Common data pointer for all paths; make sure it's freed!
        imageio::ImageIOData data               = nullptr;
        imageio::ImageIOData ourData            = nullptr;
        const size_t         requiredComponents = 4;
        const size_t         mip0SizeBytes      = tex.mipSizeInBytes(0);
        const size_t         fullSizeBytes      = tex.fullSizeInBytes();

        // Load texture data onto the CPU
        if(tex.storageLocation == GPUTextureContainer::Storage::eImageFile)
        {
          // We should only be loading images from disk for the hi-res mesh.
          assert(textureIndex.vec == GPUTextureIndex::VectorIndex::eInput);

          size_t w = 0, h = 0, comp = 0;
          LOGI("Loading compressed image %s\n", tex.filePath.c_str());
          data = ourData =
              imageio::loadGeneral(tex.filePath.c_str(), &w, &h, &comp, requiredComponents, tex.bytesPerComponent() * 8);
          if(!data)
          {
            LOGE("Error: Failed to load %s!\n", tex.filePath.c_str());
            thisThreadOk = false;
          }
        }
        else if(tex.storageLocation == GPUTextureContainer::Storage::eToolImage)
        {
          // We should only be loading images from disk for the hi-res mesh.
          assert(textureIndex.vec == GPUTextureIndex::VectorIndex::eInput);
          assert(textureIndex.idx < m_highMesh->images().size());
          const micromesh_tool::ToolImage& toolImage = *m_highMesh->images()[textureIndex.idx];
          if(toolImage.info().components == requiredComponents)
          {
            data = toolImage.raw();
            if(!data)
            {
              thisThreadOk = false;
            }
          }
          else
          {
            // TODO: support this
            LOGE("Image has unsupported components (%zu, %zu required)\n", toolImage.info().components, requiredComponents);
            thisThreadOk = false;
          }
        }
        else if(tex.storageLocation == GPUTextureContainer::Storage::eCachedFile)
        {
          // NOTE: Cached data is in a format where we could perform a
          // direct-to-GPU upload here.
          data = ourData                  = imageio::allocateData(mip0SizeBytes);
          const std::string cacheFilename = getCacheFilename(textureIndex);
          LOGI("Loading cached image %s\n", cacheFilename.c_str());
          std::ifstream inputFile(cacheFilename.c_str(), std::ifstream::binary);
          // Note: If the cached file is somehow less than expectedDataSize bytes long,
          // then this will set inputFile's failbit.
          inputFile.read(reinterpret_cast<char*>(data), mip0SizeBytes);
          if(!inputFile)
          {
            LOGE("Error: Failed to load cached file %s\n", cacheFilename.c_str());
            thisThreadOk = false;
          }
          inputFile.close();

          // Try to delete the cached file
          std::error_code err;  // So we don't generate an exception
          fs::remove(cacheFilename, err);
        }
        else if(tex.storageLocation == GPUTextureContainer::Storage::eCreateOnFirstUse)
        {
          // Nothing to do
        }
        else
        {
          assert(!"Unknown or invalid texture storage location! This should never happen if the BakerManager set up the textures correctly.");
        }

        // Critical section: upload to the GPU once it's free.
        if(thisThreadOk)
        {
          const std::lock_guard<std::mutex> lock(gpuAccess);

          const VkExtent2D imageSize{tex.info.extent.width, tex.info.extent.height};

          // Output textures need mipmaps allocated for pull-push filtering to work.
          tex.info = nvvk::makeImage2DCreateInfo(imageSize, tex.info.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                 (tex.info.mipLevels > 1));

          tex.texture = m_alloc->createTexture(cmdBuf, 0, nullptr, tex.info, samplerCreateInfo, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

          if(tex.storageLocation == GPUTextureContainer::Storage::eCreateOnFirstUse)
          {
            // Make sure we clear it - otherwise, our distance buffers could
            // contain uninitialized memory. Also, distance buffers use a
            // different clear color than image buffers.
            VkClearColorValue clearValue;
            if(textureIndex.vec == GPUTextureIndex::VectorIndex::eDistance)
            {
              clearValue = {std::numeric_limits<float>::max(), 0.0f, 0.0f, 0.0f};
            }
            else
            {
              clearValue = {0.0f, 0.0f, 0.0f, 0.0f};
            }
            VkImageSubresourceRange mip0Range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            vkCmdClearColorImage(cmdBuf, tex.texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &mip0Range);
          }
          else
          {
            VkImageSubresourceLayers mip0Layers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            m_alloc->getStaging()->cmdToImage(cmdBuf, tex.texture.image, VkOffset3D{0, 0, 0}, tex.info.extent,
                                              mip0Layers, fullSizeBytes, data);
          }

          nvvk::cmdBarrierImageLayout(cmdBuf, tex.texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
          tex.texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

          m_textureCacheFIFO.push_back(textureIndex);
          m_currentTextureMemoryUsage += fullSizeBytes;
        }

        // Free texture data
        if(ourData != nullptr)
        {
          imageio::freeData(&ourData);
        }

        tex.storageLocation = GPUTextureContainer::Storage::eVRAM;

        if(!thisThreadOk)
        {
          allThreadsOk.store(false);
        }
      },
      std::min(m_numThreads, uint32_t(textureIndices.size())));

  cmdPool.submitAndWait(cmdBuf);
  m_alloc->finalizeAndReleaseStaging();

  return allThreadsOk.load();
}

std::string BakerManager::getCacheFilename(GPUTextureIndex textureIndex) const
{
  return m_cacheFilePrefix + "-" + std::to_string(uint32_t(textureIndex.vec)) + "-" + std::to_string(textureIndex.idx);
}

bool BakerManager::cacheResamplingTexturesToDisk(nvvk::Context::Queue queueT, const std::vector<GPUTextureIndex>& textureIndices)
{
  if(textureIndices.empty())
    return true;
  LOGI("Caching %zu images to disk\n", textureIndices.size());

  nvvk::CommandPool cmdPool(m_device, queueT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, queueT.queue);

  std::mutex        gpuAccess;
  std::atomic<bool> allThreadsOk = true;
  nvh::parallel_batches<1>(
      textureIndices.size(),
      [&](uint64_t i) {
        const GPUTextureIndex textureIndex = textureIndices[i];
        GPUTextureContainer&  tex          = getResamplingTexture(textureIndex);
        assert(tex.storageLocation == GPUTextureContainer::Storage::eVRAM);
        const size_t mip0SizeBytes = tex.mipSizeInBytes(0);

        // Get image data from the GPU
        const void* dataMapped;
        {
          std::lock_guard<std::mutex> lock(gpuAccess);
          VkCommandBuffer             cmdBuf = cmdPool.createCommandBuffer();
          assert(tex.texture.descriptor.imageLayout == VK_IMAGE_LAYOUT_GENERAL);
          nvvk::cmdBarrierImageLayout(cmdBuf, tex.texture.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
          tex.texture.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          dataMapped                         = downloadImage(*m_alloc, cmdBuf, tex);
          cmdPool.submitAndWait(cmdBuf);
        }

        const std::string cachePath = getCacheFilename(textureIndex);
        assert(!fs::exists(cachePath));

        LOGI("Caching %s\n", cachePath.c_str());
        std::ofstream outfile(cachePath.c_str(), std::ofstream::binary | std::ofstream::trunc);
        outfile.write(reinterpret_cast<const char*>(dataMapped), mip0SizeBytes);
        outfile.close();
        if(!outfile)
        {
          LOGE("Caching %s failed!\n", cachePath.c_str());
          allThreadsOk.store(false);
        }

        // Free the image; note that tex.info's resolution is unchanged.
        {
          std::lock_guard<std::mutex> lock(gpuAccess);
          m_alloc->destroy(tex.texture);
          m_currentTextureMemoryUsage -= tex.fullSizeInBytes();
        }

        tex.storageLocation = GPUTextureContainer::Storage::eCachedFile;
      },
      std::min(m_numThreads, uint32_t(textureIndices.size())));

  m_alloc->finalizeAndReleaseStaging();

  return allThreadsOk.load();
}

void BakerManager::deleteCachedFilesEarly()
{
  auto tryDeletes = [this](GPUTextureIndex::VectorIndex vec, size_t len) {
    for(size_t i = 0; i < len; i++)
    {
      const fs::path cachePath = fs::path(getCacheFilename(GPUTextureIndex{vec, i}));
      if(fs::exists(cachePath))
      {
        std::error_code err;
        fs::remove(cachePath, err);
      }
    }
  };

  tryDeletes(GPUTextureIndex::VectorIndex::eInput, m_resamplingInputStorage.size());
  tryDeletes(GPUTextureIndex::VectorIndex::eOutput, m_resamplingOutputStorage.size());
  tryDeletes(GPUTextureIndex::VectorIndex::eDistance, m_resamplingDistanceStorage.size());
}

GPUTextureContainer& BakerManager::getResamplingTexture(GPUTextureIndex idx)
{
  // Uses const version, based on technique from Effective Modern C++:
  return const_cast<GPUTextureContainer&>(const_cast<const BakerManager*>(this)->getResamplingTexture(idx));
}

const GPUTextureContainer& BakerManager::getResamplingTexture(GPUTextureIndex idx) const
{
  switch(idx.vec)
  {
    case GPUTextureIndex::VectorIndex::eInput:
      return m_resamplingInputStorage[idx.idx];
    case GPUTextureIndex::VectorIndex::eOutput:
      return m_resamplingOutputStorage[idx.idx];
    case GPUTextureIndex::VectorIndex::eDistance:
    default:
      return m_resamplingDistanceStorage[idx.idx];
  }
}

}  // namespace tool_bake
