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

#include "micromesh_util.hpp"
#include <nvh/nvprint.hpp>

void setExtensionUsed(std::vector<std::string>& extensionsUsed, const std::string& extensionName, bool used)
{
  auto it     = std::find(extensionsUsed.begin(), extensionsUsed.end(), extensionName);
  bool exists = it != extensionsUsed.end();
  if(exists && !used)
  {
    extensionsUsed.erase(it);
  }
  else if(!exists && used)
  {
    extensionsUsed.emplace_back(extensionName);
  }
}

bool getPrimitiveDisplacementMicromap(const tinygltf::Primitive& primitive, NV_displacement_micromap& extension)
{
  const auto& ext_map = primitive.extensions.find(NV_DISPLACEMENT_MICROMAP);
  if(ext_map == primitive.extensions.end())
  {
    return false;
  }

  auto& ext = ext_map->second;
  nvh::getInt(ext, "directionBounds", extension.directionBounds);
  nvh::getInt(ext, "directionBoundsOffset", extension.directionBoundsOffset);
  nvh::getInt(ext, "directions", extension.directions);
  nvh::getInt(ext, "directionsOffset", extension.directionsOffset);
  nvh::getInt(ext, "groupIndex", extension.groupIndex);
  nvh::getInt(ext, "mapIndices", extension.mapIndices);
  nvh::getInt(ext, "mapIndicesOffset", extension.mapIndicesOffset);
  nvh::getInt(ext, "mapOffset", extension.mapOffset);
  nvh::getInt(ext, "micromap", extension.micromap);
  nvh::getInt(ext, "primitiveFlags", extension.primitiveFlags);
  nvh::getInt(ext, "primitiveFlagsOffset", extension.primitiveFlagsOffset);
  return true;
}

bool getMaterialsDisplacement(const tinygltf::Material& material, nvh::KHR_materials_displacement& extension)
{
  const auto& ext_map = material.extensions.find(KHR_MATERIALS_DISPLACEMENT_NAME);
  if(ext_map == material.extensions.end())
  {
    return false;
  }

  auto& ext = ext_map->second;
  nvh::getTexId(ext, "displacementGeometryTexture", extension.displacementGeometryTexture);
  nvh::getFloat(ext, "displacementGeometryFactor", extension.displacementGeometryFactor);
  nvh::getFloat(ext, "displacementGeometryOffset", extension.displacementGeometryOffset);
  return true;
}

void setMaterialsDisplacement(const nvh::KHR_materials_displacement& extension, tinygltf::Model& model, tinygltf::Material& material)
{
  tinygltf::Value::Object ext;
  tinygltf::Value::Object index;
  index.emplace("index", extension.displacementGeometryTexture);
  ext.emplace("displacementGeometryTexture", index);
  ext.emplace("displacementGeometryFactor", extension.displacementGeometryFactor);
  ext.emplace("displacementGeometryOffset", extension.displacementGeometryOffset);
  material.extensions[KHR_MATERIALS_DISPLACEMENT_NAME] = std::move(tinygltf::Value(ext));

  if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), KHR_MATERIALS_DISPLACEMENT_NAME)
     == model.extensionsUsed.end())
    model.extensionsUsed.emplace_back(KHR_MATERIALS_DISPLACEMENT_NAME);
}

bool getPrimitiveLegacyBarycentricDisplacement(const tinygltf::Primitive& primitive,
                                               NV_displacement_micromap&  dispExt,
                                               NV_micromap_tooling&       toolExt,
                                               int32_t&                   image)
{
  const auto& ext_map = primitive.extensions.find(NV_LEGACY_BARYCENTRIC_DISPLACEMENT);
  if(ext_map == primitive.extensions.end())
  {
    return false;
  }

  auto& ext = ext_map->second;
  nvh::getInt(ext, "directionBounds", dispExt.directionBounds);
  toolExt.directionBounds = dispExt.directionBounds;
  nvh::getInt(ext, "directionBoundsOffset", dispExt.directionBoundsOffset);
  nvh::getInt(ext, "directions", dispExt.directions);
  toolExt.directions = dispExt.directions;
  nvh::getInt(ext, "directionsOffset", dispExt.directionsOffset);
  nvh::getInt(ext, "groupOffset", dispExt.groupIndex);
  nvh::getInt(ext, "image", image);
  nvh::getInt(ext, "mapIndices", dispExt.mapIndices);
  toolExt.mapIndices = dispExt.mapIndices;
  nvh::getInt(ext, "mapIndicesOffset", dispExt.mapIndicesOffset);
  nvh::getInt(ext, "mapOffset", dispExt.mapOffset);
  toolExt.mapOffset = dispExt.mapOffset;
  nvh::getInt(ext, "subdivisionLevels", toolExt.subdivisionLevels);
  nvh::getInt(ext, "topologyFlags", dispExt.primitiveFlags);
  toolExt.primitiveFlags = dispExt.primitiveFlags;
  nvh::getInt(ext, "topologyFlagsOffset", dispExt.primitiveFlagsOffset);
  return true;
}


void setPrimitiveDisplacementMicromap(tinygltf::Primitive& primitive, const NV_displacement_micromap& extension)
{
  NV_displacement_micromap defaults;
  using namespace tinygltf;
  Value::Object ext;
  if(defaults.directionBounds != extension.directionBounds)
    ext.emplace("directionBounds", extension.directionBounds);
  if(defaults.directionBoundsOffset != extension.directionBoundsOffset)
    ext.emplace("directionBoundsOffset", extension.directionBoundsOffset);
  if(defaults.directions != extension.directions)
    ext.emplace("directions", extension.directions);
  if(defaults.directionsOffset != extension.directionsOffset)
    ext.emplace("directionsOffset", extension.directionsOffset);
  if(defaults.groupIndex != extension.groupIndex)
    ext.emplace("groupIndex", extension.groupIndex);
  if(defaults.micromap != extension.micromap)
    ext.emplace("micromap", extension.micromap);
  if(defaults.mapIndices != extension.mapIndices)
    ext.emplace("mapIndices", extension.mapIndices);
  if(defaults.mapIndicesOffset != extension.mapIndicesOffset)
    ext.emplace("mapIndicesOffset", extension.mapIndicesOffset);
  if(defaults.mapOffset != extension.mapOffset)
    ext.emplace("mapOffset", extension.mapOffset);
  if(defaults.primitiveFlags != extension.primitiveFlags)
    ext.emplace("primitiveFlags", extension.primitiveFlags);
  if(defaults.primitiveFlagsOffset != extension.primitiveFlagsOffset)
    ext.emplace("primitiveFlagsOffset", extension.primitiveFlagsOffset);

  primitive.extensions[NV_DISPLACEMENT_MICROMAP] = std::move(tinygltf::Value(ext));
}

bool getPrimitiveMicromapTooling(const tinygltf::Primitive& primitive, NV_micromap_tooling& extension)
{
  const auto& ext_map = primitive.extensions.find(NV_MICROMAP_TOOLING);
  if(ext_map == primitive.extensions.end())
  {
    return false;
  }

  auto& ext = ext_map->second;
  nvh::getInt(ext, "directionBounds", extension.directionBounds);
  nvh::getInt(ext, "directions", extension.directions);
  nvh::getInt(ext, "mapIndices", extension.mapIndices);
  nvh::getInt(ext, "mapOffset", extension.mapOffset);
  nvh::getInt(ext, "primitiveFlags", extension.primitiveFlags);
  nvh::getInt(ext, "subdivisionLevels", extension.subdivisionLevels);
  return true;
}

void setPrimitiveMicromapTooling(tinygltf::Primitive& primitive, const NV_micromap_tooling& extension)
{
  NV_micromap_tooling defaults;
  using namespace tinygltf;
  Value::Object ext;
  if(defaults.directionBounds != extension.directionBounds)
    ext.emplace("directionBounds", extension.directionBounds);
  if(defaults.directions != extension.directions)
    ext.emplace("directions", extension.directions);
  if(defaults.mapIndices != extension.mapIndices)
    ext.emplace("mapIndices", extension.mapIndices);
  if(defaults.mapOffset != extension.mapOffset)
    ext.emplace("mapOffset", extension.mapOffset);
  if(defaults.primitiveFlags != extension.primitiveFlags)
    ext.emplace("primitiveFlags", extension.primitiveFlags);
  if(defaults.subdivisionLevels != extension.subdivisionLevels)
    ext.emplace("subdivisionLevels", extension.subdivisionLevels);

  primitive.extensions[NV_MICROMAP_TOOLING] = std::move(tinygltf::Value(ext));
}

// Retrieves the NV_micromaps extension as a vector of objects. Returns nullptr
// if it did not exist.
const tinygltf::Value::Array* getNVMicromapExtension(const tinygltf::Model& model)
{
  const auto& it1 = model.extensions.find(NV_MICROMAPS);
  if(it1 == model.extensions.end())
  {
    return nullptr;
  }

  const tinygltf::Value& extValue = it1->second;
  if(!extValue.IsObject() || !extValue.Has("micromaps"))
  {
    return nullptr;
  }

  const tinygltf::Value& micromapsValue = extValue.Get("micromaps");
  if(!micromapsValue.IsArray())
  {
    return nullptr;
  }

  return &micromapsValue.Get<tinygltf::Value::Array>();
}

tinygltf::Value::Array* getNVMicromapExtensionMutable(tinygltf::Model& model)
{
  return const_cast<tinygltf::Value::Array*>(getNVMicromapExtension(model));
}

bool getGLTFMicromapCount(const tinygltf::Model& model, size_t& count)
{
  count                                   = 0;
  const tinygltf::Value::Array* micromaps = getNVMicromapExtension(model);
  if(micromaps == nullptr)
    return false;

  count = micromaps->size();
  return true;
}

bool getGLTFMicromap(const tinygltf::Model& model, int32_t n, NV_micromap& result)
{
  if(n < 0)
    return false;

  const tinygltf::Value::Array* micromaps = getNVMicromapExtension(model);
  if(micromaps == nullptr)
    return false;

  if(static_cast<size_t>(n) >= micromaps->size())
    return false;

  const tinygltf::Value& micromap = micromaps->at(n);
  if(!micromap.IsObject())
    return false;

  nvh::getInt(micromap, "bufferView", result.bufferView);
  if(micromap.Has("uri"))
  {
    result.uri = micromap.Get("uri").Get<std::string>();
  }
  if(micromap.Has("mimeType"))
  {
    result.mimeType = micromap.Get("mimeType").Get<std::string>();
  }
  return true;
}

bool setGLTFMicromap(tinygltf::Model& model, int32_t n, const NV_micromap& extension)
{
  if(n < 0)
    return false;

  tinygltf::Value::Array* micromaps = getNVMicromapExtensionMutable(model);
  if(micromaps == nullptr)
    return false;

  if(static_cast<size_t>(n) >= micromaps->size())
    return false;

  micromaps->at(n) = tinygltf::Value(createTinygltfMicromapObject(extension));
  return true;
}

tinygltf::Value::Object createTinygltfMicromapObject(NV_micromap micromap)
{
  NV_micromap             defaults;
  tinygltf::Value::Object result;
  if(defaults.uri != micromap.uri)
    result.emplace("uri", micromap.uri);
  if(defaults.mimeType != micromap.mimeType)
    result.emplace("mimeType", micromap.uri);
  if(defaults.bufferView != micromap.bufferView)
    result.emplace("bufferView", micromap.bufferView);
  return result;
}

int32_t addTinygltfMicromap(tinygltf::Model& model, const NV_micromap& nvMicromap)
{
  tinygltf::Value& micromapsExtValue = model.extensions[NV_MICROMAPS];
  if(!micromapsExtValue.IsObject())  // If it was default-constructed...
  {
    micromapsExtValue = tinygltf::Value(tinygltf::Value::Object());  // Make it an empty object.
  }

  tinygltf::Value& micromapsArrayValue = micromapsExtValue.Get<tinygltf::Value::Object>()["micromaps"];
  if(!micromapsArrayValue.IsArray())  // If it was default-constructed...
  {
    micromapsArrayValue = tinygltf::Value(tinygltf::Value::Array());  // Make it an empty array.
  }

  tinygltf::Value::Array& micromapsArray = micromapsArrayValue.Get<tinygltf::Value::Array>();
  micromapsArray.emplace_back(createTinygltfMicromapObject(nvMicromap));
  size_t idx = micromapsArray.size() - 1;

  setExtensionUsed(model.extensionsUsed, NV_MICROMAPS, true);

  // The TinyGLTF extension is limited to a 32-bit signed int type.
  if(idx > std::numeric_limits<int32_t>::max())
  {
    assert(!"Trying to add too many micromaps!");
    return -1;
  }
  return static_cast<int32_t>(idx);
}

int32_t addTinygltfMicromap(tinygltf::Model& model, const std::string& micromapUri)
{
  NV_micromap micromap;
  micromap.uri = micromapUri;
  return addTinygltfMicromap(model, micromap);
}

// Updates the glTF `extensionsUsed` list if any primitives used any extensions.
// Since tinyGLTF's extensionsUsed is a vector, this searches through it once
// instead of for every primitive, but maybe the cost isn't that much.
void updateExtensionsUsed(tinygltf::Model& model)
{
  bool hadNVMicromaps            = (model.extensions.find(NV_MICROMAPS) != model.extensions.end());
  bool hadNVDisplacementMicromap = false;
  bool hadNVMicromapTooling      = false;

  for(const auto& mesh : model.meshes)
  {
    for(const auto& prim : mesh.primitives)
    {
      if(prim.extensions.find(NV_DISPLACEMENT_MICROMAP) != prim.extensions.end())
      {
        hadNVDisplacementMicromap = true;
      }
      if(prim.extensions.find(NV_MICROMAP_TOOLING) != prim.extensions.end())
      {
        hadNVMicromapTooling = true;
      }
    }
  }

  bool nvMicromapsListed            = false;
  bool nvDisplacementMicromapListed = false;
  bool micromapToolingListed        = false;
  for(const std::string& ext : model.extensionsUsed)
  {
    if(!nvMicromapsListed && ext == NV_MICROMAPS)
    {
      nvMicromapsListed = true;
    }
    if(!nvDisplacementMicromapListed && ext == NV_DISPLACEMENT_MICROMAP)
    {
      nvDisplacementMicromapListed = true;
    }
    if(!micromapToolingListed && ext == NV_MICROMAP_TOOLING)
    {
      micromapToolingListed = true;
    }
  }

  if(hadNVMicromaps && !nvMicromapsListed)
  {
    model.extensionsUsed.push_back(NV_MICROMAPS);
  }
  if(hadNVDisplacementMicromap && !nvDisplacementMicromapListed)
  {
    model.extensionsUsed.push_back(NV_DISPLACEMENT_MICROMAP);
  }
  if(hadNVMicromapTooling && !micromapToolingListed)
  {
    model.extensionsUsed.push_back(NV_MICROMAP_TOOLING);
  }
}

bool updateNVBarycentricDisplacementToNVDisplacementMicromap(tinygltf::Model& model)
{
  constexpr int32_t NO_ENTRY = -1;
  // For each original glTF image, its new index in the micromaps array iff it
  // was referenced by a micromap extension; -1 otherwise.
  std::vector<int32_t> imageToNewMicromap(model.images.size(), NO_ENTRY);
  // Iterate over all primitives, marking which images were referenced,
  // updating extensions, and copying micromaps to the new micromaps array.
  for(tinygltf::Mesh& mesh : model.meshes)
  {
    for(tinygltf::Primitive& prim : mesh.primitives)
    {
      NV_displacement_micromap displacementExtension;
      NV_micromap_tooling      toolingExtension;
      int32_t                  baryIndexInImages{-1};
      if(getPrimitiveLegacyBarycentricDisplacement(prim, displacementExtension, toolingExtension, baryIndexInImages))
      {
        // Remove the legacy extension.
        prim.extensions.erase(NV_LEGACY_BARYCENTRIC_DISPLACEMENT);
        // If there's an `image`, it's now NV_displacement_micromap;
        // otherwise, it was used to store baking data and it's now
        // NV_micromap_tooling.
        const bool wasDisplacementMicromap = (baryIndexInImages >= 0);
        if(wasDisplacementMicromap)
        {
          if(static_cast<size_t>(baryIndexInImages) >= model.images.size())
          {
            // This extension was invalid! Reject it
            LOGE(
                "%s: A primitive using the NV_barycentric_displacement extension referenced image index %i, which was "
                "out of bounds (the images array contains %zu items). This is not a valid use of the extension.\n",
                __func__, baryIndexInImages, model.images.size());
            return false;
          }

          // Did we already turn this image into a micromap?
          if(imageToNewMicromap[baryIndexInImages] != NO_ENTRY)
          {
            // Then all we need is the new index.
            displacementExtension.micromap = imageToNewMicromap[baryIndexInImages];
            setPrimitiveDisplacementMicromap(prim, displacementExtension);
            continue;
          }

          // Create the micromaps array if it wasn't there already,
          // then copy the image to the new array. We'll remove the old image later.
          tinygltf::Image micromapAsImage = model.images[baryIndexInImages];
          NV_micromap     nvMicromap;
          nvMicromap.uri             = micromapAsImage.uri;
          nvMicromap.mimeType        = micromapAsImage.mimeType;
          nvMicromap.bufferView      = micromapAsImage.bufferView;
          const size_t micromapIndex = addTinygltfMicromap(model, nvMicromap);

          displacementExtension.micromap        = static_cast<int32_t>(micromapIndex);
          imageToNewMicromap[baryIndexInImages] = displacementExtension.micromap;

          // Add the new extension.
          setPrimitiveDisplacementMicromap(prim, displacementExtension);
        }
        else
        {
          // Add the tooling extension.
          setPrimitiveMicromapTooling(prim, toolingExtension);
        }
      }
    }
  }

  // Remove images that previously contained referenced .bary files.
  // Previously, we didn't re-index images, but it turns out TinyGLTF refuses
  // to save files with null images (since it is invalid glTF). So we have to
  // re-index them. However, textures should never have pointed to a .bary
  int32_t              numOutputImages = 0;
  std::vector<int32_t> inputImageToNewImage(model.images.size(), NO_ENTRY);
  for(int32_t inputImage = 0; inputImage < int32_t(model.images.size()); inputImage++)
  {
    if(imageToNewMicromap[inputImage] == NO_ENTRY)
    {
      // This copy from model.images to itself is safe, since we guarantee
      // numOutputImages <= inputImage.
      assert(numOutputImages <= inputImage);
      model.images[numOutputImages]    = model.images[inputImage];
      inputImageToNewImage[inputImage] = numOutputImages;
      numOutputImages++;
    }
  }
  model.images.resize(numOutputImages);

  for(size_t textureIdx = 0; textureIdx < model.textures.size(); textureIdx++)
  {
    tinygltf::Texture& tex = model.textures[textureIdx];
    // If tex.source wasn't specified, before, that's OK. Maybe there's an
    // extension that provided its data?
    if(tex.source < 0)
    {
      LOGW(
          "%s: Texture %zu did not specify a source. If one of its extensions had an index to an image, it may no "
          "longer be valid.\n",
          __func__, textureIdx);
      continue;
    }
    // Additionally, some models (e.g. media/cornellBox.gltf) have no images,
    // but also have textures where the source is set to 0. Let's accept them
    // but print a warning for now.
    if(size_t(tex.source) >= inputImageToNewImage.size())
    {
      LOGW("%s: Texture %zu's source field (%i) was greater than the number of images (%zu).\n", __func__, textureIdx,
           tex.source, inputImageToNewImage.size());
      continue;
    }
    // It's an error if tex.source pointed to a .bary image.
    if(size_t(tex.source) >= inputImageToNewImage.size() || inputImageToNewImage[tex.source] == NO_ENTRY)
    {
      LOGE(
          "%s: Texture %zu's source field (%i) pointed to an image that was used by a primitive for the legacy "
          "NV_barycentric_displacement extension. This is invalid, because in NV_barycentric_displacement, images "
          "could only be used as either micromaps or as textures, not both.\n",
          __func__, textureIdx, tex.source);
      return false;
    }

    tex.source = inputImageToNewImage[tex.source];
  }

  updateExtensionsUsed(model);
  return true;
}
