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

#include "gltf.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>
#include "nvmath/nvmath.h"
#include "nvh/nvprint.hpp"
#include "nvh/timesampler.hpp"
#include "gltf/micromesh_util.hpp"
#include "tiny_obj_loader.h"
#include "fileformats/tiny_converter.hpp"
#include "meshops/meshops_operations.h"
#include "mesh_view_conv.hpp"
#include "microutils/microutils.hpp"

namespace fs = std::filesystem;

const std::set<std::string>& micromapExtensionNames()
{
  static const std::set<std::string> extensions{"NV_displacement_micromap", "NV_micromap_tooling", "NV_micromaps"};
  return extensions;
}

int makeView(tinygltf::Model& model, int bufferID, size_t bufferOffsetBytes, size_t sizeBytes, size_t stride, int target)
{
  int viewID = static_cast<int>(model.bufferViews.size());
  {
    tinygltf::BufferView bufferView;
    bufferView.buffer     = bufferID;
    bufferView.byteOffset = bufferOffsetBytes;
    bufferView.byteLength = sizeBytes;
    bufferView.byteStride = stride;
    bufferView.target     = target;
    model.bufferViews.push_back(std::move(bufferView));
  }
  return viewID;
}

int makeAccessor(tinygltf::Model&    model,
                 int                 viewID,
                 size_t              byteOffset,
                 size_t              elementCount,
                 int                 gltfComponentType,
                 int                 gltfType,
                 std::vector<double> minValues,
                 std::vector<double> maxValues)
{
  bool validBounds = true;
  for(auto& x : minValues)
  {
    if(std::isnan(x) || std::isinf(x))
      validBounds = false;
  }
  for(auto& x : maxValues)
  {
    if(std::isnan(x) || std::isinf(x))
      validBounds = false;
  }
  if(!validBounds)
  {
    minValues = std::vector<double>(minValues.size(), -1.0);
    maxValues = std::vector<double>(minValues.size(), 1.0);
    LOGW("Warning: invalid min/max bounds when writing gltf accessor\n");
  }

  int accessorID = static_cast<int>(model.accessors.size());
  {
    tinygltf::Accessor accessor;
    accessor.bufferView    = viewID;
    accessor.byteOffset    = byteOffset;
    accessor.componentType = gltfComponentType;
    accessor.type          = gltfType;
    accessor.count         = elementCount;
    accessor.minValues     = minValues;
    accessor.maxValues     = maxValues;
    model.accessors.push_back(std::move(accessor));
  }
  return accessorID;
}

tinygltf::Primitive tinygltfAppendPrimitive(tinygltf::Model& model, const meshops::MeshView& meshView, bool writeDisplacementMicromapExt)
{
  assert(meshView.triangleCount());
  assert(meshView.vertexCount());
  bool        addsMicromapExt{false};
  const char* extName = writeDisplacementMicromapExt ? NV_DISPLACEMENT_MICROMAP : NV_MICROMAP_TOOLING;
  meshops::ConstArrayView<uint32_t> indices(meshView.triangleVertices);

  // Data is added to the last existing buffer. Create one if it doesn't exist.
  if(model.buffers.empty())
  {
    model.buffers.emplace_back();
  }
  tinygltf::Buffer& buffer   = model.buffers.back();
  int               bufferID = static_cast<int>(model.buffers.size() - 1);

  // This function generates a few buffer views. The first contains the
  // per-vertex attributes in an interleaved layout. The choice of interleaving
  // here is arbitrary; it could be switched to a non-interleaved layout if
  // reasons to favor one or the other arise.
  // The second contains primitive flags, and if writing the
  // NV_micromap_tooling extension, the next one contains subdivision levels.
  // These are separate to avoid alignment issues.
  // Finally, it generates a buffer view for the triangle indices.

  // Put all per-vertex attributes in the same buffer
  size_t vertexAttribBufferOffset;
  size_t vertexAttribBufferSize = meshView.vertexPositions.size() * sizeof(*meshView.vertexPositions.data())
                                  + meshView.vertexNormals.size() * sizeof(*meshView.vertexNormals.data())
                                  + meshView.vertexTexcoords0.size() * sizeof(*meshView.vertexTexcoords0.data())
                                  + meshView.vertexTangents.size() * sizeof(*meshView.vertexTangents.data())
                                  + meshView.vertexDirections.size() * sizeof(*meshView.vertexDirections.data())
                                  + meshView.vertexDirectionBounds.size() * sizeof(*meshView.vertexDirectionBounds.data());
  size_t     vertexAttribOffsetPositions;
  size_t     vertexAttribOffsetNormals;
  size_t     vertexAttribOffsetTexcoords0;
  size_t     vertexAttribOffsetTangents;
  size_t     vertexAttribOffsetDirections;
  size_t     vertexAttribOffsetDirectionBounds;
  size_t     vertexAttribStride{};
  const bool primitiveFlagsExist = !meshView.trianglePrimitiveFlags.empty();
  size_t     primitiveFlagsBufferOffset;
  const bool subdivisionLevelsExist =
      (!meshView.triangleSubdivisionLevels.empty()) && (strcmp(extName, NV_MICROMAP_TOOLING) == 0);
  size_t subdivisionLevelsBufferOffset;
  size_t indicesOffset;
  size_t indicesSize = indices.size() * sizeof(*indices.data());
  {
    // Vertex attrib offsets
    vertexAttribBufferOffset    = buffer.data.size();
    vertexAttribOffsetPositions = 0;
    vertexAttribOffsetNormals =
        vertexAttribOffsetPositions + (meshView.vertexPositions.empty() ? 0 : sizeof(*meshView.vertexPositions.data()));
    vertexAttribOffsetTexcoords0 =
        vertexAttribOffsetNormals + (meshView.vertexNormals.empty() ? 0 : sizeof(*meshView.vertexNormals.data()));
    vertexAttribOffsetTangents =
        vertexAttribOffsetTexcoords0 + (meshView.vertexTexcoords0.empty() ? 0 : sizeof(*meshView.vertexTexcoords0.data()));
    // Skip bitangents. In glTF, they are stored using the .w component of the
    // tangent.
    vertexAttribOffsetDirections =
        vertexAttribOffsetTangents + (meshView.vertexTangents.empty() ? 0 : sizeof(*meshView.vertexTangents.data()));
    vertexAttribOffsetDirectionBounds =
        vertexAttribOffsetDirections + (meshView.vertexDirections.empty() ? 0 : sizeof(*meshView.vertexDirections.data()));
    vertexAttribStride = vertexAttribOffsetDirectionBounds
                         + (meshView.vertexDirectionBounds.empty() ? 0 : sizeof(*meshView.vertexDirectionBounds.data()));
    assert(vertexAttribStride * meshView.vertexCount() == vertexAttribBufferSize);

    // Write vertex attribs
    for(size_t i = 0; i < meshView.vertexCount(); ++i)
    {
      if(!meshView.vertexPositions.empty())
        appendRawElement(buffer.data, meshView.vertexPositions[i]);
      if(!meshView.vertexNormals.empty())
        appendRawElement(buffer.data, meshView.vertexNormals[i]);
      if(!meshView.vertexTexcoords0.empty())
        appendRawElement(buffer.data, meshView.vertexTexcoords0[i]);
      if(!meshView.vertexTangents.empty())
        appendRawElement(buffer.data, meshView.vertexTangents[i]);
      if(!meshView.vertexDirections.empty())
        appendRawElement(buffer.data, meshView.vertexDirections[i]);
      if(!meshView.vertexDirectionBounds.empty())
        appendRawElement(buffer.data, meshView.vertexDirectionBounds[i]);
    }

    // Primitive flags
    if(primitiveFlagsExist)
    {
      primitiveFlagsBufferOffset = buffer.data.size();
      appendRawData(buffer.data, meshView.trianglePrimitiveFlags);
    }

    // Subdivision levels
    if(subdivisionLevelsExist)
    {
      subdivisionLevelsBufferOffset = buffer.data.size();
      appendRawData(buffer.data, meshView.triangleSubdivisionLevels);
    }

    auto indicesBegin = appendRawData(buffer.data, indices);
    // Call begin() on a new line, otherwise it may be evaluated before a realloc.
    indicesOffset = static_cast<size_t>(indicesBegin - buffer.data.begin());
  }

  // Vertex data and layout
  int verticesBufferViewID = makeView(model, bufferID, vertexAttribBufferOffset, vertexAttribBufferSize, vertexAttribStride);
  assert(vertexAttribBufferOffset + vertexAttribBufferSize <= model.buffers.back().data.size());

  // Primitive flags data and layout
  const int primitiveFlagsBufferViewID =
      primitiveFlagsExist ? makeView(model, bufferID, primitiveFlagsBufferOffset,
                                     meshView.trianglePrimitiveFlags.size() * sizeof(uint8_t), sizeof(uint8_t)) :
                            -1;

  // Subdivision levels data and layout
  const int subdivisionLevelsBufferViewID =
      subdivisionLevelsExist ? makeView(model, bufferID, subdivisionLevelsBufferOffset,
                                        meshView.triangleSubdivisionLevels.size() * sizeof(uint16_t), sizeof(uint16_t)) :
                               -1;

  // Indices layout
  int indicesBufferViewID = makeView(model, bufferID, indicesOffset, indicesSize, 0, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);
  assert(indicesOffset + indicesSize <= model.buffers.back().data.size());

  tinygltf::Primitive primitive;
  primitive.mode = TINYGLTF_MODE_TRIANGLES;

  // Triangle indices
  primitive.indices = makeAccessor(model, indices, indicesBufferViewID, 0, 0, meshView.triangleCount() * 3);

  // Standard gltf vertex attributes
  primitive.attributes["POSITION"] = makeAccessor(model, meshView.vertexPositions, verticesBufferViewID,
                                                  vertexAttribOffsetPositions, 0, meshView.vertexCount());
  if(!meshView.vertexNormals.empty())
    primitive.attributes["NORMAL"] = makeAccessor(model, meshView.vertexNormals, verticesBufferViewID,
                                                  vertexAttribOffsetNormals, 0, meshView.vertexCount());
  if(!meshView.vertexTexcoords0.empty())
    primitive.attributes["TEXCOORD_0"] = makeAccessor(model, meshView.vertexTexcoords0, verticesBufferViewID,
                                                      vertexAttribOffsetTexcoords0, 0, meshView.vertexCount());
  if(!meshView.vertexTangents.empty())
    primitive.attributes["TANGENT"] = makeAccessor(model, meshView.vertexTangents, verticesBufferViewID,
                                                   vertexAttribOffsetTangents, 0, meshView.vertexCount());

  // Extension attributes
  tinygltf::Value::Object ext;
  if(!meshView.vertexDirections.empty())
    ext.emplace("directions", makeAccessor(model, meshView.vertexDirections, verticesBufferViewID,
                                           vertexAttribOffsetDirections, 0, meshView.vertexCount()));
  if(!meshView.vertexDirectionBounds.empty())
    ext.emplace("directionBounds", makeAccessor(model, meshView.vertexDirectionBounds, verticesBufferViewID,
                                                vertexAttribOffsetDirectionBounds, 0, meshView.vertexCount()));

  if(primitiveFlagsExist)
    ext.emplace("primitiveFlags", makeAccessor(model, meshView.trianglePrimitiveFlags, primitiveFlagsBufferViewID, 0, 0,
                                               meshView.triangleCount()));

  if(subdivisionLevelsExist)
    ext.emplace("subdivisionLevels", makeAccessor(model, meshView.triangleSubdivisionLevels,
                                                  subdivisionLevelsBufferViewID, 0, 0, meshView.triangleCount()));

  if(!ext.empty())
  {
    addsMicromapExt               = true;
    primitive.extensions[extName] = std::move(tinygltf::Value(ext));
  }

  if(addsMicromapExt)
  {
    setExtensionUsed(model.extensionsUsed, extName, true);
  }

  return primitive;
}

bool copyTinygltfModelExtra(const tinygltf::Model& src, tinygltf::Model& dst, std::set<std::string> extensionFilter)
{
  // Lambdas to filter out names in extensionFilter
  auto extensionNotInFilter = [&extensionFilter](const auto& ext) { return extensionFilter.count(ext.first) == 0; };
  auto copyExtensions       = [&](const auto& srcExtMap, auto& dstExtMap) {
    std::copy_if(srcExtMap.begin(), srcExtMap.end(), std::inserter(dstExtMap, dstExtMap.end()), extensionNotInFilter);
  };

  // Copy everything but mesh data (I.e. all but accessors, buffers, bufferViews, meshes and mesh*.primatives)
  // TODO: this will probably fail when there are dangling buffer views IDs, e.g. from unknown extensions. We should
  //       really be removing replaced meshes from the original, freeing the views, returning those blocks of memory to
  //       a free list and then allocating data for the new meshes.
  auto copyObjects = [&](const auto& srcObjects, auto& dstObjects) {
    dstObjects = srcObjects;

    // Clear and re-copy the extensions, but filtering out those in extensionFilter
    for(size_t i = 0; i < srcObjects.size(); ++i)
    {
      dstObjects[i].extensions.clear();
      copyExtensions(srcObjects[i].extensions, dstObjects[i].extensions);
    }
  };
  copyObjects(src.animations, dst.animations);
  copyObjects(src.materials, dst.materials);
  copyObjects(src.nodes, dst.nodes);
  copyObjects(src.textures, dst.textures);
  copyObjects(src.images, dst.images);
  copyObjects(src.skins, dst.skins);
  copyObjects(src.samplers, dst.samplers);
  copyObjects(src.cameras, dst.cameras);
  copyObjects(src.scenes, dst.scenes);
  copyObjects(src.lights, dst.lights);
  copyExtensions(src.extensions, dst.extensions);

  // Keep extensionsUsed in the destination, in case appendToTinygltfModel()
  // added some. Use a std::set to avoid duplicates when merging.
  std::set<std::string> extensionsUsed(dst.extensionsUsed.begin(), dst.extensionsUsed.end());
  std::copy_if(src.extensionsUsed.begin(), src.extensionsUsed.end(), std::inserter(extensionsUsed, extensionsUsed.begin()),
               [&](auto extName) { return extensionFilter.count(extName) == 0; });
  dst.extensionsUsed = {extensionsUsed.begin(), extensionsUsed.end()};

  // Same for extensionsRequired
  std::set<std::string> extensionsRequired(dst.extensionsRequired.begin(), dst.extensionsRequired.end());
  std::copy_if(src.extensionsRequired.begin(), src.extensionsRequired.end(),
               std::inserter(extensionsRequired, extensionsRequired.begin()),
               [&](auto extName) { return extensionFilter.count(extName) == 0; });
  dst.extensionsRequired = {extensionsRequired.begin(), extensionsRequired.end()};

  // Copy embedded images
  for(auto& image : dst.images)
  {
    if(image.bufferView != -1)
    {
      auto view = src.bufferViews[image.bufferView];
      assert(view.byteStride == 0);
      auto& srcBuffer = src.buffers[view.buffer].data;
      auto& dstBuffer = dst.buffers.back().data;
      auto  offset    = dstBuffer.insert(dstBuffer.end(), srcBuffer.begin() + view.byteOffset,
                                         srcBuffer.begin() + view.byteOffset + view.byteLength);
      image.bufferView =
          makeView(dst, static_cast<int>(dst.buffers.size() - 1), offset - dstBuffer.begin(), dstBuffer.end() - offset);
    }
  }

  return true;
}

void addTinygltfModelLinesMesh(tinygltf::Model&                  model,
                               const std::vector<uint32_t>&      indices,
                               const std::vector<nvmath::vec3f>& positions,
                               const std::string                 meshName,
                               const nvmath::mat4f               transform)
{
  auto& buffer   = model.buffers.back().data;
  auto  bufferID = static_cast<int>(model.buffers.size() - 1);

  auto indicesBegin        = appendRawData(buffer, indices);
  auto indicesOffset       = static_cast<size_t>(indicesBegin - buffer.begin());
  auto indicesSize         = static_cast<size_t>(buffer.end() - indicesBegin);
  int  indicesBufferViewID = makeView(model, bufferID, indicesOffset, indicesSize);

  auto positionsBegin        = appendRawData(buffer, positions);
  auto positionsOffset       = static_cast<size_t>(positionsBegin - buffer.begin());
  auto positionsSize         = static_cast<size_t>(buffer.end() - positionsBegin);
  int  positionsBufferViewID = makeView(model, bufferID, positionsOffset, positionsSize);

  tinygltf::Primitive primitive;
  primitive.mode = TINYGLTF_MODE_LINE;
  primitive.indices = makeAccessor(model, meshops::ConstArrayView<uint32_t>(indices), indicesBufferViewID, 0, 0, indices.size());
  primitive.attributes["POSITION"] =
      makeAccessor(model, meshops::ConstArrayView<nvmath::vec3f>(positions), positionsBufferViewID, 0, 0, positions.size());

  // Create a mesh for the primitive
  int meshID = static_cast<int>(model.meshes.size());
  {
    tinygltf::Mesh mesh;
    mesh.name = meshName;
    mesh.primitives.push_back(primitive);
    model.meshes.push_back(std::move(mesh));
  }

  // Instantiate the mesh
  int nodeID = static_cast<int>(model.nodes.size());
  {
    tinygltf::Node node;
    node.mesh   = meshID;
    node.name   = meshName;
    node.matrix = {transform.mat_array, transform.mat_array + 16};
    model.nodes.push_back(std::move(node));
  }

  // Add the node to the scene, creating one if none exists
  if(model.scenes.empty())
    model.scenes.emplace_back();
  model.scenes.back().nodes.push_back(nodeID);
}
