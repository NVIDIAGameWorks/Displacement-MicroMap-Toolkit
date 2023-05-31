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

#include "bary/bary_types.h"
#include "imageio/imageio.hpp"
#include "tool_image.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <tool_scene.hpp>
#include <unordered_map>
#include <nvmath/nvmath_types.h>
#include <tool_meshops_objects.hpp>
#include <tool_bary.hpp>
#include <gltf.hpp>
#include <gltf/micromesh_util.hpp>
#include <meshops/meshops_mesh_view.h>
#include <nvmath/nvmath.h>
#include <tiny_gltf.h>
#include <nvh/nvprint.hpp>
#include <nvh/boundingbox.hpp>
#include <tiny_obj_loader.h>
#include <fileformats/tiny_converter.hpp>
#include <gltf/NV_micromesh_extension_types.hpp>
#include <sstream>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include <dlib_url.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace micromesh_tool {

// Recursive call to build instance world matrices from the gltf node hierarchy
static void createInstances(const tinygltf::Model&            model,
                            std::vector<std::vector<int>>     gltfMeshToToolMeshes,
                            int                               nodeID,
                            nvmath::mat4f                     parentMatrix,
                            std::vector<ToolScene::Instance>& instances)
{
  const tinygltf::Node& node   = model.nodes[nodeID];
  nvmath::mat4f         matrix = parentMatrix * nvh::getLocalMatrix(node);

  // Create an instances if the node has a valid mesh - typically leaf nodes.
  if(node.mesh >= 0)
  {
    // Each gltf Mesh can have multiple Primitives, which map to a single
    // ToolMesh. That means multiple ToolScene instances may be needed to
    // instantiate a gltf Mesh.
    auto& gltfPrimitives  = model.meshes[node.mesh].primitives;
    auto& toolMeshIndices = gltfMeshToToolMeshes[node.mesh];
    assert(gltfPrimitives.size() == toolMeshIndices.size());
    for(size_t i = 0; i < gltfPrimitives.size(); ++i)
    {
      ToolScene::Instance instance;
      instance.worldMatrix = matrix;
      instance.mesh        = toolMeshIndices[i];
      instance.name        = node.name;
      instance.gltfNode    = nodeID;
      instances.push_back(instance);
    }
  }

  // Create instances for all children of this node, accumulating the
  // transformation matrix.
  for(int child : node.children)
  {
    createInstances(model, gltfMeshToToolMeshes, child, matrix, instances);
  }
}

//--------------------------------------------------------------------------------------------------
// Making a unique key for a glTF primitive
// Primitives can be reused by mesh with different materials, we are only interested by the
// data/geometry.
static std::string makePrimitiveKey(const tinygltf::Primitive& primitive)
{
  // TODO: add bary index+group to the key
  std::vector<std::string> attributes{"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT", "COLOR_0"};
  std::stringstream        o;
  o << primitive.indices;
  for(auto& a : attributes)
  {
    auto accessorIt = primitive.attributes.find(a);
    if(accessorIt != primitive.attributes.end())
    {
      o << ":" << accessorIt->second;
    }
  }
  return o.str();
}

std::unique_ptr<ToolScene> ToolScene::create(const fs::path& filename)
{
  auto model = std::make_unique<tinygltf::Model>();
  if(!micromesh_tool::loadTinygltfModel(filename, *model))
  {
    LOGE("Error: Failed to load '%s'\n", filename.string().c_str());
    return {};
  }

  if(!updateNVBarycentricDisplacementToNVDisplacementMicromap(*model))
  {
    return {};
  }

  // Wrap it with ToolScene to provide abstract mesh views and aux data storage
  auto basePath = fs::path(filename).parent_path();

  return ToolScene::create(std::move(model), basePath);
}

std::unique_ptr<ToolScene> ToolScene::create(std::unique_ptr<tinygltf::Model> model, const fs::path basePath)
{
  auto result = std::unique_ptr<ToolScene>(new ToolScene(std::move(model), basePath));
  if(result->meshes().size() == 0)
  {
    LOGE("Error: Creating a scene with no meshes\n");
    result.reset();
  }
  return result;
}

std::unique_ptr<ToolScene> ToolScene::create(std::unique_ptr<tinygltf::Model>          model,
                                             std::vector<std::unique_ptr<ToolImage>>&& images,
                                             std::vector<std::unique_ptr<ToolBary>>&&  barys)
{
  auto result = std::unique_ptr<ToolScene>(new ToolScene(std::move(model), std::move(images), std::move(barys)));
  if(result->meshes().size() == 0)
  {
    LOGE("Error: Creating a scene with no meshes\n");
    result.reset();
  }
  return result;
}

std::unique_ptr<ToolScene> ToolScene::create(const std::unique_ptr<ToolScene>& source)
{
  auto result = std::make_unique<ToolScene>();

  // Copy all the data that ToolScene cannot represent. Do not copy buffers (or
  // buffer views or accessors) since the mesh data is likely stale. ToolMesh
  // contains the ground truth even though initially it has pointers into the
  // gltf buffers.
  // TODO: This is probably not safe to do blindly as there are possibly many
  // gltf features relying on data in discarded buffers. We should instead only
  // keep data we know we can write out safely.
  auto& model               = result->m_model;
  model                     = std::make_unique<tinygltf::Model>();
  model->animations         = source->model().animations;
  model->materials          = source->model().materials;
  model->meshes             = source->model().meshes;  // Keep primitive extensions to preserve micromap references
  model->nodes              = source->model().nodes;
  model->textures           = source->model().textures;
  model->images             = source->model().images;  // TODO: don't copy images after adding ToolTexture?
  model->skins              = source->model().skins;
  model->samplers           = source->model().samplers;
  model->cameras            = source->model().cameras;
  model->scenes             = source->model().scenes;
  model->lights             = source->model().lights;
  model->extensions         = source->model().extensions;  // Micromaps are copied here
  model->extensionsUsed     = source->model().extensionsUsed;
  model->extensionsRequired = source->model().extensionsRequired;
  model->asset              = source->model().asset;
  model->extras             = source->model().extras;

  // Copy all ToolMesh objects
  for(auto& mesh : source->meshes())
  {
    // Copy the mesh from the other scene, which may reference mixed tinygltf
    // data and new data, into the aux MeshData container of a new ToolMesh
    result->m_meshes.push_back(std::make_unique<ToolMesh>(*mesh));
  }

  // Copy all Micromap objects
  for(auto& bary : source->m_barys)
  {
    auto copy = ToolBary::create(*bary);
    if(!copy)
    {
      LOGE("Error: failed to duplicate a ToolBary\n")
      return {};
    }
    result->m_barys.push_back(std::move(copy));
  }

  // Copy all image objects
  for(auto& image : source->m_images)
  {
    auto copy = ToolImage::create(*image);
    if(!copy)
    {
      LOGE("Error: failed to duplicate ToolImage %s. Ignoring.\n", image->relativePath().string().c_str())
    }
    result->m_images.push_back(std::move(copy));
  }

  // Copy all auxiliary image objects
  for(auto& image : source->m_auxImages)
  {
    auto copy = ToolImage::create(*image);
    if(!copy)
    {
      LOGE("Error: failed to duplicate auxiliary ToolImage %s. Ignoring.\n", image->relativePath().string().c_str())
    }
    result->m_auxImages.push_back(std::move(copy));
  }

  result->m_instances = source->m_instances;
  return result;
}

bool loadTinygltfModel(const fs::path& filename, tinygltf::Model& model)
{
  bool               result = false;
  tinygltf::TinyGLTF loader;

  auto ext = filename.extension().string();
  loader.RemoveImageLoader();

  if(ext == ".obj")
  {
    tinyobj::ObjReader objReader;
    if(objReader.ParseFromFile(filename.string()))
    {
      TinyConverter conv;
      conv.convert(model, objReader);
      result = true;
    }
  }
  else
  {
    std::string error;
    std::string warn;
    if(ext == ".gltf")
    {
      result = loader.LoadASCIIFromFile(&model, &error, &warn, filename.string());
    }
    else if(ext == ".glb")
    {
      result = loader.LoadBinaryFromFile(&model, &error, &warn, filename.string());
    }
    if(!warn.empty() || !error.empty())
    {
      LOGE("Warn: %s\n", warn.c_str());
      LOGE("Err: %s\n", error.c_str());
    }
  }

  if(!result)
    return false;

  // Translate legacy usage of NV_barycentric_displacement to NV_displacement_micromap
  return updateNVBarycentricDisplacementToNVDisplacementMicromap(model);
}

bool saveTinygltfModel(const fs::path& filename, tinygltf::Model& model)
{
  auto ext   = filename.extension().string();
  bool isGLB = false;
  if(ext == ".gltf")
  {
    isGLB = false;
  }
  else if(ext == ".glb")
  {
    isGLB = true;
  }
  else
  {
    LOGW(
        "Warning: Not sure whether the filename extension (%s) refers to an ASCII or binary glTF file. Assuming "
        "ASCII.\n",
        ext.c_str());
  }

  // Note(nbickford): Perhaps we should call updateExtensionsUsed() here?
  // Not sure whether to give this code responsibility to fix extension lists.

  // Clear existing buffer URI strings, so they are regenerated based on the output filename, rather than clobbering
  // the old one that they were loaded from.
  for(auto& buffer : model.buffers)
    buffer.uri.clear();

  LOGI("Writing %s\n", filename.string().c_str());

  tinygltf::TinyGLTF writer;
  writer.SetImageWriter(nullptr, nullptr);                   // Don't modify images
  if(!writer.WriteGltfSceneToFile(&model, filename.string(), /* filename */
                                  isGLB,                     /*embedImages*/
                                  isGLB,                     /*embedBuffers*/
                                  true,                      /*prettyPrint*/
                                  isGLB))                    /*writeBinary*/
  {
    LOGE("Error: Could not write glTF scene to %s.\n", filename.string().c_str());
    return false;
  }
  return true;
}

void ToolScene::write(tinygltf::Model& output, const std::set<std::string>& extensionFilter, bool writeDisplacementMicromapExt)
{
  // Copy non-mesh data. This clobbers anything previously on the output model
  // except mesh data.
  copyTinygltfModelExtra(model(), output, extensionFilter);

  // Fill and rewrite data from ToolScene
  rewriteMeshes(output, extensionFilter, writeDisplacementMicromapExt);
  rewriteBarys(output);
  rewriteImages(output);
}

void ToolScene::rewriteMeshes(tinygltf::Model& output, const std::set<std::string>& extensionFilter, bool writeDisplacementMicromapExt)
{
  // Rebuild gltfMeshToToolMeshes from createViews(), so that we can write
  // m_meshes in the same structure that they were originally created from. This
  // complexity exists because a gltf mesh has multiple primitives, which can
  // also share data. An alternative would be to store this mapping.
  std::vector<std::vector<int>> gltfMeshToToolMeshes;
  {
    int                                       nextMeshIndex = 0;
    std::unordered_map<std::string, uint32_t> uniqueMeshPrim;
    for(auto& mesh : m_model->meshes)
    {
      gltfMeshToToolMeshes.emplace_back();
      for(auto& primitive : mesh.primitives)
      {
        // A key is made to consolidate multiple gltf mesh primitives that
        // reference identical data.
        std::string key = makePrimitiveKey(primitive);

        // Check if the mesh primitive has been added yet
        auto it = uniqueMeshPrim.find(key);
        if(it == uniqueMeshPrim.end())
        {
          // Keep info of the primitive reference for mesh instance
          it = uniqueMeshPrim.insert({key, nextMeshIndex++}).first;
        }
        gltfMeshToToolMeshes.back().push_back(it->second);
      }
    }
  }

  // Lambdas to filter out names in extensionFilter
  auto copyExtensions = [&](const tinygltf::ExtensionMap& srcExtMap, tinygltf::ExtensionMap& dstExtMap) {
    for(const auto& scrExt : srcExtMap)
    {
      if(extensionFilter.count(scrExt.first) == 0)
      {
        dstExtMap.insert(scrExt);
      }
    }
  };

  // Write all meshes to the output model. This is done in two stages. First,
  // each ToolMesh is added to buffers, returning a gltf Primitive for each.
  assert(output.meshes.empty());     // output must be a new object
  assert(!m_model->meshes.empty());  // mesh structure is duplicated from the original gltf
  std::vector<tinygltf::Primitive> primitives;
  primitives.reserve(m_meshes.size());
  for(auto& mesh : m_meshes)
  {
    tinygltf::Primitive primitive = tinygltfAppendPrimitive(output, mesh->view(), writeDisplacementMicromapExt);

    ToolMesh::Relations& relations = mesh->relations();

    // Verify bary and group indices.
    assert(relations.bary == -1 || (relations.bary >= 0 && static_cast<size_t>(relations.bary) < m_barys.size()));
    assert(relations.bary == -1
           || (relations.group >= 0 && static_cast<size_t>(relations.group) < m_barys[relations.bary]->groups().size()));

    // Copy back the material ID. Between loading and saving the definitive
    // version lives on ToolMesh::Relations.
    primitive.material = relations.material;

    // Add any micromesh references for each mesh+primitive as they're written.
    // Note outputPrimitive is different to the ToolMesh primitive, which
    // belongs to the original ToolScene input file.
    if(relations.bary == -1)
    {
      // Unlikely but clear any existing extension just to be safe
      primitive.extensions.erase(NV_DISPLACEMENT_MICROMAP);
    }
    else
    {
      NV_displacement_micromap displacementMicromap;
      getPrimitiveDisplacementMicromap(primitive, displacementMicromap);
      displacementMicromap.micromap   = relations.bary;
      displacementMicromap.groupIndex = relations.group;
      displacementMicromap.mapOffset  = relations.mapOffset;
      setPrimitiveDisplacementMicromap(primitive, displacementMicromap);
    }

    primitives.push_back(std::move(primitive));
  }

  // Second stage of writing meshes. Gltf Primitives are written to gltf Mesh
  // structures based on gltfMeshToToolMeshes. Also copy gltf mesh and
  // mesh.primitive metadata from the original gltf. Other gltf objects are copied
  // by copyTinygltfModelExtra().
  for(size_t gltfMeshIndex = 0; gltfMeshIndex < gltfMeshToToolMeshes.size(); ++gltfMeshIndex)
  {
    tinygltf::Mesh& originalMesh = m_model->meshes[gltfMeshIndex];
    tinygltf::Mesh  outputMesh;
    assert(!gltfMeshToToolMeshes[gltfMeshIndex].empty());
    assert(originalMesh.primitives.size() == gltfMeshToToolMeshes[gltfMeshIndex].size());
    for(size_t i = 0; i < gltfMeshToToolMeshes[gltfMeshIndex].size(); ++i)
    {
      tinygltf::Primitive& originalPrimitive = originalMesh.primitives[i];
      tinygltf::Primitive& outputPrimitive   = primitives[gltfMeshToToolMeshes[gltfMeshIndex][i]];
      outputMesh.primitives.push_back(outputPrimitive);
      outputMesh.primitives.back().extras = originalPrimitive.extras;
      outputMesh.primitives.back().mode   = originalPrimitive.mode;
      copyExtensions(originalPrimitive.extensions, outputMesh.primitives.back().extensions);
    }

    // A gltf Mesh has multiple primitives, which may produce multiple ToolMesh,
    // each with its own Meta. Use the first when there are multiple.
    const ToolMesh::Meta& firstMeta = m_meshes[gltfMeshToToolMeshes[gltfMeshIndex][0]]->meta();

    // Restore gltf Mesh metadata
    outputMesh.name    = firstMeta.name;
    outputMesh.extras  = originalMesh.extras;
    outputMesh.weights = originalMesh.weights;
    copyExtensions(originalMesh.extensions, outputMesh.extensions);
    output.meshes.push_back(std::move(outputMesh));
  }

  // Update instance metadata
  for(auto& instance : m_instances)
  {
    assert(instance.gltfNode != -1);
    if(output.nodes[instance.gltfNode].name != instance.name)
    {
      LOGI("Renamed node %i from %s to %s\n", instance.gltfNode, output.nodes[instance.gltfNode].name.c_str(),
           instance.name.c_str());
      output.nodes[instance.gltfNode].name = instance.name;
    }
  }
}

void ToolScene::rewriteBarys(tinygltf::Model& output)
{
  // Clear all the gltf micromaps as we always rewrite them (some stale ones may
  // have been copied by copyTinygltfModelExtra).
  tinygltf::Value::Array* gltfMicromaps = getNVMicromapExtensionMutable(*m_model);
  if(gltfMicromaps)
  {
    gltfMicromaps->clear();
  }

  // Add a micromap entry in the gltf for each bary file
  size_t micromapCount = 0;
  (void)getGLTFMicromapCount(output, micromapCount);
  assert(micromapCount == 0);
  for(size_t i = 0; i < m_barys.size(); ++i)
  {
    auto&       bary = m_barys[i];
    NV_micromap micromapExt{};
    if(bary->relativePath().empty())
    {
      // ToolBary::save() has not been called. This may happen when temporarily
      // writing to a gltf model in memory, e.g. in tool_merge.
      micromapExt.uri = "not_saved.bary";
    }
    else
    {
      micromapExt.uri = bary->relativePath().string();
    }
    int32_t gltfMicromapIndex = addTinygltfMicromap(output, micromapExt);
    if(static_cast<size_t>(gltfMicromapIndex) != i)
    {
      assert(false);
      LOGE("Error: addTinygltfMicromap() added at unexpected index %i\n", gltfMicromapIndex);
    }
  }

  // Warn if there is both bary and heightmap displacement
  for(auto& mesh : output.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      NV_displacement_micromap displacementMicromap;
      if(!getPrimitiveDisplacementMicromap(primitive, displacementMicromap))
      {
        continue;
      }
      if(displacementMicromap.micromap != -1 && primitive.material != -1)
      {
        tinygltf::Material& material = m_model->materials[primitive.material];
        if(material.extensions.count("KHR_materials_displacement") > 0)
        {
          LOGW("Warning: mesh with bary displacement also contains a reference to KHR_materials_displacement\n");
        }
      }
    }
  }

  // Update the extensions used
  bool hasHeightmapDisplacement = false;
  bool hasNVDispacementMicromap = false;
  bool hasNVMicromapTooling     = false;
  for(auto& mesh : output.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      NV_displacement_micromap displacementMicromap;
      if(getPrimitiveDisplacementMicromap(primitive, displacementMicromap))
      {
        hasNVDispacementMicromap = true;
      }
      NV_micromap_tooling micromapTooling;
      if(getPrimitiveMicromapTooling(primitive, micromapTooling))
      {
        hasNVMicromapTooling = true;
      }
    }
  }
  for(const auto& material : output.materials)
  {
    nvh::KHR_materials_displacement displacement;
    if(getMaterialsDisplacement(material, displacement))
    {
      hasHeightmapDisplacement = true;
    }
  }
  assert(m_barys.empty() == !hasNVDispacementMicromap);
  setExtensionUsed(output.extensionsUsed, "KHR_materials_displacement", hasHeightmapDisplacement);
  setExtensionUsed(output.extensionsUsed, NV_DISPLACEMENT_MICROMAP, hasNVDispacementMicromap);
  setExtensionUsed(output.extensionsUsed, NV_MICROMAP_TOOLING, hasNVMicromapTooling);
  setExtensionUsed(output.extensionsUsed, NV_MICROMAPS, hasNVDispacementMicromap || hasNVMicromapTooling);
}

void ToolScene::rewriteImages(tinygltf::Model& output)
{
  // Replace images with those from the ToolScene.
  output.images.clear();
  output.images.resize(m_images.size());
  for(size_t i = 0; i < m_images.size(); ++i)
  {
    auto& image     = m_images[i];
    auto& gltfImage = output.images[i];
    if(!image->relativePath().empty())
    {
      // Add a reference to the relative location on disk. This will either be
      // the original texture location when loading or the last saved location.
      gltfImage.uri  = image->relativePath().string();
      gltfImage.name = image->relativePath().stem().string();
    }
    else
    {
      // Not implemented. The below doesn't work because tinygltf is not
      // compiled with stbimage.
      {
        LOGE("Error: image %zu has no relative path and embedding images has not been implemented\n", i);
        continue;
      }

      // The image has no relative path. This is typically because it came from
      // an embedded image. Add it back to the tinygltf structur to be embedded.
      gltfImage.width     = static_cast<int>(image->info().width);
      gltfImage.height    = static_cast<int>(image->info().height);
      gltfImage.component = static_cast<int>(image->info().components);
      gltfImage.bits      = static_cast<int>(image->info().componentBitDepth);
      switch(gltfImage.bits)
      {
        case 8:
          gltfImage.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
          break;
        case 16:
          gltfImage.pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
          break;
        default:
          LOGE("Cannot embed image with unsupported %i bits per components\n", gltfImage.bits);
          continue;
      }
      uint8_t* rawBytes = static_cast<uint8_t*>(image->raw());
      gltfImage.image   = {rawBytes, rawBytes + image->info().totalBytes()};
    }
  }
}

bool ToolScene::save(const fs::path& filename)
{
  auto basePath = fs::absolute(filename).parent_path();

  // Save all bary files. Filenames are based on the filename stem, numbered if
  // there are multiple.
  for(size_t i = 0; i < m_barys.size(); ++i)
  {
    auto&    bary         = m_barys[i];
    fs::path relativePath = bary->relativePath();
    if(relativePath.empty())
    {
      relativePath = filename.stem().string() + (m_barys.size() > 1 ? std::to_string(i) : std::string()) + ".bary";
    }
    if(!m_barys[i]->save(basePath, relativePath))
    {
      return false;
    }
  }

  // Saving generated images sets the "original data" state, so record it first
  bool originalImages = isOriginalImageData();

  // Save all the images. If there is no relative filename, it must have come
  // from an embedded gltf image and should be returned there.
  for(auto& image : m_images)
  {
    // Ignore failures. Errors will be printed.
    (void)image->save(basePath, image->relativePath());
  }

  // Save all the aux images, which are not referenced by the gltf.
  for(auto& image : m_auxImages)
  {
    // Ignore failures. Errors will be printed.
    (void)image->save(basePath, image->relativePath());

    if(image->relativePath().empty())
    {
      assert(false);
      LOGE("Error: auxiliary image has no path and will not be be embedded\n");
    }
  }

  // If any mesh attributes were generated, we need to re-create the gltf model,
  // combining both the original and the new data.
  tinygltf::Model  rewrittenModel;
  tinygltf::Model* outModel = m_model.get();
  if(!isOriginalMeshData())
  {
    LOGI("Rewriting the gltf model as new mesh data was generated\n");
    bool writeDisplacementMicromapExt = !m_barys.empty();
    write(rewrittenModel, {}, writeDisplacementMicromapExt);
    outModel = &rewrittenModel;
  }
  else
  {
    // Always rewrite the gltf NV_micromap array
    rewriteBarys(*outModel);

    // Rewrite outModel->m_images to use image paths saved above.
    if(!originalImages)
    {
      rewriteImages(*outModel);
    }
  }

  return saveTinygltfModel(filename, *outModel);
}

bool ToolScene::getHeightmap(int materialID, float& bias, float& scale, int& imageIndex) const
{
  if(materialID == -1)
    return false;

  nvh::KHR_materials_displacement displacement;
  if(!getMaterialsDisplacement(model().materials[materialID], displacement))
    return false;

  if(displacement.displacementGeometryTexture == -1)
    return false;

  imageIndex = model().textures[displacement.displacementGeometryTexture].source;
  bias       = displacement.displacementGeometryOffset;
  scale      = displacement.displacementGeometryFactor;
  return true;
}


void ToolScene::createViews()
{
  // Adding a reference to all primitives and a flat list
  // of nodes representing the primitive in world space.
  std::vector<std::vector<int>>             gltfMeshToToolMeshes;
  std::unordered_map<std::string, uint32_t> uniqueMeshPrim;
  bool                                      warnedDuplicates = false;
  for(auto& mesh : m_model->meshes)
  {
    gltfMeshToToolMeshes.emplace_back();
    ToolMesh::Meta meta(mesh);
    for(auto& primitive : mesh.primitives)
    {
      // Load and validate the glTF micromap reference
      ToolMesh::Relations relations(primitive);
      if(relations.bary != -1
         && (relations.bary < 0 || static_cast<size_t>(relations.bary) >= barys().size() || relations.group < 0
             || static_cast<size_t>(relations.group) >= barys()[relations.bary]->groups().size()))
      {
        LOGE("Error: NV_displacement_micromap contains invalid indices, micromap %i and groupIndex %i\n",
             relations.bary, relations.group);
        relations.bary  = ToolMesh::Relations().bary;
        relations.group = ToolMesh::Relations().group;
      }

      // A key is made to consolidate multiple gltf mesh primitives that
      // reference identical data.
      std::string key = makePrimitiveKey(primitive);

      // Create a ToolMesh if the the mesh primitive is unique
      auto it = uniqueMeshPrim.find(key);
      if(it == uniqueMeshPrim.end())
      {
        const bary::ContentView* baryView{};
        if(relations.bary != -1)
        {
          baryView = &barys()[relations.bary]->groups()[relations.group];
        }

        int meshIndex = static_cast<int>(m_meshes.size());
        it            = uniqueMeshPrim.insert({key, meshIndex}).first;
        m_meshes.push_back(std::make_unique<ToolMesh>(m_model.get(), relations, meta, &primitive, baryView));
      }
      else if(!warnedDuplicates)
      {
        // Leave a message, just in case this has unintended side effects
        LOGI("Note: Consolidated duplicate primitives in gltf mesh primitives");
        warnedDuplicates = true;
      }

      gltfMeshToToolMeshes.back().push_back(it->second);
    }
  }

  // Create instances from model nodes recursively
  for(int nodeID : m_model->scenes[0].nodes)
  {
    if(nodeID < 0 || static_cast<size_t>(nodeID) > m_model->nodes.size())
    {
      LOGW("Invalid scene node reference %i\n", nodeID);
      continue;
    }
    createInstances(*m_model, gltfMeshToToolMeshes, nodeID, nvmath::mat4f(1), m_instances);
  }

  // Build the first instance array to avoid linearly searching for every query
  for(size_t i = 0; i < m_instances.size(); ++i)
  {
    if(m_meshes[m_instances[i].mesh]->relations().firstInstance == -1)
    {
      m_meshes[m_instances[i].mesh]->relations().firstInstance = static_cast<int>(i);
    }
  }

  // Clear all references to the old bary files. This is not necessary but
  // avoids accidental stale data access.
  for(auto& mesh : m_model->meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      primitive.extensions.erase(NV_MICROMAP_TOOLING);
      primitive.extensions.erase(NV_DISPLACEMENT_MICROMAP);
    }
  }
}

void ToolScene::setMesh(size_t meshIndex, std::unique_ptr<ToolMesh> mesh)
{
  assert(meshIndex < m_meshes.size());
  [[maybe_unused]] ToolMesh::Relations& relations = mesh->relations();
  assert(relations.bary == -1 || (relations.bary >= 0 && static_cast<size_t>(relations.bary) < m_barys.size()));
  assert(relations.bary == -1
         || (relations.group >= 0 && static_cast<size_t>(relations.group) < m_barys[relations.bary]->groups().size()));
  m_meshes[meshIndex]      = std::move(mesh);
}

void ToolScene::setImage(size_t imageIndex, std::unique_ptr<ToolImage> image)
{
  assert(imageIndex < m_images.size());
  m_images[imageIndex]      = std::move(image);
}

size_t ToolScene::replaceBarys(std::unique_ptr<ToolBary> bary)
{
  clearBarys();

  // Add the given micromap as the sole entry in m_micromaps
  size_t baryIndex = m_barys.size();
  m_barys.push_back(std::move(bary));
  return baryIndex;
}

void ToolScene::linkBary(size_t baryIndex, size_t groupIndex, size_t meshIndex)
{
  // Update the references on the ToolMesh
  std::unique_ptr<ToolMesh>& mesh = m_meshes[meshIndex];
  mesh->relations().bary  = static_cast<int32_t>(baryIndex);
  mesh->relations().group = static_cast<int32_t>(groupIndex);

  // Remove heightmap displacement from this mesh's material. Note: this may
  // remove heightmap displacement from other meshes that share this material,
  // but we don't have a system to duplicate and consolidate materials yet.
  float bias;
  float scale;
  int   imageIndex;
  if(getHeightmap(mesh->relations().material, bias, scale, imageIndex))
  {
    tinygltf::Material& material = materials()[mesh->relations().material];
    auto                ext      = material.extensions.find(KHR_MATERIALS_DISPLACEMENT_NAME);
    if(ext != material.extensions.end())
    {
      material.extensions.erase(ext);
      LOGI(
          "Removing KHR_materials_displacement from material %i (%s) after adding a micromap. This will affect all "
          "meshes using it.\n",
          mesh->relations().material, material.name.c_str());
    }

    // Count the remaining material references to this heightmap
    int references = 0;
    for(auto& material : materials())
    {
      nvh::KHR_materials_displacement displacement;
      if(getMaterialsDisplacement(material, displacement))
      {
        if(textures()[displacement.displacementGeometryTexture].source == imageIndex)
        {
          ++references;
        }
      }
    }

    // If this is the last use of this heightmap, delete the image and re-index
    // all the glTF textures.
    if(references == 0)
    {
      LOGI(
          "Removing image %s as there are no more heightmap references. It is possible this was referenced by other "
          "material textures.\n",
          m_images[imageIndex]->relativePath().string().c_str());
      for(auto& texture : textures())
      {
        if(texture.source == imageIndex)
        {
          texture.source = -1;
          // tinygltf writes "null" if there are no fields and won't read the
          // result. Add a name to prevent this.
          if(texture.name.empty())
          {
            std::string imageName = images()[imageIndex]->relativePath().string();
            texture.name          = imageName.empty() ? "heightmap removed" : "removed " + imageName;
          }
        }
        else if(texture.source > imageIndex)
        {
          --texture.source;
        }
      }
      m_images.erase(m_images.begin() + imageIndex);
    }
  }
}

void ToolScene::clearBarys()
{
  // Clear all references to the old bary files.
  for(auto& mesh : meshes())
  {
    mesh->relations().bary  = -1;
    mesh->relations().group = 0;
  }
  m_barys.clear();
}

uint32_t ToolScene::createImage()
{
  auto index = static_cast<uint32_t>(m_images.size());
  // Place an invalid image as a marker
  m_images.push_back(micromesh_tool::ToolImage::createInvalid());
  return index;
}

void ToolScene::appendAuxImage(std::unique_ptr<ToolImage> image)
{
  m_auxImages.push_back(std::move(image));
}

bool ToolScene::loadBarys(const fs::path& basePath)
{
  assert(m_barys.empty());
  if(!m_barys.empty())
    return false;

  size_t micromapCount = 0;
  (void)getGLTFMicromapCount(model(), micromapCount);

  m_barys.resize(micromapCount);
  for(size_t i = 0; i < micromapCount; ++i)
  {
    NV_micromap micromapExt;
    if(!getGLTFMicromap(model(), static_cast<uint32_t>(i), micromapExt))
    {
      LOGE("Error: Failed to get micromap at index %zu\n", i);
      m_barys.clear();
      return false;
    }

    fs::path baryFilename = dlib::urldecode(micromapExt.uri);
    m_barys[i]            = ToolBary::create(basePath, baryFilename);
    if(!m_barys[i])
    {
      m_barys.clear();
      return false;
    }
  }

  // Clear all the micromap references from the tinygltf Model, now that we have
  // transferred them to m_barys. New gltf extensions will be written on save()
  // anyway, but this avoids merging old references from m_model when saving and
  // accidental stale access via model() by micromesh tools beforehand.
  m_model->extensions.erase(NV_MICROMAPS);

  return true;
}

bool ToolScene::loadImages(const fs::path& basePath)
{
  assert(m_images.empty());
  if(!m_images.empty())
    return false;

  // We don't currently read all the KHR_texture_transform properties that
  // can exist, so print a warning if the hi or lo file use this extension:
  if(std::find(m_model->extensionsUsed.begin(), m_model->extensionsUsed.end(), KHR_TEXTURE_TRANSFORM_EXTENSION_NAME)
     != m_model->extensionsUsed.end())
  {
    LOGW(
        "Warning: The hi-res model used the KHR_texture_transform glTF extension, which this code does not support. "
        "Outputs may be incorrect.\n");
  }

  for(auto& gltfImage : m_model->images)
  {
    m_images.emplace_back();
    if(!gltfImage.uri.empty())
    {
      fs::path basePathAbs  = basePath.empty() ? fs::current_path() : fs::absolute(basePath);
      fs::path relativePath = dlib::urldecode(gltfImage.uri);
      m_images.back()       = ToolImage::create(basePathAbs, relativePath);
      if(!m_images.back())
      {
        LOGE("Failed to create ToolImage for scene\n");
        return false;
      }
    }
    else if(!gltfImage.image.empty())
    {
      ToolImage::Info info{};
      info.width             = gltfImage.width;
      info.height            = gltfImage.height;
      info.components        = gltfImage.component;
      info.componentBitDepth = gltfImage.bits;
      // Ignore image load failures. An error will already be printed.
      std::string name = (gltfImage.name.empty() ? "embedded_image_" + std::to_string(m_images.size() - 1) : gltfImage.name);
      m_images.back()  = ToolImage::create(info, name + ".png");
      if(!m_images.back())
      {
        LOGE("Failed to create ToolImage for scene\n");
        return false;
      }
      if(m_images.back())
      {
        memcpy(m_images.back()->raw(), gltfImage.image.data(), gltfImage.image.size());
      }
    }
    else if(gltfImage.bufferView != -1)
    {
      const tinygltf::BufferView& bufferView = m_model->bufferViews[gltfImage.bufferView];
      const tinygltf::Buffer&     buffer     = m_model->buffers[bufferView.buffer];
      const uint8_t*              srcData    = buffer.data.data() + bufferView.byteOffset;
      if(bufferView.byteStride > 1)
      {
        LOGE(
            "Error: Image referenced bufferView %i, which had a byteStride of %zu; this byte stride is not "
            "supported.\n",
            gltfImage.bufferView, bufferView.byteStride);
        continue;
      }

      // Decompress the image immediately as ToolImage does not support lazy
      // decompression.
      ToolImage::Info info{};
      if(!imageio::infoFromMemory(srcData, bufferView.byteLength, &info.width, &info.height, &info.components))
      {
        LOGE("Error: failed to read embedded image %s\n", gltfImage.name.c_str());
        continue;
      }

      // TODO: this is totally untested
      info.componentBitDepth = 8;
      if(gltfImage.bits)
        info.componentBitDepth = gltfImage.bits;

      LOGI("Decompressing embedded image %s\n", gltfImage.name.c_str());
      imageio::ImageIOData data = imageio::loadGeneralFromMemory(srcData, bufferView.byteLength, &info.width, &info.height,
                                                                 &info.components, info.components, info.componentBitDepth);

      // Create the image from the raw data. ToolImage takes ownership of data
      // and will free it. Ignore image load failures.
      m_images.back() = ToolImage::create(info, gltfImage.name, data);
      if(!m_images.back())
      {
        LOGE("Error: failed to decompress embedded image %s\n", gltfImage.name.c_str());
        return false;
      }
    }
    else
    {
      LOGW("Warning: invalid gltf - empty gltf image\n");
      m_images.back() = ToolImage::createInvalid();
    }
  }
  return true;
}

ToolSceneDimensions::ToolSceneDimensions(const ToolScene& scene)
{
  nvh::Bbox scnBbox;

  // Find the union of all object space bounding boxes transformed to world space
  for(auto& instance : scene.instances())
  {
    auto&           mesh        = scene.meshes()[instance.mesh];
    bool            foundMinMax = false;
    nvmath::vec3f   posMin;
    nvmath::vec3f   posMax;

    // If the mesh is still consistent with the gltf, it is safe to read the
    // bounds from the file
    if(mesh->isOriginalData() && mesh->gltfPrimitive())
    {
      auto primitive     = mesh->gltfPrimitive();
      auto posAccessorIt = primitive->attributes.find("POSITION");
      if(posAccessorIt == primitive->attributes.end())
      {
        LOGW("Warning: gltf primitive has no POSITION attribute\n");
      }
      else
      {
        const tinygltf::Accessor& posAccessor = scene.model().accessors[posAccessorIt->second];
        if(posAccessor.minValues.empty() || posAccessor.maxValues.empty())
        {
          LOGW("Warning: gltf primitive POSITION attribute is missing minValues and maxValues\n");
        }
        else
        {
          posMin      = nvmath::vec3f(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
          posMax      = nvmath::vec3f(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);
          foundMinMax = true;
        }
      }
    }

    // Fall back to re-computing the mesh's bounding box. Redundantly computed
    // per instance, but this is likely uncommon.
    if(!foundMinMax)
    {
      auto positions = mesh->view().vertexPositions;
      auto minMax    = minmax_elements_op(positions.begin(), positions.end(),
                                          static_cast<BinaryOp<nvmath::vec3f>*>(&nv_min2<nvmath::vec3f>),
                                          static_cast<BinaryOp<nvmath::vec3f>*>(&nv_max2<nvmath::vec3f>));
      posMin         = minMax.first;
      posMax         = minMax.second;
    }

    if(!nvmath::isreal(posMin) || !nvmath::isreal(posMax))
    {
      LOGW("Warning: mesh %i has an invalid bounding box\n", instance.mesh);
      continue;
    }

    nvh::Bbox bbox(posMin, posMax);
    bbox = bbox.transform(instance.worldMatrix);
    scnBbox.insert(bbox);
  }

  if(scnBbox.isEmpty() || !scnBbox.isVolume())
  {
    LOGW("Warning: glTF scene bounding box invalid. Setting to: [-1,-1,-1], [1,1,1]\n");
    scnBbox.insert({-1.0f, -1.0f, -1.0f});
    scnBbox.insert({1.0f, 1.0f, 1.0f});
  }

  min    = scnBbox.min();
  max    = scnBbox.max();
  size   = scnBbox.extents();
  center = scnBbox.center();
  radius = scnBbox.radius();
}

ToolSceneStats::ToolSceneStats(const ToolScene& scene)
{
  for(auto& mesh : scene.meshes())
  {
    triangles += mesh->view().triangleCount();
    vertices += mesh->view().vertexCount();

    if(mesh->relations().bary != -1)
    {
      micromaps = true;
    }
    if(mesh->relations().material != -1)
    {
      auto& material = scene.materials()[mesh->relations().material];
      if(material.normalTexture.index != -1)
      {
        normalmaps = true;
      }
      if(mesh->view().vertexTangents.empty() && material.normalTexture.index != -1)
      {
        normalmapsMissingTangents = true;
      }
    }
  }

  for(const auto& material : scene.model().materials)
  {
    nvh::KHR_materials_displacement displacement;
    if(getMaterialsDisplacement(material, displacement))
    {
      heightmaps = true;
    }
  }

  for(const auto& bary : scene.barys())
  {
    for(const auto& group : bary->groups())
    {
      maxBarySubdivLevel = std::max(maxBarySubdivLevel, group.basic.groups[0].maxSubdivLevel);
    }
  }
}

std::string ToolSceneStats::str()
{
  // This string should contain a few easily identifiable attributes that
  // frequently change when performing operations on the scene.
  std::stringstream desc;
  desc << triangles << " triangles, " << vertices << " vertices";
  if(images)
    desc << ", " << images << " images";
  if(micromaps)
    desc << ", micromaps";
  if(heightmaps)
    desc << ", heightmaps";
  return desc.str();
}

// Print char, uint8_t etc. as a number and not characters
namespace numerical_chars {
inline std::ostream& operator<<(std::ostream& os, char c)
{
  return os << +c;
}

inline std::ostream& operator<<(std::ostream& os, signed char c)
{
  return os << +c;
}

inline std::ostream& operator<<(std::ostream& os, unsigned char c)
{
  return os << +c;
}
}  // namespace numerical_chars

template <class T>
std::ostream& operator<<(std::ostream& os, const nvmath::vector2<T>& v)
{
  return os << "{" << v.x << ", " << v.y << "}";
}

template <class T>
std::ostream& operator<<(std::ostream& os, const nvmath::vector3<T>& v)
{
  return os << "{" << v.x << ", " << v.y << ", " << v.z << "}";
}

template <class T>
std::ostream& operator<<(std::ostream& os, const nvmath::vector4<T>& v)
{
  return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}";
}

template <class Array>
void writeArray(const Array& array, std::ostream& os)
{
  bool first = true;
  for(auto& value : array)
  {
    if(first)
    {
      first = false;
    }
    else
    {
      os << ", ";
    }

    using namespace numerical_chars;
    os << value;
  }
}

template <class Array>
void writeArrayPreview(const Array& array, std::ostream& os)
{
  const size_t previewBeginEnd = 3;
  os << "{";
  if(array.size() < previewBeginEnd * 2)
  {
    writeArray(array, os);
  }
  else
  {
    writeArray(meshops::ArrayView(array).slice(0, previewBeginEnd), os);
    os << " ... ";
    writeArray(meshops::ArrayView(array).slice(array.size() - previewBeginEnd, previewBeginEnd), os);
  }
  os << "}[" << array.size() << "]";
}

template <class Array>
void writeMinMax(const Array& array, std::ostream& os)
{
  if(array.empty())
  {
    os << "[empty]";
  }
  else
  {
    using namespace numerical_chars;
    using T     = typename Array::value_type;
    auto minMax = minmax_elements_op(std::begin(array), std::end(array), static_cast<BinaryOp<T>*>(&nv_min2<T>),
                                     static_cast<BinaryOp<T>*>(&nv_max2<T>));
    os << "[" << minMax.first << ", " << minMax.second << "]";
  }
}

void sceneWriteDebug(const ToolScene& scene, std::ostream& os)
{
  os << "Scene (" << ToolSceneStats(scene).str() << "):" << std::endl;
  os << "Meshes: " << scene.meshes().size() << std::endl;
  for(size_t i = 0; i < scene.meshes().size(); ++i)
  {
    const auto& mesh = scene.meshes()[i];
    const auto& view = mesh->view();
    os << "  Mesh " << i << std::endl;
    // clang-format off
    os << "    triangleVertices:          "; writeArrayPreview(view.triangleVertices, os);          os << ", range "; writeMinMax(view.triangleVertices, os);          os << std::endl;
    os << "    vertexPositions:           "; writeArrayPreview(view.vertexPositions, os);           os << ", range "; writeMinMax(view.vertexPositions, os);           os << std::endl;
    os << "    vertexNormals:             "; writeArrayPreview(view.vertexNormals, os);             os << ", range "; writeMinMax(view.vertexNormals, os);             os << std::endl;
    os << "    vertexTexcoords0:          "; writeArrayPreview(view.vertexTexcoords0, os);          os << ", range "; writeMinMax(view.vertexTexcoords0, os);          os << std::endl;
    os << "    vertexTangents:            "; writeArrayPreview(view.vertexTangents, os);            os << ", range "; writeMinMax(view.vertexTangents, os);            os << std::endl;
    os << "    vertexDirections:          "; writeArrayPreview(view.vertexDirections, os);          os << ", range "; writeMinMax(view.vertexDirections, os);          os << std::endl;
    os << "    vertexDirectionBounds:     "; writeArrayPreview(view.vertexDirectionBounds, os);     os << ", range "; writeMinMax(view.vertexDirectionBounds, os);     os << std::endl;
    os << "    vertexImportance:          "; writeArrayPreview(view.vertexImportance, os);          os << ", range "; writeMinMax(view.vertexImportance, os);          os << std::endl;
    os << "    triangleSubdivisionLevels: "; writeArrayPreview(view.triangleSubdivisionLevels, os); os << ", range "; writeMinMax(view.triangleSubdivisionLevels, os); os << std::endl;
    os << "    trianglePrimitiveFlags:    "; writeArrayPreview(view.trianglePrimitiveFlags, os);    os << ", range "; writeMinMax(view.trianglePrimitiveFlags, os);    os << std::endl;
    // clang-format on
    if(mesh->relations().bary != -1)
    {
      const auto& bary  = scene.barys()[mesh->relations().bary];
      const auto& group = bary->groups()[mesh->relations().group];
      meshops::ArrayView baryValues(group.basic.values, group.basic.valuesInfo->valueCount * group.basic.valuesInfo->valueByteSize);
      os << "    Bary " << mesh->relations().bary << " (group " << mesh->relations().group << ")" << std::endl;
      os << "      values:";
      writeArrayPreview(baryValues, os);
      os << " (" << baryFormatGetName(group.basic.valuesInfo->valueFormat) << ")" << std::endl;
    }
  }
  os << "Images: " << scene.images().size() << std::endl;
  for(size_t i = 0; i < scene.images().size(); ++i)
  {
    const auto& image = scene.images()[i];
    const auto& path  = image->relativePath();
    const auto& info  = image->info();
    os << "  Image " << i << " " << path << " " << info.width << "x" << info.height;
    os << ", " << info.components << "x" << info.componentBitDepth << "bit" << std::endl;
  }
}

}  // namespace micromesh_tool
