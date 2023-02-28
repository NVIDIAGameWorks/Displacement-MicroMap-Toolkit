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

// Unique couple of node/mesh/primitive and material
struct PrimitiveNode
{
  nvmath::mat4f              matrix;
  const tinygltf::Node&      node;
  const tinygltf::Mesh&      mesh;
  const tinygltf::Primitive& primitive;
  int                        nodeIndex;
};

//--------------------------------------------------------------------------------------------------
// Collecting a linear list of all primitives, navigating down the scene tree list
//
static void collectPrimitiveNodes(int nodeID, nvmath::mat4f parentMatrix, std::vector<PrimitiveNode>& primNodes, const tinygltf::Model* model)
{
  const tinygltf::Node& node   = model->nodes[nodeID];
  nvmath::mat4f         matrix = parentMatrix * nvh::getLocalMatrix(node);

  // Check if the node have a valid mesh, sometimes nodes only have a matrix and children
  if(node.mesh >= 0)
  {
    const tinygltf::Mesh& mesh = model->meshes[node.mesh];
    for(const tinygltf::Primitive& primitive : mesh.primitives)
    {
      PrimitiveNode primNode{matrix, node, mesh, primitive, nodeID};
      primNodes.push_back(primNode);
    }
  }

  // Recursion for all children of the node
  for(int child : node.children)
  {
    collectPrimitiveNodes(child, matrix, primNodes, model);
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

micromesh::Result ToolScene::create(const fs::path& filename)
{
  assert(this);
  auto model = std::make_unique<tinygltf::Model>();
  if(!micromesh_tool::loadTinygltfModel(filename, *model))
  {
    LOGE("Error: Failed to load '%s'\n", filename.string().c_str());
    return micromesh::Result::eFailure;
  }

  if(!updateNVBarycentricDisplacementToNVDisplacementMicromap(*model))
  {
    return micromesh::Result::eFailure;
  }

  // Wrap it with ToolScene to provide abstract mesh views and aux data storage
  auto basePath = fs::path(filename).parent_path();

  return create(std::move(model), basePath);
}

micromesh::Result ToolScene::create(std::unique_ptr<tinygltf::Model> model, const fs::path basePath)
{
  assert(this);
  assert(m_meshes.empty());  // call ToolBary::destroy()
  *this = ToolScene(std::move(model), basePath);
  if(meshes().size() == 0)
  {
    LOGE("Error: Creating a scene with no meshes\n");
    return micromesh::Result::eFailure;
  }
  return micromesh::Result::eSuccess;
}

micromesh::Result ToolScene::create(std::unique_ptr<tinygltf::Model>          model,
                                    std::vector<std::unique_ptr<ToolImage>>&& images,
                                    std::vector<std::unique_ptr<ToolBary>>&&  barys)
{
  assert(this);
  *this = ToolScene(std::move(model), std::move(images), std::move(barys));
  if(meshes().size() == 0)
  {
    LOGE("Error: Creating a scene with no meshes\n");
    return micromesh::Result::eFailure;
  }
  return micromesh::Result::eSuccess;
}

micromesh::Result ToolScene::create(const std::unique_ptr<ToolScene>& source)
{
  assert(this);
  assert(m_meshes.empty());  // call ToolBary::destroy()

  // Copy all the data that ToolScene cannot represent. Do not copy buffers (or
  // buffer views or accessors) since the mesh data is likely stale. ToolMesh
  // contains the ground truth even though initially it has pointers into the
  // gltf buffers.
  // TODO: This is probably not safe to do blindly as there are possibly many
  // gltf features relying on data in discarded buffers. We should instead only
  // keep data we know we can write out safely.
  auto model                = std::make_unique<tinygltf::Model>();
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
  std::vector<std::unique_ptr<ToolMesh>> meshes;
  for(auto& mesh : source->meshes())
  {
    // Copy the mesh from the other scene, which may reference mixed tinygltf
    // data and new data, into the aux MeshData container of a new ToolMesh
    meshes.push_back(std::make_unique<ToolMesh>(*mesh));
  }

  // Copy all Micromap objects
  std::vector<std::unique_ptr<ToolBary>> barys;
  for(auto& bary : source->m_barys)
  {
    barys.push_back(std::make_unique<ToolBary>());
    if(barys.back()->create(*bary) != micromesh::Result::eSuccess)
    {
      LOGE("Error: failed to duplicate a ToolBary\n")
      return micromesh::Result::eFailure;
    }
  }

  // Copy all image objects
  std::vector<std::unique_ptr<ToolImage>> images;
  for(auto& image : source->m_images)
  {
    images.push_back(std::make_unique<ToolImage>());
    if(images.back()->create(*image) != micromesh::Result::eSuccess)
    {
      LOGE("Error: failed to duplicate ToolImage %s. Ignoring.\n", image->relativePath().string().c_str())
    }
  }

  // Copy all auxiliary image objects
  std::vector<std::unique_ptr<ToolImage>> auxImages;
  for(auto& image : source->m_auxImages)
  {
    auxImages.push_back(std::make_unique<ToolImage>());
    if(auxImages.back()->create(*image) != micromesh::Result::eSuccess)
    {
      LOGE("Error: failed to duplicate auxiliary ToolImage %s. Ignoring.\n", image->relativePath().string().c_str())
    }
  }

  // Clear and update the current object. This is done last in case there is an
  // error and the ToolScene can be left in a consistent state.
  *this           = {};
  m_model         = std::move(model);
  m_meshes        = std::move(meshes);
  m_barys         = std::move(barys);
  m_images        = std::move(images);
  m_auxImages     = std::move(auxImages);
  m_primInstances = source->m_primInstances;
  for(auto& mesh : m_meshes)
  {
    m_constMeshes.push_back(mesh.get());
  }
  for(auto& bary : m_barys)
  {
    m_constBarys.push_back(bary.get());
  }
  for(auto& image : m_images)
  {
    m_constImages.push_back(image.get());
  }
  return micromesh::Result::eSuccess;
}

void ToolScene::destroy()
{
  // Necessary only to catch failures to call ToolBary::destroy()
  for(auto& bary : m_barys)
  {
    bary->destroy();
  }
  m_barys.clear();
  m_constBarys.clear();
  m_primInstances.clear();
  // Signal that the scene has been "destroyed"
  m_meshes.clear();
  m_constMeshes.clear();
  m_model.reset();
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
  // Attempt to match the structure of gltf meshes and mesh.primitives in the
  // original gltf m_model when writing m_meshes. Rather than tracking indices
  // and mesh instances (gltf Nodes), this block re-creates the mapping from
  // gltf Nodes to ToolMesh instances that is done in ToolScene::createViews().
  std::vector<PrimitiveNode> primNodes;
  std::vector<PrimitiveNode> uniquePrimNodes;
  std::vector<int>           nodeToMeshIndices(m_model->nodes.size(), -1);
  std::vector<int>           nodeToInstanceIndices(m_model->nodes.size(), -1);
  {
    // Get the list of nodes, meshes and primitives that made the ToolMesh
    for(int nodeID : m_model->scenes[0].nodes)
    {
      collectPrimitiveNodes(nodeID, nvmath::mat4f(1), primNodes, m_model.get());
    }

    // Loop over those primitive nodes and add a tinygltf::Node only if the PrimitiveNode
    // is unique, exactly the same way the ToolMesh array was created
    std::unordered_map<std::string, uint32_t> processedPrim;
    for(uint32_t i = 0; i < primNodes.size(); i++)
    {
      // A key is made to consolidate multiple mesh primitives that reference identical data.
      std::string key = makePrimitiveKey(primNodes[i].primitive);

      // Keep only the unique primitive/mesh nodes.
      auto it = processedPrim.find(key);
      if(it == processedPrim.end())
      {
        // Expected ToolMesh index
        int meshIndex = static_cast<int>(uniquePrimNodes.size());
        it            = processedPrim.insert({key, meshIndex}).first;
        uniquePrimNodes.push_back(primNodes[i]);
      }

      nodeToMeshIndices[primNodes[i].nodeIndex] = it->second;
      nodeToInstanceIndices[primNodes[i].nodeIndex] = i;
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

  // Write all meshes to the output model
  assert(output.meshes.empty());  // output must be a new object
  for(size_t meshID = 0; meshID < m_meshes.size(); meshID++)
  {
    const auto& mesh = m_meshes[meshID];

    appendToTinygltfModel(output, static_cast<const meshops::MeshView>(mesh->view()), writeDisplacementMicromapExt);

    ToolMesh::Relations& relations = mesh->relations();

    // Verify bary and group indices.
    assert(relations.bary == -1 || (relations.bary >= 0 && static_cast<size_t>(relations.bary) < m_barys.size()));
    assert(relations.bary == -1
           || (relations.group >= 0 && static_cast<size_t>(relations.group) < m_barys[relations.bary]->groups().size()));

    // Assumes appendToTinygltfModel creates a single mesh with a single
    // primitive.
    assert(output.meshes.size() == meshID + 1);
    assert(output.meshes.back().primitives.size() == 1);
    tinygltf::Mesh&      outputMesh      = output.meshes.back();
    tinygltf::Primitive& outputPrimitive = output.meshes.back().primitives.back();

    // Copy back the material ID. Between loading and saving the definitive
    // version lives on ToolMesh::Relations.
    outputPrimitive.material = relations.material;

    // Copy mesh mesh.primitive metadata from original gltf. Other gltf objects
    // are copied by copyTinygltfModelExtra().
    assert(uniquePrimNodes.size() == m_meshes.size());  // uniquePrimNodes should be populated identically to m_meshes.
    if(meshID < uniquePrimNodes.size())
    {
      PrimitiveNode& prim_node = uniquePrimNodes[meshID];
      outputMesh.name          = prim_node.mesh.name;
      outputMesh.extras        = prim_node.mesh.extras;
      outputMesh.weights       = prim_node.mesh.weights;
      outputPrimitive.extras   = prim_node.primitive.extras;
      outputPrimitive.mode     = prim_node.primitive.mode;
      copyExtensions(prim_node.mesh.extensions, outputMesh.extensions);
      copyExtensions(prim_node.primitive.extensions, outputPrimitive.extensions);
    }

    // Add micromesh references as each mesh+primitive as they're written. Note
    // the output primitive is different to the ToolMesh primitive, which
    // belongs to the original ToolScene input file.
    NV_displacement_micromap displacementMicromap;
    getPrimitiveDisplacementMicromap(outputPrimitive, displacementMicromap);
    displacementMicromap.micromap   = relations.bary;
    displacementMicromap.groupIndex = relations.group;
    displacementMicromap.mapOffset  = relations.mapOffset;
    setPrimitiveDisplacementMicromap(outputPrimitive, displacementMicromap);
  }

  // Copy non-mesh data. This clobbers anything previously on the output model
  // except mesh data.
  copyTinygltfModelExtra(model(), output, extensionFilter);

  // Re-writing meshes and mesh primitives in the order of uniquePrimNodes can
  // change the node indices, so they need rewriting too.
  assert(nodeToMeshIndices.size() == output.nodes.size());
  assert(nodeToInstanceIndices.size() == output.nodes.size());
  assert(primNodes.size() == m_primInstances.size());
  for(size_t nodeID = 0; nodeID < output.nodes.size(); nodeID++)
  {
    if(output.nodes[nodeID].mesh != nodeToMeshIndices[nodeID])
    {
      LOGI("Reindexing node %zu's mesh from %i to %i\n", nodeID, output.nodes[nodeID].mesh, nodeToMeshIndices[nodeID]);
    }
    output.nodes[nodeID].mesh = nodeToMeshIndices[nodeID];

    if(nodeToInstanceIndices[nodeID] != -1)
    {
      output.nodes[nodeID].name = m_primInstances[nodeToInstanceIndices[nodeID]].name;
    }
  }

  // In the case there is a disparity between the number of ToolMesh and the original number of meshes
  // The nodes of the scene will be adjusted; flatten and refer directly to the mesh
  if(model().meshes.size() != output.meshes.size())
  {
    output.nodes.clear();
    for(uint32_t i = 0; i < uniquePrimNodes.size(); i++)
    {
      auto& node = output.nodes.emplace_back(uniquePrimNodes[i].node);
      node.mesh  = i;
    }
  }

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
  setExtensionUsed(output.extensionsUsed, "KHR_materials_displacement", hasHeightmapDisplacement);
  setExtensionUsed(output.extensionsUsed, NV_DISPLACEMENT_MICROMAP, hasNVDispacementMicromap);
  setExtensionUsed(output.extensionsUsed, NV_MICROMAP_TOOLING, hasNVMicromapTooling);
  setExtensionUsed(output.extensionsUsed, NV_MICROMAPS, hasNVDispacementMicromap || hasNVMicromapTooling);
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
  if(!isOriginalData())
  {
    LOGI("Rewriting the gltf model as new mesh data was generated\n");
    bool writeDisplacementMicromapExt = !m_barys.empty();
    write(rewrittenModel, {}, writeDisplacementMicromapExt);
    outModel = &rewrittenModel;
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
  // Get the list of all primitives
  std::vector<PrimitiveNode> primNodes;
  for(int nodeID : m_model->scenes[0].nodes)
  {
    collectPrimitiveNodes(nodeID, nvmath::mat4f(1), primNodes, m_model.get());
  }

  // Adding a reference to all primitives and a flat list
  // of nodes representing the primitive in world space.
  std::unordered_map<std::string, uint32_t> processedPrim;
  bool                                      warnedDuplicates = false;
  for(uint32_t i = 0; i < primNodes.size(); i++)
  {
    PrimitiveNode& prim_node = primNodes[i];

    NV_displacement_micromap displacement;
    bool hasMicromap = getPrimitiveDisplacementMicromap(prim_node.primitive, displacement) && displacement.micromap != -1;
    int32_t baryIndex{-1};
    int32_t baryGroup{0};
    if(hasMicromap)
    {
      if(displacement.micromap >= 0 && static_cast<size_t>(displacement.micromap) < barys().size() && displacement.groupIndex >= 0
         && static_cast<size_t>(displacement.groupIndex) < barys()[displacement.micromap]->groups().size())
      {
        baryIndex = displacement.micromap;
        baryGroup = displacement.groupIndex;
      }
      else
      {
        LOGE("Error: NV_displacement_micromap contains invalid indices, micromap %i and groupIndex %i\n",
             displacement.micromap, displacement.groupIndex);
      }
    }

    // A key is made to consolidate multiple mesh primitives that reference identical data.
    std::string key = makePrimitiveKey(prim_node.primitive);

    // Adding a reference to the primitive only if it wasn't processed
    if(processedPrim.find(key) == processedPrim.end())
    {
      const bary::ContentView* baryView{};
      if(baryIndex != -1)
      {
        baryView = &barys()[baryIndex]->groups()[baryGroup];
      }

      // Keep info of the primitive reference for mesh instance
      processedPrim[key] = static_cast<uint32_t>(m_meshes.size());
      m_meshes.push_back(std::make_unique<ToolMesh>(m_model.get(), (tinygltf::Mesh*)&prim_node.mesh,
                                                    (tinygltf::Primitive*)&prim_node.primitive, prim_node.matrix, baryView));
      m_constMeshes.push_back(m_meshes.back().get());
    }
    else if(!warnedDuplicates)
    {
      // Leave a message, just in case this has unintended side effects
      LOGI("Note: Consolidated duplicate primitives in gltf mesh primitives");
      warnedDuplicates = true;
    }

    // The primitive was processed, and we have a different matrices and material referencing the primitives
    {
      PrimitiveInstance primInstance;
      primInstance.worldMatrix = prim_node.matrix;                 // Unique world matrix
      primInstance.primMeshRef = processedPrim[key];               // Same data
      primInstance.material    = primNodes[i].primitive.material;  // unique material
      primInstance.name        = primNodes[i].node.name;
      m_primInstances.push_back(primInstance);
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

//--------------------------------------------------------------------------------------------------
// Create a list of unique primitive in the hierarchy tree.
//
void ToolScene::setMesh(size_t meshIndex, std::unique_ptr<ToolMesh> mesh)
{
  assert(meshIndex < m_meshes.size());
  ToolMesh::Relations& relations = mesh->relations();
  assert(relations.bary == -1 || (relations.bary >= 0 && static_cast<size_t>(relations.bary) < m_barys.size()));
  assert(relations.bary == -1
         || (relations.group >= 0 && static_cast<size_t>(relations.group) < m_barys[relations.bary]->groups().size()));
  m_meshes[meshIndex]      = std::move(mesh);
  m_constMeshes[meshIndex] = m_meshes[meshIndex].get();
}

void ToolScene::setImage(size_t imageIndex, std::unique_ptr<ToolImage> image)
{
  assert(imageIndex < m_images.size());
  m_images[imageIndex]      = std::move(image);
  m_constImages[imageIndex] = m_images[imageIndex].get();
}

size_t ToolScene::replaceBarys(std::unique_ptr<ToolBary> bary)
{
  clearBarys();

  // Add the given micromap as the sole entry in m_micromaps
  size_t baryIndex = m_barys.size();
  m_barys.push_back(std::move(bary));
  m_constBarys.push_back(m_barys.back().get());
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
  if(mesh->relations().material != -1)
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

  for(auto& bary : m_barys)
  {
    bary->destroy();
  }
  m_barys.clear();
  m_constBarys.clear();
}

uint32_t ToolScene::createImage()
{
  auto index = static_cast<uint32_t>(m_images.size());
  m_images.push_back(std::make_unique<micromesh_tool::ToolImage>());
  m_constImages.push_back(m_images.back().get());
  return index;
}

ToolImage& ToolScene::createAuxImage()
{
  m_auxImages.push_back(std::make_unique<micromesh_tool::ToolImage>());
  return *m_auxImages.back();
}

bool ToolScene::loadBarys(const fs::path& basePath)
{
  assert(m_barys.empty());
  if(!m_barys.empty())
    return false;

  size_t micromapCount = 0;
  (void)getGLTFMicromapCount(model(), micromapCount);

  m_barys.resize(micromapCount);
  m_constBarys.resize(micromapCount);
  for(size_t i = 0; i < micromapCount; ++i)
  {
    NV_micromap micromapExt;
    if(!getGLTFMicromap(model(), static_cast<uint32_t>(i), micromapExt))
    {
      LOGE("Error: Failed to get micromap at index %zu\n", i);
      m_barys.clear();
      m_constBarys.clear();
      return false;
    }

    fs::path baryFilename = dlib::urldecode(micromapExt.uri);
    m_barys[i]            = std::make_unique<ToolBary>();
    if(m_barys[i]->create(basePath, baryFilename) != micromesh::Result::eSuccess)
    {
      m_barys.clear();
      m_constBarys.clear();
      return false;
    }
    m_constBarys[i] = m_barys[i].get();
  }

  // Clear all the gltf micromap references now that we have transferred them to
  // m_barys. They are rewritten on save() anyway, but this avoids accidental
  // stale data access.
  tinygltf::Value::Array* gltfMicromaps = getNVMicromapExtensionMutable(*m_model);
  if(gltfMicromaps)
  {
    gltfMicromaps->clear();
  }

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
    m_images.push_back(std::make_unique<micromesh_tool::ToolImage>());
    m_constImages.push_back(m_images.back().get());
    if(!gltfImage.uri.empty())
    {
      fs::path basePathAbs  = basePath.empty() ? fs::current_path() : fs::absolute(basePath);
      fs::path relativePath = dlib::urldecode(gltfImage.uri);
      // Ignore image load failures. An error will already be printed.
      (void)m_images.back()->create(basePathAbs, relativePath);
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
      if(m_images.back()->create(info, name + ".png") == micromesh::Result::eSuccess)
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
      if(m_images.back()->create(info, gltfImage.name, data) == micromesh::Result::eSuccess)
      {
        LOGE("Error: failed to decompress embedded image %s\n", gltfImage.name.c_str());
        continue;
      }
    }
    else
    {
      LOGW("Warning: invalid gltf - empty gltf image\n");
    }
  }
  return true;
}

ToolSceneDimensions::ToolSceneDimensions(const ToolScene& scene)
{
  nvh::Bbox scnBbox;

  // Find the union of all object space bounding boxes transformed to world space
  for(auto& instance : scene.getPrimitiveInstances())
  {
    const ToolMesh* mesh        = scene.meshes()[instance.primMeshRef];
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
