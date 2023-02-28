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

#include "include/tool_merge.hpp"

#include <limits>
#include <memory>
#include <set>
#include "gltf/micromesh_util.hpp"
#include "nvh/nvprint.hpp"
#include "tool_scene.hpp"

namespace tool_merge {

// Offsets a glTF index, using the tinyGLTF convention where -1 means the
// index doesn't exist.
int indexAdd(int firstIndex, size_t offset)
{
  if(firstIndex == -1)
    return firstIndex;
  // Check for overflow
  assert(static_cast<int>(offset) <= std::numeric_limits<int>::max() - firstIndex);
  return firstIndex + static_cast<int>(offset);
}

void indexAddTo(int& firstIndex, size_t offset)
{
  firstIndex = indexAdd(firstIndex, offset);
}

// Merges b into a. On failure, logs and returns false.
bool mergeIntoFirst(tinygltf::Model& a, const tinygltf::Model& b, const std::string bFilename)
{
  // Proceed up the glTF dependency graph.
  // Micromaps
  size_t micromapsOffset = 0;  // Set by later code if using micromaps
  {
    const tinygltf::Value::Array* bMicromaps = getNVMicromapExtension(b);
    if(bMicromaps != nullptr)
    {
      const size_t numBMicromaps = bMicromaps->size();
      assert(numBMicromaps <= std::numeric_limits<int>::max());
      for(int i = 0; i < static_cast<int>(numBMicromaps); i++)
      {
        NV_micromap micromap{};
        if(!getGLTFMicromap(b, i, micromap))
        {
          assert(!"This should never happen");
          LOGE("Error: Could not access micromap %i of %s.\n", i, bFilename.c_str());
          return false;
        }
        const int32_t newIdx = addTinygltfMicromap(a, micromap);
        if(newIdx < 0)
        {
          LOGE("Error: Could not add a new micromap to the scene.\n");
          return false;
        }
        if(i == 0)
        {
          micromapsOffset = static_cast<size_t>(newIdx);
        }
      }
    }
  }

  // Buffers
  const size_t buffersOffset = a.buffers.size();
  a.buffers.insert(a.buffers.end(), b.buffers.begin(), b.buffers.end());

  // Buffer views
  const size_t bufferViewsOffset = a.bufferViews.size();
  for(tinygltf::BufferView bufferView : b.bufferViews)
  {
    indexAddTo(bufferView.buffer, buffersOffset);
    a.bufferViews.push_back(bufferView);
  }

  // Accessors
  const size_t accessorsOffset = a.accessors.size();
  for(tinygltf::Accessor accessor : b.accessors)
  {
    indexAddTo(accessor.bufferView, bufferViewsOffset);
    indexAddTo(accessor.sparse.values.bufferView, bufferViewsOffset);
    a.accessors.push_back(accessor);
  }

  // Images
  const size_t imagesOffset = a.images.size();
  for(tinygltf::Image image : b.images)
  {
    indexAddTo(image.bufferView, bufferViewsOffset);
    a.images.push_back(image);
  }

  // Samplers
  const size_t samplersOffset = a.samplers.size();
  a.samplers.insert(a.samplers.end(), b.samplers.begin(), b.samplers.end());

  // Textures
  const size_t texturesOffset = a.textures.size();
  for(tinygltf::Texture texture : b.textures)
  {
    indexAddTo(texture.sampler, samplersOffset);
    indexAddTo(texture.source, imagesOffset);
    a.textures.push_back(texture);
  }

  // Materials
  const size_t materialsOffset = a.materials.size();
  for(tinygltf::Material material : b.materials)
  {
    indexAddTo(material.normalTexture.index, texturesOffset);
    indexAddTo(material.occlusionTexture.index, texturesOffset);
    indexAddTo(material.pbrMetallicRoughness.baseColorTexture.index, texturesOffset);
    indexAddTo(material.pbrMetallicRoughness.metallicRoughnessTexture.index, texturesOffset);
    // We don't currently support any material extensions.
    for(const auto& extension : material.extensions)
    {
      LOGE("Unknown material extension: %s\n", extension.first.c_str());
    }
    a.materials.push_back(material);
  }

  // Meshes
  const size_t meshOffset = a.meshes.size();
  for(tinygltf::Mesh mesh : b.meshes)
  {
    for(tinygltf::Primitive& prim : mesh.primitives)
    {
      for(auto& attribute : prim.attributes)
      {
        indexAddTo(attribute.second, accessorsOffset);
      }
      indexAddTo(prim.indices, accessorsOffset);
      indexAddTo(prim.material, materialsOffset);
      NV_displacement_micromap displacementExt{};
      if(getPrimitiveDisplacementMicromap(prim, displacementExt))
      {
        indexAddTo(displacementExt.directionBounds, accessorsOffset);
        indexAddTo(displacementExt.directions, accessorsOffset);
        indexAddTo(displacementExt.mapIndices, accessorsOffset);
        indexAddTo(displacementExt.micromap, micromapsOffset);
        indexAddTo(displacementExt.primitiveFlags, accessorsOffset);
        setPrimitiveDisplacementMicromap(prim, displacementExt);
      }
      NV_micromap_tooling toolingExt{};
      if(getPrimitiveMicromapTooling(prim, toolingExt))
      {
        indexAddTo(toolingExt.directionBounds, accessorsOffset);
        indexAddTo(toolingExt.directions, accessorsOffset);
        indexAddTo(toolingExt.primitiveFlags, accessorsOffset);
        indexAddTo(toolingExt.subdivisionLevels, accessorsOffset);
      }
      // Produce errors for any unknown extensions
      for(const auto& extension : prim.extensions)
      {
        if(extension.first != NV_DISPLACEMENT_MICROMAP && extension.first != NV_MICROMAP_TOOLING)
        {
          LOGE("Unknown primitive extension: %s\n", extension.first.c_str());
        }
      }
    }
    a.meshes.push_back(mesh);
  }

  // Lights
  const size_t lightsOffset = a.lights.size();
  a.lights.insert(a.lights.end(), b.lights.begin(), b.lights.end());

  // Cameras
  const size_t camerasOffset = a.cameras.size();
  a.cameras.insert(a.cameras.end(), b.cameras.begin(), b.cameras.end());

  // Skins and nodes
  const size_t skinsOffset = a.skins.size();
  const size_t nodesOffset = a.nodes.size();
  for(tinygltf::Skin skin : b.skins)
  {
    indexAddTo(skin.skeleton, nodesOffset);
    indexAddTo(skin.inverseBindMatrices, accessorsOffset);
    for(int& joint : skin.joints)
    {
      indexAddTo(joint, nodesOffset);
    }
    a.skins.push_back(skin);
  }
  for(tinygltf::Node node : b.nodes)
  {
    indexAddTo(node.camera, camerasOffset);
    for(int& child : node.children)
    {
      indexAddTo(child, nodesOffset);
    }
    indexAddTo(node.mesh, meshOffset);
    a.nodes.push_back(node);
  }

  // Animations?
  const size_t animationOffset = a.animations.size();
  for(tinygltf::Animation animation : b.animations)
  {
    for(tinygltf::AnimationChannel& channel : animation.channels)
    {
      indexAddTo(channel.target_node, nodesOffset);
    }
    for(tinygltf::AnimationSampler& animationSampler : animation.samplers)
    {
      indexAddTo(animationSampler.input, accessorsOffset);
      indexAddTo(animationSampler.output, accessorsOffset);
    }
    a.animations.push_back(animation);
  }

  // Scenes
  // If both only have one scene, merge the nodes in the scene rather than
  // the list of nodes.
  if(a.scenes.size() == 1 && b.scenes.size() == 1)
  {
    auto&       aSceneNodes = a.scenes[0].nodes;
    const auto& bSceneNodes = b.scenes[0].nodes;
    for(const int& n : bSceneNodes)
    {
      aSceneNodes.push_back(indexAdd(n, nodesOffset));
    }
  }
  else
  {
    a.scenes.insert(a.scenes.end(), b.scenes.begin(), b.scenes.end());
  }

  // extensionsUsed
  std::set<std::string> extensionsUsed(a.extensionsUsed.begin(), a.extensionsUsed.end());
  std::copy(b.extensionsUsed.begin(), b.extensionsUsed.end(),  //
            std::inserter(extensionsUsed, extensionsUsed.end()));
  a.extensionsUsed.clear();
  a.extensionsUsed.insert(a.extensionsUsed.end(), extensionsUsed.begin(), extensionsUsed.end());

  // extensionsRequired
  std::set<std::string> extensionsRequired(a.extensionsRequired.begin(), a.extensionsRequired.end());
  std::copy(b.extensionsRequired.begin(), b.extensionsRequired.end(),  //
            std::inserter(extensionsRequired, extensionsRequired.end()));
  a.extensionsRequired.clear();
  a.extensionsRequired.insert(a.extensionsRequired.end(), extensionsRequired.begin(), extensionsRequired.end());

  return true;
}

bool toolMerge(const ToolMergeArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene)
{
  if(args.inputs.size() == 0)
  {
    LOGW(
        "Warning: Only one input mesh was specified, so merging will only copy the input to the output (updating "
        "extensions).\n");
  }

  // Start with the first input, then merge the rest into it. This should be
  // linear-time. This copies everything out of the ToolScene, but toolMerge()
  // is very gltf-centric anyway. An alternative might be to append extra
  // ToolMesh objects to the existing ToolScene, but keeping the ToolScene's
  // internal gltf consistent would be difficult.
  auto result = std::make_unique<tinygltf::Model>();
  scene->write(*result);

  for(size_t i = 0; i < args.inputs.size(); i++)
  {
    tinygltf::Model model;
    if(!micromesh_tool::loadTinygltfModel(args.inputs[i], model))
    {
      LOGE("Error: Could not load %s\n", args.inputs[i].c_str());
      return false;
    }
    if(!mergeIntoFirst(*result, model, args.inputs[i]))
    {
      return false;
    }
  }

  // Pack the output back into a ToolScene.
  scene->destroy();
  scene = std::make_unique<micromesh_tool::ToolScene>();
  if(scene->create(std::move(result), "") != micromesh::Result::eSuccess)
  {
    LOGE("Error: Failed to create ToolScene from merged gltf\n");
    return false;
  }

  return true;
}

}  // namespace tool_merge