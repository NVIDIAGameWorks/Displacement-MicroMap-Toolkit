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

#include <tool_bake.hpp>
#include "tool_meshops_objects.hpp"
#include "bary/bary_types.h"
#include "tool_scene.hpp"
#include <meshops/meshops_mesh_view.h>
#include <microutils/microutils.hpp>
#include <meshops_internal/heightmap.hpp>
#include <exception>
#include <gltf.hpp>
#include <gltf/micromesh_util.hpp>
#include <cstdint>
#include <nvmath/nvmath_types.h>
#include <micromesh/micromesh_types.h>
#include <baker_manager.hpp>
#include <meshops/meshops_operations.h>
#include <meshops/meshops_vk.h>
#include <cstddef>
#include <filesystem>
#include <inputparser.hpp>
#include <json.hpp>
#include <memory>
#include <nvh/fileoperations.hpp>
#include <nvh/gltfscene.hpp>
#include <thread>
#include <meshops/bias_scale.hpp>

namespace fs = std::filesystem;
using namespace micromesh_tool;

namespace tool_bake {

// Baking can split highres geometry into batches to stay under the memory limit. This doesn't work when a quad needs to
// be expanded to subdiv 12 for example. As a workaround, quads are pre-tessellated on the highres mesh too, which gives
// batching finer control over how many triangles get generated. This value is the desired difference in subdivision
// levels between the geometry and the heightmap resolution. Further tessellation will still happen, up to
// HEIGHTMAP_SUBDIV_BIAS.
#define HEIGHTMAP_PRETESSELLATE_QUAD_SUBDIV_DIFF 6

void debugWriteDisplacedReferenceMesh(const meshops::MeshView&           meshView,
                                      const micromesh::Matrix_float_4x4* transform,
                                      uint32_t                           batchIndex,
                                      uint32_t                           batchTotal,
                                      void*                              userPtr)
{
  size_t meshIndex = *reinterpret_cast<size_t*>(userPtr);

  std::stringstream filename;
  filename << "highres_m" << meshIndex << "_b" << batchIndex << ".gltf";
  if((fs::status(filename.str()).permissions() & fs::perms::owner_write) == fs::perms::none)
  {
    LOGE("Error: Failed to write %s. No write permissions.\n", filename.str().c_str());
    return;
  }

  meshops::MeshSetView meshSetView;
  meshSetView.flat = meshView;
  meshSetView.slices.emplace_back(meshView.triangleCount(), meshView.vertexCount());
  tinygltf::Model model;
  appendToTinygltfModel(model, meshSetView);

  // Single material
  int materialID = static_cast<int>(model.materials.size());
  {
    tinygltf::Material material;
    material.pbrMetallicRoughness.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    material.doubleSided                          = true;
    model.materials.push_back(std::move(material));
  }

  tinygltf::Scene scene;
  for(size_t meshId = 0; meshId < model.meshes.size(); ++meshId)
  {
    // Update all primitves to use the one material
    for(auto& primitive : model.meshes[meshId].primitives)
      primitive.material = materialID;

    // Instantiate all meshes
    int nodeID = static_cast<int>(model.nodes.size());
    {
      tinygltf::Node node;
      node.mesh   = static_cast<int>(meshId);
      node.matrix = {&transform->columns[0].x, &transform->columns[0].x + 16};
      model.nodes.push_back(std::move(node));
    }
    scene.nodes.push_back(nodeID);
  }

  model.scenes.push_back(std::move(scene));
  model.asset.copyright = "NVIDIA Corporation";
  model.asset.generator = "uMesh glTF ";
  model.asset.version   = "2.0";  // glTF version 2.0

  tinygltf::TinyGLTF saver;
  saver.SetImageWriter(nullptr, nullptr);  // Don't modify images
  LOGI("Writing %s (debug mesh)\n", filename.str().c_str());
  if(!saver.WriteGltfSceneToFile(&model, filename.str(), false, false, true, false))
  {
    LOGE("Error: Failed to write %s\n", filename.str().c_str());
  }
}

static micromesh::Result writeDirectionBoundsMeshes(const std::unique_ptr<ToolScene>& scene)
{
  // Make two copies of the scene. One for the lower bounds and one for the
  // upper bounds.
  // TODO: a ToolScene::createView() that has meshes pointing to the old scene
  // would be more efficient, but there's not much point until ToolImage can be
  // shared
  ToolScene insideScene;
  insideScene.create(scene);
  ToolScene outsideScene;
  outsideScene.create(scene);
  for(size_t i = 0; i < scene->meshes().size(); ++i)
  {
    auto& mesh        = scene->meshes()[i];
    auto& insideMesh  = insideScene.meshes()[i];
    auto& outsideMesh = outsideScene.meshes()[i];

    for(size_t i = 0; i < mesh->view().vertexPositions.size(); ++i)
    {
      BiasScalef biasScale(mesh->view().vertexDirectionBounds[i]);

      // Move the vertex positions of the "lower" mesh to the lower shell bound
      // and negate the normal.
      insideMesh->view().vertexPositions[i] += insideMesh->view().vertexDirections[i] * biasScale.unit_min();
      insideMesh->view().vertexNormals[i] *= -1.0F;

      // Move the vertex positions of the "outside" mesh to the upper sell
      // bound.
      outsideMesh->view().vertexPositions[i] += outsideMesh->view().vertexDirections[i] * biasScale.unit_max();
    }

    // Remove directions and bounds, since they have just been baked into the positions
    insideMesh->view().resize(meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit
                                  | meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit,
                              0, 0);
    outsideMesh->view().resize(meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit
                                   | meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit,
                               0, 0);
  }

  for(auto& mesh : insideScene.meshes())
    mesh->meta().name += "BoundsLower";
  for(auto& instance : insideScene.instances())
    instance.name += "BoundsLower";
  for(auto& mesh : outsideScene.meshes())
    mesh->meta().name += "BoundsUpper";
  for(auto& instance : outsideScene.instances())
    instance.name += "BoundsUpper";

  // Write the lower bounds mesh, including displacement lines
  tinygltf::Model insideGltfModel;
  insideScene.write(insideGltfModel);
  microutils::ScopedOpContextMsg context(std::thread::hardware_concurrency());
  for(size_t i = 0; i < scene->meshes().size(); ++i)
  {
    auto& mesh = scene->meshes()[i];
    if(mesh->relations().bary == -1)
    {
      continue;
    }

    const bary::BasicView& basic     = scene->barys()[mesh->relations().bary]->groups()[mesh->relations().group].basic;
    auto&                  baryGroup = basic.groups[0];
    meshops::ArrayView     distances(reinterpret_cast<const float*>(basic.values), basic.valuesInfo->valueCount);

    if(basic.valuesInfo->valueFormat != bary::Format::eR32_sfloat)
    {
      LOGE(
          "Warning: Not writing displacement lines in intermediate mesh %zu as values are not float 32. Bake with "
          "'--compressed false'.\n",
          i);
      continue;
    }

    std::vector<uint32_t>      indices;
    std::vector<nvmath::vec3f> positions;
    micromesh::Result          result =
        generateDisplacementLines(context, mesh->view(), basic, baryGroup, indices, positions, distances.data());
    assert(result == micromesh::Result::eSuccess);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Failed to generate displacement lines for the direction bounds mesh\n");
      return result;
    }

    addTinygltfModelLinesMesh(insideGltfModel, indices, positions, mesh->meta().name + "Displacements",
                              mesh->firstInstanceTransform());
  }
  if(!saveTinygltfModel("lowres_inside_bounds.gltf", insideGltfModel))
  {
    LOGE("Error: failed to write intermediate mesh lowres_inside_bounds.gltf\n");
    return micromesh::Result::eFailure;
  }

  // Write the upper bounds mesh. Don't use save() to avoid writing textures and
  // the bary file.
  tinygltf::Model outsideGltfModel;
  outsideScene.write(outsideGltfModel);
  if(!saveTinygltfModel("lowres_outside_bounds.gltf", outsideGltfModel))
  {
    LOGE("Error: failed to write intermediate mesh lowres_outside_bounds.gltf\n");
    return micromesh::Result::eFailure;
  }

  insideScene.destroy();
  outsideScene.destroy();
  return micromesh::Result::eSuccess;
}

void toolBakeAddRequirements(meshops::ContextConfig& contextConfig)
{
  contextConfig.requiresDeviceContext = true;
}

bool toolBake(micromesh_tool::ToolContext& context, const ToolBakeArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& base)
{
  // Load the reference scene, if there is one. Otherwise, reuse the base mesh
  fs::path                     highBasePath;
  std::unique_ptr<micromesh_tool::ToolScene> reference;
  if(!args.highFilename.empty())
  {
    auto referenceGltfModel = std::make_unique<tinygltf::Model>();
    if(!micromesh_tool::loadTinygltfModel(args.highFilename, *referenceGltfModel))
    {
      LOGE("Error: Failed to load '%s'\n", args.highFilename.c_str());
      return false;
    }

    reference = std::make_unique<ToolScene>();
    if(reference->create(std::move(referenceGltfModel), fs::path(args.highFilename).parent_path()) != micromesh::Result::eSuccess)
    {
      return false;
    }
    highBasePath = fs::path(args.highFilename).parent_path();
  }
  else
  {
    // TODO: Don't copy. Unfortunately we can't just point to the same scene,
    // because we need to generate different subdiv levels for the reference
    // mesh. We should instead be able to create a temporary "View" of the base
    // scene's model but have our own aux data for the reference scene.
    reference = std::make_unique<ToolScene>();
    if(reference->create(base) != micromesh::Result::eSuccess)
    {
      return false;
    }

    if(!micromesh_tool::ToolSceneStats(*reference).heightmaps)
    {
      LOGW(
          "Warning: Baking without a reference mesh. Using the base mesh, but it has no heightmaps. Displacements will "
          "be flat.\n");
    }
  }

  ToolBakeArgs bakeRefArgs(args);
  bakeRefArgs.highFilename.clear();

  bool result = toolBake(context, bakeRefArgs, *reference, base);
  reference->destroy();

  return result;
}

bool toolBake(micromesh_tool::ToolContext&                context,
              const ToolBakeArgs&                         args,
              const micromesh_tool::ToolScene&            reference,
              std::unique_ptr<micromesh_tool::ToolScene>& base)
{
  if(!args.highFilename.empty())
  {
    LOGE("Error: toolBake(..., reference, ...) may not be called with a highFilename\n");
    assert(false);
    return false;
  }

  meshops::ContextVK* contextVK = context.meshopsContextVK();
  if(!contextVK)
  {
    LOGE("Error: toolBake() requires a vulkan context\n");
    return false;
  }

  nvvk::Context* nvvkContext = contextVK->context;

  if(base->meshes().size() == 0)
  {
    LOGE("Error: baker input scene has no meshes\n");
    return false;
  }

  std::vector<tool_bake::ResampleMeshInstructions> resampleInstructions;
  tool_bake::BakerManager                          bakerManager(nvvkContext->m_device, *contextVK->resAllocator);
  tool_bake::BakerManagerConfig                    managerConfig;
  managerConfig.outTextureStem         = args.outputTextureStem;
  managerConfig.texturesToResample     = args.texturesToResample;
  managerConfig.resampleExtraTextures  = args.resampleExtraTextures;
  managerConfig.resampleResolution     = args.resampleResolution;
  managerConfig.quaternionTexturesStem = args.quaternionTexturesStem;
  managerConfig.offsetTexturesStem     = args.offsetTexturesStem;
  managerConfig.heightTexturesStem     = args.heightTexturesStem;
  if(!bakerManager.generateInstructions(managerConfig, &reference, base, resampleInstructions))
  {
    LOGE("BakerManager::generateInstructions() failed\n");
    return false;
  }

  // Determine how much memory we can use to load images and to tessellate
  // hi-res geometry, out of the limit of info.memLimitMb.
  uint64_t memLimitBytes = static_cast<uint64_t>(args.memLimitMb) << 20;
  uint64_t textureMemLimit, bakerMemLimit;
  if(memLimitBytes == 0)
  {
    textureMemLimit = 0;
    bakerMemLimit   = 0;
  }
  else
  {
    uint64_t texturesMinimumBytes, texturesIdealBytes;
    bakerManager.getTextureMemoryRequirements(texturesMinimumBytes, texturesIdealBytes, resampleInstructions);

    if(texturesMinimumBytes > memLimitBytes)
    {
      // We'll almost certainly exceed our VRAM limit. In this case, we refuse
      // to continue (to prevent the app as a whole from crashing). We print
      // a larger size than strictly needed so that we also have memory for
      // mesh data.
      LOGE(
          "Error: This file had a mesh that used textures that were too large to load into %d MB of GPU memory. Please "
          "consider re-running the baker with a higher memory limit, such as %zu MB.\n",
          args.memLimitMb, ((2 * texturesMinimumBytes) >> 20) + 1);
      return false;
    }
    else
    {
      // Above the minimum, we limit textures to 25% of total memory.
      // TODO: The 25% here is ad-hoc; it wouldbe nice to test a variety of
      // fractions and find the one that gives the best performance.
      textureMemLimit = std::max(texturesMinimumBytes, std::min(texturesIdealBytes, memLimitBytes / 4));
      bakerMemLimit   = memLimitBytes - textureMemLimit;
    }
  }
  bakerManager.setMemoryLimit(textureMemLimit);

  BakeOperator               bakeOperator(context.meshopsContext());
  meshops::OpBake_properties bakeProperties;
  meshops::meshopsBakeGetProperties(context.meshopsContext(), bakeOperator, bakeProperties);


  // Bake one mesh at a time
  std::vector<baryutils::BaryContentData> baryContents;
  std::vector<size_t>                     baryGroupToMeshIndex;
  for(size_t meshIndex = 0; meshIndex < base->meshes().size(); ++meshIndex)
  {
    LOGI("Mesh %zu/%zu\n", meshIndex + 1, base->meshes().size());

    const std::unique_ptr<ToolMesh>& baseMesh      = base->meshes()[meshIndex];
    const ToolMesh*    referenceMesh = reference.meshes()[meshIndex];

    // Create some temporary reference mesh data storage. Some attributes, such
    // as triangleSubdivLevels, may need to be generated for baking, but it
    // should not persist on the original mesh.
    meshops::MeshData referenceMeshAux;
    meshops::ResizableMeshView referenceView(referenceMesh->view(), meshops::makeResizableMeshViewCallback(referenceMeshAux));

    std::vector<std::unique_ptr<MeshopsTexture>> meshopsTextures;

    // Reference mesh heightmap config
    meshops::OpBake_heightmap heightmapDesc;
    meshops::TextureConfig    heightmapConfig{};
    heightmapDesc.normalizeDirections           = true;
    heightmapDesc.usesVertexNormalsAsDirections = referenceView.vertexDirections.empty() && !args.heightmapDirectionsGen;
    heightmapDesc.pnTriangles                   = args.heightmapPNtriangles;
    int                                        heightmapImageIndex;
    std::unique_ptr<micromesh_tool::ToolImage> heightmapOverride;
    micromesh_tool::ToolImage*                 heightmapImage{};
    if(!referenceView.triangleSubdivisionLevels.empty())
    {
      heightmapDesc.maxSubdivLevel =
          *std::max_element(referenceView.triangleSubdivisionLevels.begin(), referenceView.triangleSubdivisionLevels.end());
    }
    // Look for the reference mesh's heightmap - either given as an argument or in the mesh's material
    if(meshIndex < args.heightmaps.size())
    {
      heightmapOverride = std::make_unique<micromesh_tool::ToolImage>();
      if(heightmapOverride->create(fs::current_path(), args.heightmaps[meshIndex]) != micromesh::Result::eSuccess)
      {
        LOGE("Error: Failed to create image for heightmap\n");
        return false;
      }
      heightmapImage = heightmapOverride.get();
    }
    else if(reference.getHeightmap(referenceMesh->relations().material, heightmapDesc.bias, heightmapDesc.scale, heightmapImageIndex))
    {
      // Add global scale/bias from args
      BiasScalef globalBiasScale(args.heightmapBias, args.heightmapScale);
      BiasScalef biasScale = globalBiasScale * BiasScalef(heightmapDesc.bias, heightmapDesc.scale);
      heightmapDesc.bias   = biasScale.bias;
      heightmapDesc.scale  = biasScale.scale;

      heightmapImage = base->images()[heightmapImageIndex].get();
    }
    // Load the heightmap, if there is one
    if(heightmapImage)
    {
      // Lazy loading may fail
      std::unique_ptr<HeightMap>& heightmapSampler = heightmapImage->heigtmap();
      if(!heightmapSampler)
      {
        LOGE("Failed to read heightmap on scene\n");
        return false;
      }
      heightmapConfig.width            = heightmapSampler->width;
      heightmapConfig.height           = heightmapSampler->height;
      heightmapConfig.mips             = 1;
      heightmapConfig.baseFormat       = micromesh::Format::eR32_sfloat;
      heightmapConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;
      meshopsTextures.push_back(
          std::make_unique<MeshopsTexture>(context.meshopsContext(), meshops::eTextureUsageBakerHeightmapSource, heightmapConfig,
                                           heightmapSampler->width * heightmapSampler->height * sizeof(float),
                                           reinterpret_cast<const micromesh::MicromapValue*>(heightmapSampler->raw())));
      if(!meshopsTextures.back()->valid())
      {
        LOGE("Error: meshopsTextureCreateFromData() failed to import the heightmap texture\n");
        return false;
      }
      heightmapDesc.texture = *meshopsTextures.back();
    }

    // Load textures to resample and prepare output images
    const tool_bake::ResampleMeshInstructions& meshInstructions = resampleInstructions[meshIndex];
    if(!bakerManager.prepareTexturesForMesh(contextVK->queueGCT, contextVK->queueT, meshInstructions))
    {
      LOGE("Error: BakerManager::prepareTexturesForMesh() failed\n");
      return false;
    }

    const std::vector<tool_bake::ResampleTextureContainer> resampleTextures = bakerManager.getResampleTextures(meshInstructions);
    std::vector<meshops::OpBake_resamplerInput> resamplerInput;
    std::vector<meshops::Texture>               resamplerOutput;
    for(auto& texture : resampleTextures)
    {
      meshops::OpBake_resamplerInput input;

      // Texture to resample
      input.textureType = texture.texelContent;
      if(texture.input.texture.image != VK_NULL_HANDLE)
      {
        meshopsTextures.push_back(std::make_unique<MeshopsTexture>(context.meshopsContext(),
                                                                   meshops::eTextureUsageBakerResamplingSource, texture.input));
        if(!meshopsTextures.back()->valid())
        {
          LOGE("Error: meshopsTextureCreateVK() failed to import a resampler input texture\n");
          return false;
        }
        input.texture = *meshopsTextures.back();
      }

      // Distance texture
      meshopsTextures.push_back(std::make_unique<MeshopsTexture>(context.meshopsContext(), meshops::eTextureUsageBakerResamplingDistance,
                                                                 texture.distance));
      if(!meshopsTextures.back()->valid())
      {
        LOGE("Error: meshopsTextureCreateVK() failed to import a resampler distance texture\n");
        return false;
      }
      input.distance = *meshopsTextures.back();

      resamplerInput.push_back(input);

      // Output texture
      meshopsTextures.push_back(std::make_unique<MeshopsTexture>(context.meshopsContext(), meshops::eTextureUsageBakerResamplingDestination,
                                                                 texture.output));
      if(!meshopsTextures.back()->valid())
      {
        LOGE("Error: meshopsTextureCreateVK() failed to import a resampler distance texture\n");
        return false;
      }
      resamplerOutput.push_back(*meshopsTextures.back());
    }

    // Remove direction bounds from the input if args.discardDirectionBounds
    if(args.discardDirectionBounds && !baseMesh->view().vertexDirectionBounds.empty())
    {
      LOGW("Dicarding direction vector bounds on mesh %zu due to --discard-direction-bounds\n", meshIndex)
      baseMesh->view().resize(meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit, 0, 0);
    }

    bool                                baseFileHasDirectionBounds = !baseMesh->view().vertexDirectionBounds.empty();
    micromesh_tool::ToolSceneDimensions referenceSceneDimensions(reference);

    // Warn if the presence of direction vectors are taking precedence over
    // --maxDisplacement. To avoid spam, this should be done only if
    // --maxDisplacement was given on the command line. Since we don't know,
    // only warn if it's different to the default.
    if(!args.overrideDirectionLength && baseFileHasDirectionBounds && args.maxDisplacement != ToolBakeArgs().maxDisplacement)
    {
      LOGW("Warning: input mesh has displacement bounds. Ignoring --maxDisplacement.\n");
    }

    meshops::OpBake_settings settings;
    settings.memLimitBytes      = memLimitBytes;
    settings.maxTraceLength     = args.overrideDirectionLength || !baseFileHasDirectionBounds ?
                                      args.maxDisplacement * referenceSceneDimensions.radius / 100.0f :
                                      0.0f;
    settings.maxDistanceFactor  = args.maxDistanceFactor;
    settings.uniDirectional     = args.uniDirectional;
    settings.fitDirectionBounds = args.fitDirectionBounds;
    if(args.writeIntermediateMeshes)
    {
      settings.debugDisplacedReferenceMeshCallback = debugWriteDisplacedReferenceMesh;
      settings.debugDisplacedReferenceMeshUserPtr  = &meshIndex;
    }

    ToolBakeArgs::BakingMethod method;
    if(args.method == ToolBakeArgs::eCustomOrUniform)
    {
      method = baseMesh->view().triangleSubdivisionLevels.empty() ? ToolBakeArgs::eUniform : ToolBakeArgs::eCustom;
    }
    else
    {
      method = args.method;
    }

    if(method == ToolBakeArgs::eCustom)
    {
      // Error out if explicitly asking for custom subdiv levels and not eCustomOrUniform
      if(baseMesh->view().triangleSubdivisionLevels.empty())
      {
        LOGE("Error: missing subdivision levels in the base mesh, required by --subdivmode custom.\n");
        return false;
      }

      settings.level = 0;
      for(size_t i = 0; i < baseMesh->view().triangleSubdivisionLevels.size(); i++)
      {
        settings.level = std::max(settings.level, uint32_t(baseMesh->view().triangleSubdivisionLevels[i]));
      }
    }
    else
    {
      settings.level = args.level;
    }

    // Create the base mesh topology
    meshops::MeshTopologyData baseMeshTopology;
    if(buildTopologyData(context.meshopsContext(), baseMesh->view(), baseMeshTopology) != micromesh::Result::eSuccess)
    {
      LOGE("Error: failed to build mesh topology\n");
      return false;
    }

    // Make sure subdivision levels get generated unless explicitly requesting uniform values
    bool uniformSubdivLevels = method == ToolBakeArgs::eUniform;

    // Query the mesh attributes needed to bake
    meshops::OpBake_requirements meshRequirements;
    meshops::meshopsBakeGetRequirements(context.meshopsContext(), bakeOperator, settings, meshops::ArrayView(resamplerInput),
                                        uniformSubdivLevels, heightmapDesc.texture != nullptr,
                                        heightmapDesc.usesVertexNormalsAsDirections, meshRequirements);

    if(!uniformSubdivLevels)
    {
      // while the baker doesn't need the basemesh with triangle primitive flags, the resulting
      // mesh must be consistent for further processing / saving etc.
      meshRequirements.baseMeshAttribFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit;
    }

    meshops::MeshAttributeFlags referenceTopologyDependencies =
        generationRequiresTopology(referenceView.getMeshAttributeFlags(), meshRequirements.referenceMeshAttribFlags);
    bool buildReferenceTopology = referenceTopologyDependencies != 0 || meshRequirements.referenceMeshTopology;
    if(!meshRequirements.referenceMeshTopology && referenceTopologyDependencies != 0)
    {
      LOGI("Reference mesh topology is expensive to create but is needed to generate %s\n",
           meshops::meshAttribBitsString(referenceTopologyDependencies).c_str());
    }

    // The reference mesh topology is only required for attribute generation and
    // baking heightmaps. It's slow to build, so best to avoid if possible.
    meshops::MeshTopologyData referenceMeshTopology;
    if(buildReferenceTopology
       && buildTopologyData(context.meshopsContext(), referenceView, referenceMeshTopology) != micromesh::Result::eSuccess)
    {
      LOGE("Error: failed to build mesh topology\n");
      return false;
    }

    // Generate any missing attributes
    {
      // If we want uniform subdiv levels, we should not pass in a per-triangle array.
      // If we want generated subdiv levels we need to clear and re-generate existing ones.
      bool generateSubdivLevels = method == ToolBakeArgs::eAdaptive3D || method == ToolBakeArgs::eAdaptiveUV;
      if(method == ToolBakeArgs::eUniform || generateSubdivLevels)
      {
        if(!baseMesh->view().triangleSubdivisionLevels.empty())
        {
          LOGW("Warning: clearing base mesh's subdivision levels due to --subdivmode.\n");
        }
        baseMesh->view().triangleSubdivisionLevels = {};

        if(!baseMesh->view().trianglePrimitiveFlags.empty())
        {
          LOGW("Warning: clearing base mesh's primitive flags due to --subdivmode.\n");
        }
        baseMesh->view().trianglePrimitiveFlags = {};
      }

      // Warn if the input subdiv level is all ones or zeroes
      if(!baseMesh->view().triangleSubdivisionLevels.empty())
      {
        int maxSubdivLevel = *std::max_element(baseMesh->view().triangleSubdivisionLevels.begin(),
                                               baseMesh->view().triangleSubdivisionLevels.end());
        if(maxSubdivLevel < 2)
        {
          LOGW("Warning: max input subdivision level in the base mesh is only %i\n", maxSubdivLevel);
        }
      }

      // Base mesh
      {
        meshops::OpGenerateSubdivisionLevel_input baseSubdivSettings;
        baseSubdivSettings.maxSubdivLevel = settings.level;
        baseSubdivSettings.relativeWeight = args.adaptiveFactor;
        baseSubdivSettings.useTextureArea = method == ToolBakeArgs::eAdaptiveUV;
        if(baseSubdivSettings.useTextureArea)
        {
          if(heightmapDesc.texture == nullptr)
          {
            LOGE("Error: --subdivmode adaptiveUV given, but the reference mesh has no heightmap.\n");
            return false;
          }
          baseSubdivSettings.textureWidth  = heightmapConfig.width;
          baseSubdivSettings.textureHeight = heightmapConfig.height;
        }
        uint32_t          maxGeneratedSubdivLevel;
        micromesh::Result result =
            generateMeshAttributes(context.meshopsContext(), meshRequirements.baseMeshAttribFlags, &baseSubdivSettings,
                                   baseMeshTopology, baseMesh->view(), maxGeneratedSubdivLevel,
                                   NormalReduceOp::eNormalReduceNormalizedLinear, args.tangentAlgorithm);
        if(result != micromesh::Result::eSuccess)
        {
          LOGE("Error: generating attributes for base mesh %zu failed\n", meshIndex);
          return false;
        }
      }

      // Reference mesh
      // Updates heightmapDesc.maxSubdivLevel if subdiv levels are generated (it is unlikely to already have them)
      {
        meshops::OpGenerateSubdivisionLevel_input referenceSubdivSettings;
        referenceSubdivSettings.maxSubdivLevel  = bakeProperties.maxHeightmapTessellateLevel;
        referenceSubdivSettings.subdivLevelBias = args.highTessBias;
        referenceSubdivSettings.textureWidth    = heightmapConfig.width;
        referenceSubdivSettings.textureHeight   = heightmapConfig.height;
        referenceSubdivSettings.useTextureArea  = true;
        micromesh::Result result =
            generateMeshAttributes(context.meshopsContext(), meshRequirements.referenceMeshAttribFlags,
                                   &referenceSubdivSettings, referenceMeshTopology, referenceView,
                                   heightmapDesc.maxSubdivLevel, args.heightmapDirectionsOp, args.tangentAlgorithm);
        if(result != micromesh::Result::eSuccess)
        {
          LOGE("Error: generating attributes for reference mesh %zu failed\n", meshIndex);
          return false;
        }
      }
    }

    // If bounds exist after generation, make sure uniDirectional is set.
    if(!settings.uniDirectional && baseFileHasDirectionBounds)
    {
      LOGW("Warning:Enabling --uniDirectional because mesh has direction bounds\n");
      settings.uniDirectional = true;
    }

    baryutils::BaryBasicData baryUncompressedTemp;

    // Mesh transforms
    nvmath::mat4f baseMeshTransform      = baseMesh->firstInstanceTransform();
    nvmath::mat4f referenceMeshTransform = referenceMesh->firstInstanceTransform();

    // Bake and resample. Compute displacement distances for all base mesh microvertices-to-be.
    {
      meshops::OpBake_input input;
      input.settings          = settings;
      input.baseMeshView      = baseMesh->view();
      input.baseMeshTopology  = baseMeshTopology;
      input.referenceMeshView = referenceView;
      input.referenceMeshTopology =
          meshRequirements.referenceMeshTopology ? static_cast<const micromesh::MeshTopology*>(referenceMeshTopology) : nullptr;
      input.referenceMeshHeightmap                                   = heightmapDesc;
      input.resamplerInput                                           = meshops::ArrayView(resamplerInput);
      reinterpret_cast<nvmath::mat4f&>(input.baseMeshTransform)      = baseMeshTransform;
      reinterpret_cast<nvmath::mat4f&>(input.referenceMeshTransform) = referenceMeshTransform;

      // Create a BaryContentData object to hold the baker output. This is
      // linearized when saving so that each becomes its own bary group.
      baryGroupToMeshIndex.push_back(meshIndex);
      baryContents.emplace_back();

      meshops::OpBake_output output;
      output.resamplerTextures        = meshops::ArrayView(resamplerOutput);
      output.uncompressedDisplacement = args.compressed ? &baryUncompressedTemp : &baryContents.back().basic;
      output.vertexDirectionBounds    = baseMesh->view().vertexDirectionBounds;

      micromesh::Result result = meshops::meshopsOpBake(context.meshopsContext(), bakeOperator, input, output);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Error: Baking mesh %zu failed\n", meshIndex);
        return false;
      }
    }

    if(args.compressed)
    {
      bary::BasicView uncompressedView = baryUncompressedTemp.getView();

      meshops::OpCompressDisplacementMicromap_input input;
      input.meshTopology                       = baseMeshTopology;
      input.meshView                           = baseMesh->view();
      input.settings.minimumPSNR               = args.minPSNR;
      input.settings.validateInputs            = true;
      input.settings.validateOutputs           = true;
      input.uncompressedDisplacement           = &uncompressedView;
      input.uncompressedDisplacementGroupIndex = 0;

      meshops::OpCompressDisplacementMicromap_output output;
      output.compressedDisplacement           = &baryContents.back().basic;
      output.compressedDisplacementRasterMips = args.compressedRasterData ? &baryContents.back().misc : nullptr;

      micromesh::Result result = meshops::meshopsOpCompressDisplacementMicromaps(context.meshopsContext(), 1, &input, &output);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Error: Compressing mesh %zu failed\n", meshIndex);
        return false;
      }
    }

    // Subdiv levels on the base mesh can be deleted as they have been consumed
    // by the baker and should now be in BaryBasicData::triangles.
    baseMesh->view().resize(meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit, 0, 0);

    // Write the resampled textures to disk
    bakerManager.finishTexturesForMesh(contextVK->queueGCT, meshInstructions);
  }

  // Add the bary data, generating a relative path from the output directory
  std::unique_ptr<ToolBary> bary = std::make_unique<ToolBary>();
  if(bary->create(std::move(baryContents), args.baryFilename) != micromesh::Result::eSuccess)
  {
    return false;
  }
  size_t baryIndex = base->replaceBarys(std::move(bary));

  // Link the meshes with the bary groups in the micromap file. This is in case
  // we need to support skipping some meshes during baking.
  for(size_t groupIndex = 0; groupIndex < baryGroupToMeshIndex.size(); ++groupIndex)
  {
    size_t meshIndex = baryGroupToMeshIndex[groupIndex];
    base->linkBary(baryIndex, groupIndex, meshIndex);
  }

  // Debug meshes
  if(args.writeIntermediateMeshes)
  {
    writeDirectionBoundsMeshes(base);
  }

  /*
  double shellVolume = m_microMesh.baryBaker()->m_barySet.computeShellVolume(m_microMesh.meshSet(), true, true);
  LOGI("Shell volume: %f\n", shellVolume);
  */
  return true;
}

}  // namespace tool_bake
