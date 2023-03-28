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

#pragma once

#include <memory>
#include <tool_mesh.hpp>
#include <tool_bary.hpp>
#include <tool_image.hpp>
#include <micromesh/micromesh_types.h>
#include <filesystem>
#include <tiny_gltf.h>
#include <meshops/meshops_operations.h>
#include <vector>
#include <set>
#include <filesystem>

namespace micromesh_tool {

namespace fs = std::filesystem;

// Loads *.gltf, *.glb or *.obj into a tinygltf::Model representation
[[nodiscard]] bool loadTinygltfModel(const fs::path& filename, tinygltf::Model& model);

// Saves a tinygltf::Model to a *.gltf or *.glb file.
// `model` should effectively be const, but isn't marked as such due to
// TinyGLTF's function signature here.
[[nodiscard]] bool saveTinygltfModel(const fs::path& filename, tinygltf::Model& model);

/**
 * @brief Class to store MeshViews of a tinygltf Model
 *
 * The Model must remain valid for the lifetime of this object and its mesh views.
 */
// TODO: rename to GltfMeshViews
class ToolScene
{
public:
  struct PrimitiveInstance
  {
    nvmath::mat4f worldMatrix{};
    int           primMeshRef{-1};  // reference to the array of primMesh
    int           material{-1};     // unique primitive material
    std::string   name;
  };

  micromesh::Result create(const fs::path& filename);
  micromesh::Result create(std::unique_ptr<tinygltf::Model> model, const fs::path basePath);
  micromesh::Result create(std::unique_ptr<tinygltf::Model>          model,
                           std::vector<std::unique_ptr<ToolImage>>&& images,
                           std::vector<std::unique_ptr<ToolBary>>&&  barys);
  micromesh::Result create(const std::unique_ptr<ToolScene>& source);

  void destroy();

  ToolScene() = default;
  ~ToolScene()
  {
    assert(m_meshes.empty());  // call ToolBary::destroy()
  }

  void write(tinygltf::Model& output, const std::set<std::string>& extensionFilter = {}, bool writeDisplacementMicromapExt = false);

  bool save(const fs::path& filename);

  // Mutable material properties, required by the resampler to generate new output textures.
  // TODO: have the resampler provide the output to apply to the new file without changing the original.
  std::vector<tinygltf::Material>&       materials() { return m_model->materials; }
  const std::vector<tinygltf::Material>& materials() const { return m_model->materials; }
  std::vector<tinygltf::Texture>&        textures() { return m_model->textures; }
  std::vector<tinygltf::Texture>&        textures() const { return m_model->textures; }

  // Returns true and sets heightmap information if it exists for the given material
  bool getHeightmap(int materialID, float& bias, float& scale, int& imageIndex) const;

  void setMesh(size_t meshIndex, std::unique_ptr<ToolMesh> mesh);

  void setImage(size_t imageIndex, std::unique_ptr<ToolImage> image);

  // Clears all barys and replaces them with a single entry. There is no use
  // case for mixed references. Returns the index of the added bary, which
  // will be zero.
  size_t replaceBarys(std::unique_ptr<ToolBary> bary);

  // Updates the gltf Model to mark the ToolMesh at meshIndex as displaced by
  // the ToolBary at baryIndex's group groupIndex. Gltf calls these micromaps -
  // a gltf micromap references a bary file. Removes any existing displacement
  // references in the gltf, e.g. previous micromap or heightmap.
  void linkBary(size_t baryIndex, size_t groupIndex, size_t meshIndex);

  // Creates a new un-allocated image. Returns the index to be used in
  // images()[index] and referencing the new image in the gltf textures() array.
  // images()[index].create() should be called after this.
  uint32_t createImage();

  // Creates a new image that is not part of the scene, but will be saved at the
  // same time later on.
  ToolImage& createAuxImage();

  // Clears all m_barys and removes references from gltf primitives.
  void clearBarys();

  // Getters
  const std::vector<std::unique_ptr<ToolMesh>>&  meshes() { return m_meshes; }
  const std::vector<std::unique_ptr<ToolBary>>&  barys() { return m_barys; }
  const std::vector<std::unique_ptr<ToolImage>>& images() { return m_images; }
  meshops::ArrayView<PrimitiveInstance>          instances() { return m_primInstances; }  // Mutable but not resizable
  const std::vector<PrimitiveInstance>&          getPrimitiveInstances() const { return m_primInstances; }

  // Const getters. These casts are probably UB, but it's less buggy than trying
  // to sync two vectors and faster than populating a const vector for every
  // call.
  const std::vector<std::unique_ptr<const ToolMesh>>& meshes() const
  {
    return reinterpret_cast<const std::vector<std::unique_ptr<const ToolMesh>>&>(m_meshes);
  }
  const std::vector<std::unique_ptr<const ToolBary>>& barys() const
  {
    return reinterpret_cast<const std::vector<std::unique_ptr<const ToolBary>>&>(m_barys);
  }
  const std::vector<std::unique_ptr<const ToolImage>>& images() const
  {
    return reinterpret_cast<const std::vector<std::unique_ptr<const ToolImage>>&>(m_images);
  }

  // Valid only if the scene exist, not empty and meshes exist
  bool valid() const { return m_model && !m_model->scenes.empty() && !m_model->meshes.empty(); }

  // Getter for the input gltf model. This will contain stale or invalid mesh
  // data and invalid relations to it. It is used to store materials and other
  // non-mesh data. The model is required to save a new mesh with original
  // transforms and extensions, but disallow in-place mesh modification by not
  // providing a non-const ref.
  const tinygltf::Model& model() const { return *m_model; }

  // Return true if all the data in the views came from the gltf file.
  bool isOriginalData() const
  {
    for(auto& mesh : meshes())
      if(!mesh->isOriginalData())
        return false;
    return true;
  }

private:
  ToolScene& operator=(ToolScene&& other) = default;

  // Construct from a filled model. The base path is used to find and load bary
  // files on-demand during getBaryView().
  ToolScene(std::unique_ptr<tinygltf::Model> model, const fs::path basePath)
      : m_model(std::move(model))
  {
    loadBarys(basePath);
    loadImages(basePath);
    createViews();
  }

  // Construct from a filled model and populate barys with in-memory bary
  // data for getBaryView().
  ToolScene(std::unique_ptr<tinygltf::Model>          model,
            std::vector<std::unique_ptr<ToolImage>>&& images,
            std::vector<std::unique_ptr<ToolBary>>&&  barys)
      : m_model(std::move(model))
      , m_images(std::move(images))
      , m_barys(std::move(barys))
  {
    createViews();
  }

  // Create ToolBary objects from the current m_model
  bool loadBarys(const fs::path& basePath);

  // Create ToolImage objects from the current m_model
  bool loadImages(const fs::path& basePath);

  // Create ToolMesh and PrimitiveInstance objects from the current m_model
  void createViews();

  // Input model. Referenced by ToolMesh. Data may be modified in-place or
  // overridden completely.
  std::unique_ptr<tinygltf::Model> m_model;

  // ToolMesh references that may be backed by either m_model or their own
  // storage.
  std::vector<std::unique_ptr<ToolMesh>> m_meshes;

  // ToolBary references that may be backed by a mapped file or bary
  // displacements generated by tool_bake.
  std::vector<std::unique_ptr<ToolBary>> m_barys;

  // ToolImage references that may be backed by just a location on disk, a file
  // loaded into memory or runtime-generated image data, e.g. from the
  // resampler.
  std::vector<std::unique_ptr<ToolImage>> m_images;

  // Flat list of all visible ToolMesh with their world matrix.
  std::vector<PrimitiveInstance> m_primInstances;

  // Textures generated by the resampler that are not part of the scene, but
  // need to be saved in the same location. E.g. quaternion and offset maps,
  // generated heightmaps and extra resampled textures.
  std::vector<std::unique_ptr<ToolImage>> m_auxImages;
};

struct ToolSceneDimensions
{
  ToolSceneDimensions(const ToolScene& scene);
  nvmath::vec3f min = nvmath::vec3f(std::numeric_limits<float>::max());
  nvmath::vec3f max = nvmath::vec3f(std::numeric_limits<float>::lowest());
  nvmath::vec3f size{0.f};
  nvmath::vec3f center{0.f};
  float         radius{0};
};

// Utility class to provide a summary for identifying intermediate meshes and
// their state with a human readable string.
struct ToolSceneStats
{
  ToolSceneStats(const ToolScene& scene);
  std::string str();
  size_t      triangles{};
  size_t      vertices{};
  size_t      images{};
  bool        micromaps{};
  bool        heightmaps{};
  bool        normalmapsMissingTangents{};
  uint32_t    maxBarySubdivLevel{};
};

void sceneWriteDebug(const ToolScene& scene, std::ostream& os);

}  // namespace micromesh_tool
