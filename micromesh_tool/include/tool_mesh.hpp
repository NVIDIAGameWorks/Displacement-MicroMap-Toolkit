/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include <meshops/meshops_mesh_view.h>
#include <meshops/meshops_mesh_data.h>
#include <tiny_gltf.h>
#include <bary/bary_types.h>

namespace micromesh_tool {

struct ToolSceneDimensions;

// Tracks a mesh view, that initially points to data from a gltf model, but may
// be aggregated and resized, in which case data is dynamically allocated in
// auxiliary buffers. Non-mesh data references to the input gltf model are also
// maintained, although pointers to buffers may contain stale information. The
// input gltf model must remain valid for the lifetime of this object and any
// copied from it.
class ToolMesh
{
public:
  struct Relations
  {
    Relations() = default;
    Relations(const tinygltf::Primitive& primitive);

    int32_t bary{-1};           // scene.barys()[bary]. May be -1.
    int32_t group{0};           // scene.barys()[bary].groups()[group]
    int32_t mapOffset{0};       // NV_displacement_micromap::mapOffset offset to mapIndices
    int32_t material{-1};       // scene.materials()[material]. May be -1.
    int32_t firstInstance{-1};  // scene.instances()[firstInstance]. May be -1.
  };

  struct Meta
  {
    Meta() = default;
    Meta(const tinygltf::Mesh& tinygltfMesh);

    std::string name;
  };

  // Constructs a ToolMesh with the initial view populated from the gltf model.
  // Warning: makeResizableMeshViewCallback() does not copy data after a resize,
  // so any ToolMesh::view().resize() will result in cleared new data. There are
  // currently no use cases where new/resized data is not completely rewritten.
  ToolMesh(tinygltf::Model*           model,
           const Relations&           relations,
           const Meta&                meta,
           const tinygltf::Primitive* primitive,
           const bary::ContentView*   baryMeshView);

  // Constructs a ToolMesh by moving data out of a MeshData, keeping non-mesh
  // gltf references from an original ToolMesh created from a model. This
  // ToolMesh must then be added back to the same ToolScene.
  explicit ToolMesh(const ToolMesh& other, meshops::MeshData&& initialData)
      : m_aux(std::move(initialData))
      , m_view(m_aux, makeResizableMeshViewCallback(m_aux))
      , m_relations(other.m_relations)
      , m_meta(other.m_meta)
  {
  }

  // Construct a ToolMesh from another ToolMesh but for a different scene.
  explicit ToolMesh(const ToolMesh& other)
      : m_aux(other.view())
      , m_view(m_aux, makeResizableMeshViewCallback(m_aux))
      , m_relations(other.relations())
      , m_meta(other.m_meta)
  {
  }

  bool isOriginalData() const;

  // Be very careful not to take a copy of the returned ResizableMeshView as
  // resize() will not update the view stored in ToolMesh.
  meshops::ResizableMeshView&       view() { return m_view; }
  const meshops::ResizableMeshView& view() const { return m_view; }
  Relations&                        relations() { return m_relations; }
  const Relations&                  relations() const { return m_relations; }
  Meta&                             meta() { return m_meta; }
  const Meta&                       meta() const { return m_meta; }

private:
  // Auxiliary mesh data. May be populated when rewriting mesh data or even for
  // translated data when loading from a gltf model. E.g. when mesh indices are
  // unsigned shorts, they are incompatible with unsigned ints in MeshView and
  // we need to create real storage rather than point to the existing data.
  meshops::MeshData m_aux;

  // The primary interface to the mesh data. Initially, this meshView holds
  // pointers to the input gltf model. Any resize operations cause those
  // attribute pointers to instead point to m_meshAux.
  meshops::ResizableMeshView m_view;

  // May point to stale mesh data if !isOriginalData().
  // TODO: can we remove this? Maybe replace with a material reference?
  const tinygltf::Primitive* m_gltfPrimitive{nullptr};

  Relations m_relations;

  Meta m_meta;

  // Allow access to the primitive from ToolSceneDimensions so it can access the
  // gltf position min/maxs.
  friend struct ToolSceneDimensions;
  const tinygltf::Primitive* gltfPrimitive() const { return m_gltfPrimitive; }
};

}  // namespace micromesh_tool
