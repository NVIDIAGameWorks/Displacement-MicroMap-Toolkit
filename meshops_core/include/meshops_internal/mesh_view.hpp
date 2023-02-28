//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

/**
 * @file mesh_view.hpp
 * @brief Defines DynamicMeshView, which includes resize() callbacks from
 * DynamicArrayView
 *
 * It is common to linearize mesh data. To help accessing it, MeshView::slice()
 * is provided. The slice is defined by a MeshSlice struct, holding a range of
 * triangle and vertex indices.
 *
 * Equivalent MeshSet classes are defined to conveniently hold a single view of
 * linearized mesh data, and an array of slices for access to individual meshes.
 */

#pragma once

#include <cstdint>
#include <meshops/meshops_mesh_view.h>
#include <meshops/meshops_mesh_data.h>
#include <nvmath/nvmath.h>
#include <functional>

namespace meshops {

// TODO remove this class
struct DynamicMeshView : public MeshViewBase<DynamicArrayView>
{
  void resize_nonempty(size_t newTriangleCount, size_t newVertexCount)
  {
    assert(triangleVertices.empty() || triangleVertices.resizable());
    assert(vertexPositions.empty() || vertexPositions.resizable());
    assert(vertexNormals.empty() || vertexNormals.resizable());
    assert(vertexTexcoords0.empty() || vertexTexcoords0.resizable());
    assert(vertexTangents.empty() || vertexTangents.resizable());
    assert(vertexDirections.empty() || vertexDirections.resizable());
    assert(vertexDirectionBounds.empty() || vertexDirectionBounds.resizable());
    assert(triangleSubdivisionLevels.empty() || triangleSubdivisionLevels.resizable());
    assert(trianglePrimitiveFlags.empty() || trianglePrimitiveFlags.resizable());
    if(!triangleVertices.empty())
      triangleVertices.resize(newTriangleCount);
    if(!vertexPositions.empty())
      vertexPositions.resize(newVertexCount);
    if(!vertexNormals.empty())
      vertexNormals.resize(newVertexCount);
    if(!vertexTexcoords0.empty())
      vertexTexcoords0.resize(newVertexCount);
    if(!vertexTangents.empty())
      vertexTangents.resize(newVertexCount);
    if(!vertexDirections.empty())
      vertexDirections.resize(newVertexCount, {0.0F, 1.0F, 0.0F});
    if(!vertexDirectionBounds.empty())
      vertexDirectionBounds.resize(newVertexCount, {0.0F, 1.0F});
    if(!triangleSubdivisionLevels.empty())
      triangleSubdivisionLevels.resize(newTriangleCount);
    if(!trianglePrimitiveFlags.empty())
      trianglePrimitiveFlags.resize(newTriangleCount);
  }

  void resize_nonempty(size_t newTriangleCount, size_t newVertexCount, const MeshView& matchEmpty)
  {
    assert(matchEmpty.triangleVertices.empty() || triangleVertices.resizable());
    assert(matchEmpty.vertexPositions.empty() || vertexPositions.resizable());
    assert(matchEmpty.vertexNormals.empty() || vertexNormals.resizable());
    assert(matchEmpty.vertexTexcoords0.empty() || vertexTexcoords0.resizable());
    assert(matchEmpty.vertexTangents.empty() || vertexTangents.resizable());
    assert(matchEmpty.vertexDirections.empty() || vertexDirections.resizable());
    assert(matchEmpty.vertexDirectionBounds.empty() || vertexDirectionBounds.resizable());
    assert(matchEmpty.triangleSubdivisionLevels.empty() || triangleSubdivisionLevels.resizable());
    assert(matchEmpty.trianglePrimitiveFlags.empty() || trianglePrimitiveFlags.resizable());
    if(!matchEmpty.triangleVertices.empty())
      triangleVertices.resize(newTriangleCount);
    if(!matchEmpty.vertexPositions.empty())
      vertexPositions.resize(newVertexCount);
    if(!matchEmpty.vertexNormals.empty())
      vertexNormals.resize(newVertexCount);
    if(!matchEmpty.vertexTexcoords0.empty())
      vertexTexcoords0.resize(newVertexCount);
    if(!matchEmpty.vertexTangents.empty())
      vertexTangents.resize(newVertexCount);
    if(!matchEmpty.vertexDirections.empty())
      vertexDirections.resize(newVertexCount, {0.0F, 1.0F, 0.0F});
    if(!matchEmpty.vertexDirectionBounds.empty())
      vertexDirectionBounds.resize(newVertexCount, {0.0F, 1.0F});
    if(!matchEmpty.triangleSubdivisionLevels.empty())
      triangleSubdivisionLevels.resize(newTriangleCount);
    if(!matchEmpty.trianglePrimitiveFlags.empty())
      trianglePrimitiveFlags.resize(newTriangleCount);
  }

  void resize(MeshAttributeFlags triangleFlags, size_t triangleCount, MeshAttributeFlags vertexFlags, size_t vertexCount)
  {
    if((triangleFlags & eMeshAttributeTriangleVerticesBit) != 0)
    {
      triangleVertices.resize(triangleCount);
    }
    if((vertexFlags & eMeshAttributeVertexPositionBit) != 0)
    {
      vertexPositions.resize(vertexCount);
    }
    if((vertexFlags & eMeshAttributeVertexNormalBit) != 0)
    {
      vertexNormals.resize(vertexCount);
    }
    if((vertexFlags & eMeshAttributeVertexTexcoordBit) != 0)
    {
      vertexTexcoords0.resize(vertexCount);
    }
    if((vertexFlags & eMeshAttributeVertexTangentBit) != 0)
    {
      vertexTangents.resize(vertexCount);
    }
    if((vertexFlags & eMeshAttributeVertexDirectionBit) != 0)
    {
      vertexDirections.resize(vertexCount);
    }
    if((vertexFlags & eMeshAttributeVertexDirectionBoundsBit) != 0)
    {
      vertexDirectionBounds.resize(vertexCount);
    }
    if((triangleFlags & eMeshAttributeTriangleSubdivLevelsBit) != 0)
    {
      triangleSubdivisionLevels.resize(triangleCount);
    }
    if((triangleFlags & eMeshAttributeTrianglePrimitiveFlagsBit) != 0)
    {
      trianglePrimitiveFlags.resize(triangleCount);
    }
    assert(consistent());
  }

  // Append data from the meshView into this one. Returns the MeshSlice for the inserted range of data.
  MeshSlice append(const MeshView& meshView)
  {
    MeshSlice newSlice(triangleCount(), meshView.triangleCount(), vertexCount(), meshView.vertexCount());
    resize_nonempty(triangleCount() + meshView.triangleCount(), vertexCount() + meshView.vertexCount(), meshView);
    this->slice(newSlice).copy_from(meshView);
    return newSlice;
  }

  DynamicMeshView()                             = default;
  DynamicMeshView(const DynamicMeshView& other) = default;

  // Forward any conversion constructor to MeshAttributes
  template <class OtherMeshType, class = std::enable_if_t<!std::is_same_v<DynamicMeshView, OtherMeshType>>>
  DynamicMeshView(OtherMeshType&& mesh)
      : MeshViewBase<DynamicArrayView>(std::forward<OtherMeshType>(mesh))
  {
  }
};

static_assert(std::is_constructible_v<MutableMeshView, DynamicMeshView>);
static_assert(std::is_constructible_v<DynamicMeshView, MeshData>);
static_assert(std::is_copy_constructible_v<DynamicMeshView>);

// TODO remove this class
struct MeshSetData
{
  MeshData               flat;
  std::vector<MeshSlice> slices;

  MeshSetData() = default;

  // Forward conversion constructor to MeshViewBase
  template <class OtherMeshSetType, class = std::enable_if_t<!std::is_same_v<MeshSetData, OtherMeshSetType>>>
  MeshSetData(OtherMeshSetType&& meshSet)
      : flat{std::forward<OtherMeshSetType>(meshSet).flat}
      , slices{std::forward<OtherMeshSetType>(meshSet).slices}
  {
  }

  MeshView        slice(size_t idx) const { return MeshView(flat).slice(slices[idx]); }
  MutableMeshView slice(size_t idx) { return MutableMeshView(flat).slice(slices[idx]); }
};

// TODO remove this class
// Common MeshViewSet class that holds a single linearized mesh and an array of slices to index into it
template <class MeshType>
struct MeshSetViewBase
{
  MeshType               flat;
  std::vector<MeshSlice> slices;

  MeshSetViewBase() = default;

  // Forward conversion constructor to MeshViewBase
  template <class OtherMeshSetType, class = std::enable_if_t<!std::is_same_v<MeshSetViewBase<MeshType>, OtherMeshSetType>>>
  MeshSetViewBase(OtherMeshSetType&& meshSet)
      : flat{std::forward<OtherMeshSetType>(meshSet).flat}
      , slices{std::forward<OtherMeshSetType>(meshSet).slices}
  {
  }

  typename MeshType::slice_result slice(size_t idx) const { return flat.slice(slices[idx]); }
};

using MeshSetView        = MeshSetViewBase<MeshView>;
using MutableMeshSetView = MeshSetViewBase<MutableMeshView>;
using DynamicMeshSetView = MeshSetViewBase<DynamicMeshView>;

static_assert(std::is_constructible_v<MeshSetView, MutableMeshSetView>);
static_assert(std::is_constructible_v<MutableMeshSetView, DynamicMeshSetView>);
static_assert(std::is_constructible_v<DynamicMeshSetView, MeshSetData>);
static_assert(std::is_copy_constructible_v<MeshSetData>);
static_assert(std::is_copy_constructible_v<MeshSetView>);
static_assert(std::is_copy_constructible_v<MutableMeshSetView>);
static_assert(std::is_copy_constructible_v<DynamicMeshSetView>);

// TODO remove this class
// A wrappter to add mutable auxiliary data to a mesh view that is missing some attributes.
// Note that the readable() view is invalidated if anything resizes writable() attributes.
// The intent behind this class is to have readable() always re-augment m_base with any new data added to m_auxiliary.
class MeshSetViewAux
{
public:
  MeshSetViewAux(const MeshSetView& base)
      : m_base{base}
      , m_auxiliary{}
  {
    m_auxiliary.slices = m_base.slices;
  }

  // This does NOT include arrays in the base view! They are const and cannot be converted to a DynamicMeshSetView.
  // alt name: overlay()
  MeshSetData& auxiliary() { return m_auxiliary; }

  // Returns a MeshSetView of the base mesh, where any missing attributes are replaced with those from the auxiliary
  // structure. NOTE: Resizing the auxiliary structure invalidates the underlay view.
  MeshSetView underlay()
  {
    MeshSetView result(m_base);
    result.flat.augment(m_auxiliary.flat);
    return result;
  }

  // Returns a MeshSetView of the base mesh, but replaces attributes with any existing ones from the auxiliary
  // structure. NOTE: Resizing the auxiliary structure invalidates the underlay view.
  MeshSetView overlay()
  {
    MeshSetView result(m_auxiliary);
    result.flat.augment(m_base.flat);
    return result;
  }

private:
  MeshSetView m_base;
  MeshSetData m_auxiliary;
};

}  // namespace meshops
