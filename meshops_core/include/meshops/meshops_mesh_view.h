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
 * @file meshops_mesh_view.hpp
 * @brief Defines MeshAttributes and primary derivatives: MeshView,
 * MutableMeshView and ResizableMeshView.
 *
 * MeshAttributes is intended to be an abstract definition of mesh data. The
 * following classes instantiate it.
 *
 * - MeshData is a concrete std::vector backed mesh with its own data
 * - MutableMeshView is a non-const version of MeshView
 * - ResizableMeshView includes a resize() callback that an interface can use to
 *   populate data
 */

#pragma once

#include <cstdint>
#include <meshops/meshops_array_view.h>
#include <nvmath/nvmath.h>
#include <string>
#include <functional>
#include <type_traits>

namespace meshops {

enum MeshAttributeFlagBits : uint64_t
{
  eMeshAttributeTriangleVerticesBit       = 1ull << 0,
  eMeshAttributeTriangleSubdivLevelsBit   = 1ull << 1,
  eMeshAttributeTrianglePrimitiveFlagsBit = 1ull << 2,
  //eMeshAttributeTriangleMappingBit        = 1ull << 3, // TODO later

  eMeshAttributeVertexPositionBit        = 1ull << 8,
  eMeshAttributeVertexNormalBit          = 1ull << 9,
  eMeshAttributeVertexTangentBit         = 1ull << 10,
  // eMeshAttributeVertexBitangentBit       = 1ull << 11, // Not used (tangents have 4 components)
  eMeshAttributeVertexDirectionBit       = 1ull << 12,
  eMeshAttributeVertexDirectionBoundsBit = 1ull << 13,
  eMeshAttributeVertexImportanceBit      = 1ull << 14,
  // provide 8 bits for texcoords
  eMeshAttributeVertexTexcoordBit = 1ull << 16,
  // provide 8 bits for colors
  eMeshAttributeVertexColorBit = 1ull << 24,
};
typedef uint64_t MeshAttributeFlags;


// TODO remove this class, it was a leftover of the dark ages with meshset ;)
// meshops would not use it
struct MeshSlice
{
  size_t triangleOffset{};
  size_t triangleCount{};
  size_t vertexOffset{};
  size_t vertexCount{};

  MeshSlice() = default;
  MeshSlice(size_t triangleCount, size_t vertexCount)
      : triangleCount{triangleCount}
      , vertexCount{vertexCount}
  {
  }
  MeshSlice(size_t triangleOffset, size_t triangleCount, size_t vertexOffset, size_t vertexCount)
      : triangleOffset{triangleOffset}
      , triangleCount{triangleCount}
      , vertexOffset{vertexOffset}
      , vertexCount{vertexCount}
  {
  }

  bool operator==(const MeshSlice& other) const
  {
    return triangleOffset == other.triangleOffset && triangleCount == other.triangleCount
           && vertexOffset == other.vertexOffset && vertexCount == other.vertexCount;
  }
  bool operator!=(const MeshSlice& other) const { return !(*this == other); }
};

// A single definition of all mesh attributes to be used by MeshData, MeshView MutableMeshView and DynamicMeshView.
// This is facilitated by the ArrayType template template parameter, which could be ArrayView or std::vector.
template <template <class T> class ArrayType>
struct MeshAttributes
{
  ArrayType<nvmath::vec3ui> triangleVertices;
  ArrayType<nvmath::vec3f>  vertexPositions;
  ArrayType<nvmath::vec3f>  vertexNormals;
  ArrayType<nvmath::vec2f>  vertexTexcoords0;
  ArrayType<nvmath::vec4f>  vertexTangents;
  ArrayType<nvmath::vec3f>  vertexDirections;
  ArrayType<nvmath::vec2f>  vertexDirectionBounds;
  ArrayType<float>          vertexImportance;
  ArrayType<uint16_t>       triangleSubdivisionLevels;
  ArrayType<uint8_t>        trianglePrimitiveFlags;
  //ArrayType<uint32_t>     triangleMappings; // TODO later, we don't support this yet in our tools, but micromesh core does and we should anticipate

  // Return the size of the first non-empty triangle attribute or zero if there is none
  uint64_t triangleCount() const
  {
    if(!triangleVertices.empty())
      return triangleVertices.size();
    if(!triangleSubdivisionLevels.empty())
      return triangleSubdivisionLevels.size();
    if(!trianglePrimitiveFlags.empty())
      return trianglePrimitiveFlags.size();
    return 0;
  }

  // Return the size of the first non-empty vertex attribute or zero if there is none
  uint64_t vertexCount() const
  {
    if(!vertexPositions.empty())
      return vertexPositions.size();
    if(!vertexNormals.empty())
      return vertexNormals.size();
    if(!vertexTexcoords0.empty())
      return vertexTexcoords0.size();
    if(!vertexTangents.empty())
      return vertexTangents.size();
    if(!vertexDirections.empty())
      return vertexDirections.size();
    if(!vertexDirectionBounds.empty())
      return vertexDirectionBounds.size();
    if(!vertexImportance.empty())
      return vertexImportance.size();
    return 0;
  }

  uint64_t indexCount() const { return triangleCount() * 3; }

  MeshAttributeFlags getMeshAttributeFlags() const
  {
    MeshAttributeFlags meshAttributeFlags = 0;

    if(!triangleVertices.empty())
      meshAttributeFlags |= eMeshAttributeTriangleVerticesBit;
    if(!triangleSubdivisionLevels.empty())
      meshAttributeFlags |= eMeshAttributeTriangleSubdivLevelsBit;
    if(!trianglePrimitiveFlags.empty())
      meshAttributeFlags |= eMeshAttributeTrianglePrimitiveFlagsBit;

    if(!vertexPositions.empty())
      meshAttributeFlags |= eMeshAttributeVertexPositionBit;
    if(!vertexNormals.empty())
      meshAttributeFlags |= eMeshAttributeVertexNormalBit;
    if(!vertexTangents.empty())
      meshAttributeFlags |= eMeshAttributeVertexTangentBit;
    if(!vertexDirections.empty())
      meshAttributeFlags |= eMeshAttributeVertexDirectionBit;
    if(!vertexDirectionBounds.empty())
      meshAttributeFlags |= eMeshAttributeVertexDirectionBoundsBit;
    if(!vertexImportance.empty())
      meshAttributeFlags |= eMeshAttributeVertexImportanceBit;
    if(!vertexTexcoords0.empty())
      meshAttributeFlags |= eMeshAttributeVertexTexcoordBit;

    return meshAttributeFlags;
  }

  bool hasMeshAttributeFlags(MeshAttributeFlags flags) const { return (getMeshAttributeFlags() & flags) == flags; }

  MeshAttributeFlags hasInvalidVertexCounts() const
  {
    size_t refCount = vertexCount();

    if(!vertexNormals.empty() && vertexNormals.size() != refCount)
      return eMeshAttributeVertexNormalBit;
    if(!vertexTangents.empty() && vertexTangents.size() != refCount)
      return eMeshAttributeVertexTangentBit;
    if(!vertexDirections.empty() && vertexDirections.size() != refCount)
      return eMeshAttributeVertexDirectionBit;
    if(!vertexDirectionBounds.empty() && vertexDirectionBounds.size() != refCount)
      return eMeshAttributeVertexDirectionBoundsBit;
    if(!vertexImportance.empty() && vertexImportance.size() != refCount)
      return eMeshAttributeVertexImportanceBit;
    if(!vertexTexcoords0.empty() && vertexTexcoords0.size() != refCount)
      return eMeshAttributeVertexTexcoordBit;

    return 0;
  }

  MeshAttributeFlags hasInvalidTriangleCounts() const
  {
    size_t refCount = triangleCount();

    if(!trianglePrimitiveFlags.empty() && trianglePrimitiveFlags.size() != refCount)
      return eMeshAttributeTrianglePrimitiveFlagsBit;
    if(!triangleSubdivisionLevels.empty() && triangleSubdivisionLevels.size() != refCount)
      return eMeshAttributeTriangleSubdivLevelsBit;

    return 0;
  }

  bool consistent() const { return hasInvalidVertexCounts() == 0 && hasInvalidTriangleCounts() == 0; }

  bool empty() const
  {
    return triangleVertices.empty() && vertexPositions.empty() && vertexNormals.empty() && vertexTexcoords0.empty()
           && vertexTangents.empty() && vertexDirections.empty() && vertexDirectionBounds.empty()
           && vertexImportance.empty() && triangleSubdivisionLevels.empty() && trianglePrimitiveFlags.empty();
  }

  MeshAttributes()                            = default;
  MeshAttributes(const MeshAttributes& other) = default;

  // Constuct a mesh from any other kind of mesh, e.g. MeshView(MutableMeshView) or DynamicMeshView(MeshData). This
  // pattern is used multiple times in this file. The use of templates means this constructor will not prevent the
  // default constructors from being defined automatically.
  template <class OtherMeshType,
            // Only instantiate this constructor if..
            class = std::enable_if_t<
                // this wouldn't be a copy constructor, so we don't break the rule of 5.
                !std::is_same_v<MeshAttributes<ArrayType>, OtherMeshType>>>
  MeshAttributes(OtherMeshType&& mesh)
      : triangleVertices(std::forward<OtherMeshType>(mesh).triangleVertices)
      , vertexPositions(std::forward<OtherMeshType>(mesh).vertexPositions)
      , vertexNormals(std::forward<OtherMeshType>(mesh).vertexNormals)
      , vertexTexcoords0(std::forward<OtherMeshType>(mesh).vertexTexcoords0)
      , vertexTangents(std::forward<OtherMeshType>(mesh).vertexTangents)
      , vertexDirections(std::forward<OtherMeshType>(mesh).vertexDirections)
      , vertexDirectionBounds(std::forward<OtherMeshType>(mesh).vertexDirectionBounds)
      , vertexImportance(std::forward<OtherMeshType>(mesh).vertexImportance)
      , triangleSubdivisionLevels(std::forward<OtherMeshType>(mesh).triangleSubdivisionLevels)
      , trianglePrimitiveFlags(std::forward<OtherMeshType>(mesh).trianglePrimitiveFlags)
  {
    // Assert for a nicer const conversion error message
    constexpr bool thisConst  = std::is_same_v<MeshAttributes<ArrayType>, MeshAttributes<ConstArrayView>>;
    constexpr bool otherConst = std::is_same_v<OtherMeshType, MeshAttributes<ConstArrayView>>;
    static_assert(!(!thisConst && otherConst), "Can't construct a non const MeshView view from a MeshView");
  }

  template <class OtherMeshType>
  void copy_from(const OtherMeshType& srcMesh)
  {
    // Assert for a nicer const conversion error message
    constexpr bool thisConst = std::is_same_v<MeshAttributes<ArrayType>, MeshAttributes<ConstArrayView>>;
    static_assert(!thisConst, "Can't copy to a const MeshView");
    if(!srcMesh.triangleVertices.empty())
      std::copy(srcMesh.triangleVertices.begin(), srcMesh.triangleVertices.end(), triangleVertices.begin());
    if(!srcMesh.vertexPositions.empty())
      std::copy(srcMesh.vertexPositions.begin(), srcMesh.vertexPositions.end(), vertexPositions.begin());
    if(!srcMesh.vertexNormals.empty())
      std::copy(srcMesh.vertexNormals.begin(), srcMesh.vertexNormals.end(), vertexNormals.begin());
    if(!srcMesh.vertexTexcoords0.empty())
      std::copy(srcMesh.vertexTexcoords0.begin(), srcMesh.vertexTexcoords0.end(), vertexTexcoords0.begin());
    if(!srcMesh.vertexTangents.empty())
      std::copy(srcMesh.vertexTangents.begin(), srcMesh.vertexTangents.end(), vertexTangents.begin());
    if(!srcMesh.vertexDirections.empty())
      std::copy(srcMesh.vertexDirections.begin(), srcMesh.vertexDirections.end(), vertexDirections.begin());
    if(!srcMesh.vertexDirectionBounds.empty())
      std::copy(srcMesh.vertexDirectionBounds.begin(), srcMesh.vertexDirectionBounds.end(), vertexDirectionBounds.begin());
    if(!srcMesh.vertexImportance.empty())
      std::copy(srcMesh.vertexImportance.begin(), srcMesh.vertexImportance.end(), vertexImportance.begin());
    if(!srcMesh.triangleSubdivisionLevels.empty())
      std::copy(srcMesh.triangleSubdivisionLevels.begin(), srcMesh.triangleSubdivisionLevels.end(),
                triangleSubdivisionLevels.begin());
    if(!srcMesh.trianglePrimitiveFlags.empty())
      std::copy(srcMesh.trianglePrimitiveFlags.begin(), srcMesh.trianglePrimitiveFlags.end(), trianglePrimitiveFlags.begin());
  }

  template <class OtherMeshType>
  void copy_backward_from(const OtherMeshType& srcMesh)
  {
    // Assert for a nicer const conversion error message
    constexpr bool thisConst = std::is_same_v<MeshAttributes<ArrayType>, MeshAttributes<ConstArrayView>>;
    static_assert(!thisConst, "Can't copy to a const MeshView");
    if(!srcMesh.triangleVertices.empty())
      std::copy_backward(srcMesh.triangleVertices.begin(), srcMesh.triangleVertices.end(), triangleVertices.end());
    if(!srcMesh.vertexPositions.empty())
      std::copy_backward(srcMesh.vertexPositions.begin(), srcMesh.vertexPositions.end(), vertexPositions.end());
    if(!srcMesh.vertexNormals.empty())
      std::copy_backward(srcMesh.vertexNormals.begin(), srcMesh.vertexNormals.end(), vertexNormals.end());
    if(!srcMesh.vertexTexcoords0.empty())
      std::copy_backward(srcMesh.vertexTexcoords0.begin(), srcMesh.vertexTexcoords0.end(), vertexTexcoords0.end());
    if(!srcMesh.vertexTangents.empty())
      std::copy_backward(srcMesh.vertexTangents.begin(), srcMesh.vertexTangents.end(), vertexTangents.end());
    if(!srcMesh.vertexDirections.empty())
      std::copy_backward(srcMesh.vertexDirections.begin(), srcMesh.vertexDirections.end(), vertexDirections.end());
    if(!srcMesh.vertexDirectionBounds.empty())
      std::copy_backward(srcMesh.vertexDirectionBounds.begin(), srcMesh.vertexDirectionBounds.end(),
                         vertexDirectionBounds.end());
    if(!srcMesh.vertexImportance.empty())
      std::copy_backward(srcMesh.vertexImportance.begin(), srcMesh.vertexImportance.end(), vertexImportance.end());
    if(!srcMesh.triangleSubdivisionLevels.empty())
      std::copy_backward(srcMesh.triangleSubdivisionLevels.begin(), srcMesh.triangleSubdivisionLevels.end(),
                         triangleSubdivisionLevels.end());
    if(!srcMesh.trianglePrimitiveFlags.empty())
      std::copy_backward(srcMesh.trianglePrimitiveFlags.begin(), srcMesh.trianglePrimitiveFlags.end(),
                         trianglePrimitiveFlags.end());
  }
};

// TODO I would like to get rid of this `MeshViewBase` class, no need for the slice stuff anymore
// MeshViewBase is somewhat understandable to get the const/non-const etc.
// But this here seems layer upon layer upon layer. Really hard to digest.

// Common MeshView class that defines slice() and augment() - needed by views but not MeshData
template <template <class T> class ArrayType>
struct MeshViewBase : MeshAttributes<ArrayType>
{
  using slice_result =
      std::conditional_t<std::is_same_v<MeshViewBase<ArrayType>, MeshViewBase<ConstArrayView>>, MeshViewBase<ConstArrayView>, MeshViewBase<ArrayView>>;

  slice_result slice(const MeshSlice& slice) const
  {
    slice_result result;
    result.triangleVertices      = this->triangleVertices.slice_nonempty(slice.triangleOffset, slice.triangleCount);
    result.vertexPositions       = this->vertexPositions.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.vertexNormals         = this->vertexNormals.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.vertexTexcoords0      = this->vertexTexcoords0.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.vertexTangents        = this->vertexTangents.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.vertexDirections      = this->vertexDirections.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.vertexDirectionBounds = this->vertexDirectionBounds.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.vertexImportance      = this->vertexImportance.slice_nonempty(slice.vertexOffset, slice.vertexCount);
    result.triangleSubdivisionLevels = this->triangleSubdivisionLevels.slice_nonempty(slice.triangleOffset, slice.triangleCount);
    result.trianglePrimitiveFlags = this->trianglePrimitiveFlags.slice_nonempty(slice.triangleOffset, slice.triangleCount);
    return result;
  }

  // Fill missing attributes with those in the other MeshView
  template <class OtherMeshType>
  void augment(const OtherMeshType& other)
  {
    if(!this->triangleVertices.size())
      this->triangleVertices = other.triangleVertices;
    if(!this->vertexPositions.size())
      this->vertexPositions = other.vertexPositions;
    if(!this->vertexNormals.size())
      this->vertexNormals = other.vertexNormals;
    if(!this->vertexTexcoords0.size())
      this->vertexTexcoords0 = other.vertexTexcoords0;
    if(!this->vertexTangents.size())
      this->vertexTangents = other.vertexTangents;
    if(!this->vertexDirections.size())
      this->vertexDirections = other.vertexDirections;
    if(!this->vertexDirectionBounds.size())
      this->vertexDirectionBounds = other.vertexDirectionBounds;
    if(!this->vertexImportance.size())
      this->vertexImportance = other.vertexImportance;
    if(!this->triangleSubdivisionLevels.size())
      this->triangleSubdivisionLevels = other.triangleSubdivisionLevels;
    if(!this->trianglePrimitiveFlags.size())
      this->trianglePrimitiveFlags = other.trianglePrimitiveFlags;
  }

  // Replace attribs in the current mesh with those from the other mesh, filtered by attribute flags.
  void replace(const MeshViewBase& other, MeshAttributeFlags flags)
  {
    if((flags & eMeshAttributeTriangleVerticesBit) != 0)
    {
      this->triangleVertices = other.triangleVertices;
    }
    if((flags & eMeshAttributeVertexPositionBit) != 0)
    {
      this->vertexPositions = other.vertexPositions;
    }
    if((flags & eMeshAttributeVertexNormalBit) != 0)
    {
      this->vertexNormals = other.vertexNormals;
    }
    if((flags & eMeshAttributeVertexTexcoordBit) != 0)
    {
      this->vertexTexcoords0 = other.vertexTexcoords0;
    }
    if((flags & eMeshAttributeVertexTangentBit) != 0)
    {
      this->vertexTangents = other.vertexTangents;
    }
    if((flags & eMeshAttributeVertexDirectionBit) != 0)
    {
      this->vertexDirections = other.vertexDirections;
    }
    if((flags & eMeshAttributeVertexDirectionBoundsBit) != 0)
    {
      this->vertexDirectionBounds = other.vertexDirectionBounds;
    }
    if((flags & eMeshAttributeVertexImportanceBit) != 0)
    {
      this->vertexImportance = other.vertexImportance;
    }
    if((flags & eMeshAttributeTriangleSubdivLevelsBit) != 0)
    {
      this->triangleSubdivisionLevels = other.triangleSubdivisionLevels;
    }
    if((flags & eMeshAttributeTrianglePrimitiveFlagsBit) != 0)
    {
      this->trianglePrimitiveFlags = other.trianglePrimitiveFlags;
    }
    assert(this->consistent());
  }

  MeshViewBase()                          = default;
  MeshViewBase(const MeshViewBase& other) = default;

  // Forward any conversion constructor to MeshAttributes
  template <class OtherMeshType, class = std::enable_if_t<!std::is_same_v<MeshViewBase<ArrayType>, OtherMeshType>>>
  MeshViewBase(OtherMeshType&& mesh)
      : MeshAttributes<ArrayType>(std::forward<OtherMeshType>(mesh))
  {
  }
};

// TODO MeshView really should use MeshBase class not MeshViewBase anymore, given we removed slice etc.

using MeshView        = MeshViewBase<ConstArrayView>;
using MutableMeshView = MeshViewBase<ArrayView>;

static_assert(std::is_constructible_v<MeshView, MutableMeshView>);
static_assert(std::is_copy_constructible_v<MeshView>);
static_assert(std::is_copy_constructible_v<MutableMeshView>);
static_assert(std::is_trivially_copy_constructible_v<MeshView>);
static_assert(std::is_trivially_copy_constructible_v<MutableMeshView>);

// would favor this over DynamicMeshView
// this would allow someone to use interleaved attributes for example, cause there is no need that the
// arrays are individual resizeable compared to the old class

class ResizableMeshView : public MutableMeshView
{
public:
  // the function is expected to update the views content after resizing has completed
  // it must ensure the requested flags are provided
  using resize_callback =
      std::function<void(ResizableMeshView& meshView, MeshAttributeFlags flags, size_t triangleCount, size_t vertexCount)>;

  ResizableMeshView() = default;

  // Copying a resizable mesh is likely a bug as any resize events would cause
  // the original to become stale.
  ResizableMeshView(const ResizableMeshView& other)            = delete;
  ResizableMeshView& operator=(const ResizableMeshView& other) = delete;

  // Constructor with an explicit resize callback
  ResizableMeshView(const MutableMeshView& other, resize_callback resizeCallback)
      : MutableMeshView(other)
      , m_resizeCallback(resizeCallback)
  {
  }

  ResizableMeshView& resize(MeshAttributeFlags flags, size_t triangleCount, size_t vertexCount)
  {
    m_resizeCallback(*this, flags, triangleCount, vertexCount);
    return *this;
  }

  bool resizable() { return static_cast<bool>(m_resizeCallback); }

private:
  resize_callback m_resizeCallback;
};

static_assert(std::is_constructible_v<MeshView, ResizableMeshView>);

inline const char* meshAttribFlagName(MeshAttributeFlagBits flag)
{
  switch(flag)
  {
    case eMeshAttributeTriangleVerticesBit:
      return "TriangleVertices";
    case eMeshAttributeTriangleSubdivLevelsBit:
      return "TriangleSubdivLevels";
    case eMeshAttributeTrianglePrimitiveFlagsBit:
      return "TrianglePrimitiveFlags";
    //case TriangleAttributeMappingBit: return "TriangleMapping";
    case eMeshAttributeVertexPositionBit:
      return "VertexPositions";
    case eMeshAttributeVertexNormalBit:
      return "VertexNormals";
    case eMeshAttributeVertexTangentBit:
      return "VertexTangents";
    case eMeshAttributeVertexDirectionBit:
      return "VertexDirections";
    case eMeshAttributeVertexDirectionBoundsBit:
      return "VertexDirectionBounds";
    case eMeshAttributeVertexTexcoordBit:
      return "VertexTexcoords";
    case eMeshAttributeVertexColorBit:
      return "VertexColors";
    default:
      return "Invalid";
  }
};

inline std::string meshAttribBitsString(MeshAttributeFlags bits)
{
  std::string result;
  for(size_t i = 0; i < sizeof(bits) * 8; ++i)
  {
    if((bits & (1ull << i)) != 0)
    {
      if(!result.empty())
        result += "|";
      result += meshAttribFlagName(static_cast<MeshAttributeFlagBits>(1ull << i));
    }
  }
  if(result.empty())
    result = "none";
  return result;
};

}  // namespace meshops
