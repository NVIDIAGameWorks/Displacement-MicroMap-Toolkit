//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <mesh_view_conv.hpp>
#include <type_traits>

// Utility template to give type Dst with the same constness of type Src. It
// could be used to return a const X when a function takes a const Y and
// non-const X when taking a non-const Y without having to write two functions.
template <class Src, class Dst>
using apply_const = std::conditional_t<std::is_const_v<Src>, std::add_const_t<Dst>, std::remove_const_t<Dst>>;

template <class T>
static bool setTinyGltfArrayView(apply_const<T, tinygltf::Model>& tmodel, int accessorID, meshops::ArrayView<T>& result)
{
  const tinygltf::Accessor&         accessor   = tmodel.accessors[accessorID];
  const tinygltf::BufferView&       bufferView = tmodel.bufferViews[accessor.bufferView];
  apply_const<T, tinygltf::Buffer>& buffer     = tmodel.buffers[bufferView.buffer];
  using AccessorInfo                           = tinygltfTypeInfo<std::remove_const_t<T>>;
  if(accessor.componentType != AccessorInfo::componentType || accessor.type != AccessorInfo::type)
  {
    return false;
  }
  auto ptr = buffer.data.data() + accessor.byteOffset + bufferView.byteOffset;
  result   = meshops::ArrayView<T>(reinterpret_cast<T*>(ptr), accessor.count,
                                 bufferView.byteStride ? bufferView.byteStride : sizeof(T));
  return true;
}

template <class T, class BaryAttribInfo>
static bool setBaryArrayView(const uint8_t* ptr, BaryAttribInfo* attribInfo, bary::Format expectedFormat, meshops::ArrayView<T>& result)
{
  if(!ptr || !attribInfo)
  {
    // If there is no source attribute, the update is considered successful.
    return true;
  }
  assert(attribInfo->elementByteSize == sizeof(T));
  assert(attribInfo->elementByteAlignment >= alignof(T));
  if(attribInfo->elementFormat != expectedFormat || attribInfo->elementByteSize != sizeof(T)
     || attribInfo->elementByteAlignment < alignof(T))
  {
    LOGE("Error: bary file has format %i, byte size %u, alignment %u (expected %i, %zu, %zu). Ignoring.\n",
         static_cast<int>(attribInfo->elementFormat), attribInfo->elementByteSize, attribInfo->elementByteAlignment,
         static_cast<int>(expectedFormat), sizeof(T), alignof(T));
    assert(!"unexpected attribute format in bary");
    return false;
  }
  result = meshops::ArrayView(reinterpret_cast<T*>(ptr), attribInfo->elementCount);
  return true;
}

// Copy values from a (pointer, size, stride) into the destination
// DynamicArrayView after resizing it to match.
template <class SrcT, class DstT>
static void copyConvertStrided(SrcT* srcPtr, size_t srcSize, size_t srcStride, meshops::DynamicArrayView<DstT>& dst)
{
  dst.resize(srcSize);

  // WARNING: this copies between signed and unsigned integers of different
  // sizes and could truncate or reinterpret data
  meshops::ArrayView<SrcT> src(srcPtr, srcSize, srcStride ? srcStride : sizeof(SrcT));
  // Use std::transform() instead of std::copy() to allow static_cast. dst may
  // be strided so memcpy cannot be used.
  std::transform(src.begin(), src.end(), dst.begin(), [](const SrcT& v) { return static_cast<DstT>(v); });
}

// Convert a tinygltf array of integer scalar types to the given
// DynamicArrayView. Types are copied with conversion, as though static_cast()
// were used. This converts between signed/unsigned and various sizes which may
// cause truncation and precision loss. No dynamic validation of the data is
// done. It will be resized to match the accessor before copying. Returns true
// on success.
template <class T>
static bool copyConvertTinyGltfIntScalar(const tinygltf::Model& tmodel, int accessorID, meshops::DynamicArrayView<T>& dst)
{
  static_assert(tinygltfTypeInfo<std::remove_const_t<T>>::type == TINYGLTF_TYPE_SCALAR,
                "vector types not implemented yet");
  const tinygltf::Accessor&   accessor   = tmodel.accessors[accessorID];
  const tinygltf::BufferView& bufferView = tmodel.bufferViews[accessor.bufferView];
  const tinygltf::Buffer&     buffer     = tmodel.buffers[bufferView.buffer];
  assert(accessor.type == tinygltfTypeInfo<std::remove_const_t<T>>::type);
  if(accessor.type != tinygltfTypeInfo<std::remove_const_t<T>>::type)
  {
    return false;
  }
  // Copy data from the accessor to the output dst, converting to type T.
  const unsigned char* ptr = buffer.data.data() + accessor.byteOffset + bufferView.byteOffset;
  switch(accessor.componentType)
  {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
      copyConvertStrided(reinterpret_cast<const int8_t*>(ptr), accessor.count, bufferView.byteStride, dst);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      copyConvertStrided(reinterpret_cast<const uint8_t*>(ptr), accessor.count, bufferView.byteStride, dst);
      break;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
      copyConvertStrided(reinterpret_cast<const int16_t*>(ptr), accessor.count, bufferView.byteStride, dst);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      copyConvertStrided(reinterpret_cast<const uint16_t*>(ptr), accessor.count, bufferView.byteStride, dst);
      break;
    case TINYGLTF_COMPONENT_TYPE_INT:
      copyConvertStrided(reinterpret_cast<const int32_t*>(ptr), accessor.count, bufferView.byteStride, dst);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      copyConvertStrided(reinterpret_cast<const uint32_t*>(ptr), accessor.count, bufferView.byteStride, dst);
      break;
    default:
      assert(!"unsupported component type");
      return false;
  }
  return true;
}

// Creates an ArrayView for the given tinygltf accessor. If the type doesn't
// match exactly, the fallback storage will be used to convert the data and an
// ArrayView of the converted data will be returned.
template <class T>
static bool setTinyGltfArrayViewWithFallback(apply_const<T, tinygltf::Model>&                   model,
                                             int                                                accessorID,
                                             meshops::DynamicArrayView<std::remove_const_t<T>>& fallback,
                                             meshops::ArrayView<T>&                             result)
{
  if(accessorID == -1)
  {
    // Consider setting the result a success when the attribute is missing.
    return true;
  }
  if(setTinyGltfArrayView<T>(model, accessorID, result))
  {
    return true;
  }
  else
  {
    // Automatically convert scalar integer types
    using AccessorInfo = tinygltfTypeInfo<std::remove_const_t<T>>;
    if constexpr(AccessorInfo::type == TINYGLTF_TYPE_SCALAR)
    {
      if(copyConvertTinyGltfIntScalar(model, accessorID, fallback))
      {
        LOGI("Converted attribute %i from gltf type (%i, %i) to (%i, %i).\n", accessorID, model.accessors[accessorID].type,
             model.accessors[accessorID].componentType, AccessorInfo::type, AccessorInfo::componentType);
        result = fallback;
        return true;
      }
      else
      {
        LOGE("Failed to convert attribute %i from gltf type (%i, %i) to (%i, %i).\n", accessorID,
             model.accessors[accessorID].type, model.accessors[accessorID].componentType, AccessorInfo::type,
             AccessorInfo::componentType);
      }
    }
    else
    {
      LOGW("Warnings: discarding attribute %i with unsupported gltf type (%i, %i). Expected (%i, %i).\n", accessorID,
           model.accessors[accessorID].type, model.accessors[accessorID].componentType, AccessorInfo::type,
           AccessorInfo::componentType);
    }
  }
  return false;
};

meshops::MeshView makeMeshView(const bary::ContentView* baryView, uint32_t groupIndex)
{
  meshops::MeshView result;
  // Set values from the bary view if they exist. Failures are ignored (an error is printed).
  (void)setBaryArrayView(baryView->mesh.meshDisplacementDirections, baryView->mesh.meshDisplacementDirectionsInfo,
                         bary::Format::eRGB32_sfloat, result.vertexDirections);
  (void)setBaryArrayView(baryView->mesh.meshDisplacementDirectionBounds, baryView->mesh.meshDisplacementDirectionBoundsInfo,
                         bary::Format::eRG32_sfloat, result.vertexDirectionBounds);
  (void)setBaryArrayView(baryView->mesh.meshTriangleFlags, baryView->mesh.meshTriangleFlagsInfo, bary::Format::eR8_uint,
                         result.trianglePrimitiveFlags);

  // (from bary::MeshView) uncommon, meant for debugging
  (void)setBaryArrayView(baryView->mesh.meshPositions, baryView->mesh.meshPositionsInfo, bary::Format::eRGB32_sfloat,
                         result.vertexPositions);
  meshops::ArrayView<const uint32_t> triangleIndices;
  (void)setBaryArrayView(baryView->mesh.meshTriangleIndices, baryView->mesh.meshTriangleIndicesInfo,
                         bary::Format::eR32_uint, triangleIndices);
  result.triangleVertices = meshops::ArrayView<const nvmath::vec3ui>(triangleIndices);

  // Optionally slice the mesh if the bary::MeshView has any MeshGroups
  if(baryView->mesh.meshGroupsCount != 0)
  {
    assert(groupIndex < baryView->mesh.meshGroupsCount);
    bary::MeshGroup meshGroup = baryView->mesh.meshGroups[groupIndex];
    meshops::MeshSlice slice(meshGroup.triangleFirst, meshGroup.triangleCount, meshGroup.vertexFirst, meshGroup.vertexCount);
    result = result.slice(slice);
  }
  return result;
}

meshops::MeshView makeMeshView(const bary::ContentView* baryView, uint32_t groupIndex, NV_displacement_micromap micromap, size_t triangleCount, size_t vertexCount)
{
  meshops::MeshView result = makeMeshView(baryView, groupIndex);
  result.vertexDirections = result.vertexDirections.slice_nonempty(static_cast<size_t>(micromap.directionsOffset), vertexCount);
  result.vertexDirectionBounds =
      result.vertexDirectionBounds.slice_nonempty(static_cast<size_t>(micromap.directionBoundsOffset), vertexCount);
  result.trianglePrimitiveFlags =
      result.trianglePrimitiveFlags.slice_nonempty(static_cast<size_t>(micromap.primitiveFlagsOffset), triangleCount);
  return result;
}

// In case a mutable view is required but we only have const source data, a copy
// must be made. If the given mutable view is not already set but there is data
// in the source, copy it to the fallback array and update the mutable view to
// point to it.
template <class T>
static void augmentMutableViewWithFallback(meshops::ArrayView<const T>&  source,
                                           meshops::DynamicArrayView<T>& fallback,
                                           meshops::ArrayView<T>&        mutableView)
{
  if(mutableView.empty() && !source.empty())
  {
    fallback.resize(source.size());
    std::copy(source.begin(), source.end(), fallback.begin());
    mutableView = fallback;
  }
}

template <class TinygltfModel = const tinygltf::Model>
static std::conditional_t<std::is_const_v<TinygltfModel>, meshops::MeshView, meshops::MutableMeshView> makeMeshViewTinygltf(
    TinygltfModel&             model,
    const tinygltf::Primitive& tinygltfPrim,
    meshops::DynamicMeshView&& fallbackStorage,
    const bary::ContentView*   baryView       = nullptr,
    uint32_t                   baryGroupIndex = 0)
{
  // Note(nbickford): The && in the function signature above makes
  // `fallbackStorage` into an rvalue (instead of lvalue) reference.
  // This allows a temporary to be passed to this function, which makes
  // makeMutableMeshView(..., DynamicMeshView(m_meshData)) work.
  // Pass-by-value would also work, but would require copying a
  // DynamicMeshView, which is large (>512 bytes).

  constexpr bool makingConstView = std::is_const_v<TinygltfModel>;
  std::conditional_t<makingConstView, meshops::MeshView, meshops::MutableMeshView> result;

  // Apply primitive triangle indices. GLTF stores triangle indices as an array of uints, not vec3ui. Unfortunately they
  // may be uint8_t or uint16_t, in which case a fallback allocation is needed.
  meshops::ArrayView<apply_const<TinygltfModel, uint32_t>> meshIndices;
  meshops::DynamicArrayView<uint32_t>                      fallbackIndices(fallbackStorage.triangleVertices);
  // Ignore failures to load attributes - not all are required.
  (void)setTinyGltfArrayViewWithFallback(model, tinygltfPrim.indices, fallbackIndices, meshIndices);
  result.triangleVertices = meshops::ArrayView<apply_const<TinygltfModel, nvmath::vec3ui>>(meshIndices);

  // Standard vertex attributes
  auto findGltfAttr = [&](const char* attrName) {
    auto it = tinygltfPrim.attributes.find(attrName);
    return it != tinygltfPrim.attributes.end() ? it->second : -1;
  };
  (void)setTinyGltfArrayViewWithFallback(model, findGltfAttr("POSITION"), fallbackStorage.vertexPositions, result.vertexPositions);
  (void)setTinyGltfArrayViewWithFallback(model, findGltfAttr("NORMAL"), fallbackStorage.vertexNormals, result.vertexNormals);
  (void)setTinyGltfArrayViewWithFallback(model, findGltfAttr("TEXCOORD_0"), fallbackStorage.vertexTexcoords0, result.vertexTexcoords0);
  (void)setTinyGltfArrayViewWithFallback(model, findGltfAttr("TANGENT"), fallbackStorage.vertexTangents, result.vertexTangents);

  // Apply attributes from NV_micromap_tooling extension, if it exists
  NV_micromap_tooling tooling;
  if(getPrimitiveMicromapTooling(tinygltfPrim, tooling))
  {
    (void)setTinyGltfArrayViewWithFallback(model, tooling.directions, fallbackStorage.vertexDirections, result.vertexDirections);
    (void)setTinyGltfArrayViewWithFallback(model, tooling.directionBounds, fallbackStorage.vertexDirectionBounds,
                                           result.vertexDirectionBounds);
    (void)setTinyGltfArrayViewWithFallback(model, tooling.subdivisionLevels, fallbackStorage.triangleSubdivisionLevels,
                                           result.triangleSubdivisionLevels);
    (void)setTinyGltfArrayViewWithFallback(model, tooling.primitiveFlags, fallbackStorage.trianglePrimitiveFlags,
                                           result.trianglePrimitiveFlags);
  }

  // Apply attributes from NV_displacement_micromap extension, if it exists
  NV_displacement_micromap micromap;
  if(getPrimitiveDisplacementMicromap(tinygltfPrim, micromap))
  {
    (void)setTinyGltfArrayViewWithFallback(model, micromap.directions, fallbackStorage.vertexDirections, result.vertexDirections);
    (void)setTinyGltfArrayViewWithFallback(model, micromap.directionBounds, fallbackStorage.vertexDirectionBounds,
                                           result.vertexDirectionBounds);
    (void)setTinyGltfArrayViewWithFallback(model, micromap.primitiveFlags, fallbackStorage.trianglePrimitiveFlags,
                                           result.trianglePrimitiveFlags);
  }

  // If the gltf file does not define these attributes, look in the bary file
  // if they're there. A MutableMeshView would require non-const pointers in
  // bary::MeshView. The data is instead copied into the fallbackStorage.
  if(baryView)
  {
    // TODO: allow conversion of bary arrays, just like we do with tinygltf
    meshops::MeshView baryMeshView =
        makeMeshView(baryView, baryGroupIndex, micromap, result.triangleCount(), result.vertexCount());
    if constexpr(makingConstView)
    {
      // Fill missing attributes with those from the bary file. Trivial if we're
      // making a const MeshView
      result.augment(baryMeshView);
    }
    else
    {
      // Copy extra attributes from the bary file into the fallback storage
      // because we need to provide a mutable MeshView.
      augmentMutableViewWithFallback(baryMeshView.triangleVertices, fallbackStorage.triangleVertices, result.triangleVertices);
      augmentMutableViewWithFallback(baryMeshView.vertexPositions, fallbackStorage.vertexPositions, result.vertexPositions);
      augmentMutableViewWithFallback(baryMeshView.vertexNormals, fallbackStorage.vertexNormals, result.vertexNormals);
      augmentMutableViewWithFallback(baryMeshView.vertexTexcoords0, fallbackStorage.vertexTexcoords0, result.vertexTexcoords0);
      augmentMutableViewWithFallback(baryMeshView.vertexTangents, fallbackStorage.vertexTangents, result.vertexTangents);
      augmentMutableViewWithFallback(baryMeshView.vertexDirections, fallbackStorage.vertexDirections, result.vertexDirections);
      augmentMutableViewWithFallback(baryMeshView.vertexDirectionBounds, fallbackStorage.vertexDirectionBounds,
                                     result.vertexDirectionBounds);
      augmentMutableViewWithFallback(baryMeshView.vertexImportance, fallbackStorage.vertexImportance, result.vertexImportance);
      augmentMutableViewWithFallback(baryMeshView.triangleSubdivisionLevels, fallbackStorage.triangleSubdivisionLevels,
                                     result.triangleSubdivisionLevels);
      augmentMutableViewWithFallback(baryMeshView.trianglePrimitiveFlags, fallbackStorage.trianglePrimitiveFlags,
                                     result.trianglePrimitiveFlags);
    }
  }

  if(!result.consistent())
  {
    assert(!"inconsistent tinygltf mesh view");
    result = {};
  }
  return result;
}

meshops::MeshView makeMeshView(const tinygltf::Model&     model,
                               const tinygltf::Primitive& tinygltfPrim,
                               meshops::DynamicMeshView&& fallbackStorage,
                               const bary::ContentView*   baryView,
                               uint32_t                   baryGroupIndex)
{
  return makeMeshViewTinygltf(model, tinygltfPrim, std::move(fallbackStorage), baryView, baryGroupIndex);
}

meshops::MutableMeshView makeMutableMeshView(tinygltf::Model&           model,
                                             const tinygltf::Primitive& tinygltfPrim,
                                             meshops::DynamicMeshView&& fallbackStorage,
                                             const bary::ContentView*   baryView,
                                             uint32_t                   baryGroupIndex)
{
  return makeMeshViewTinygltf(model, tinygltfPrim, std::move(fallbackStorage), baryView, baryGroupIndex);
}
