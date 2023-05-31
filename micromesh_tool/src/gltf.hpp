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

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
#include <set>
#include <numeric>
#include "gltf/NV_micromesh_extension_types.hpp"
#include <meshops/meshops_mesh_view.h>
#include "nvmath/nvmath_types.h"
#include "tiny_gltf.h"
#include <meshops/meshops_array_view.h>
#include <meshops/meshops_types.h>
#include "mesh_view_conv.hpp"

template <class T>
using BinaryOp = const T(const T&, const T&);

template <class Iterator>
std::pair<typename Iterator::value_type, typename Iterator::value_type> minmax_elements_op(Iterator begin,
                                                                                           Iterator end,
                                                                                           BinaryOp<typename Iterator::value_type> opMin,
                                                                                           BinaryOp<typename Iterator::value_type> opMax)
{
  using T = std::remove_cv_t<typename Iterator::value_type>;
  T first = *begin++;
  return {std::reduce<Iterator, T, BinaryOp<typename Iterator::value_type>>(begin, end, first, opMin),
          std::reduce<Iterator, T, BinaryOp<typename Iterator::value_type>>(begin, end, first, opMax)};
}

template <class T>
const T nv_min2(const T& a, const T& b)
{
  return nvmath::nv_min(a, b);
}
template <class T>
const T nv_max2(const T& a, const T& b)
{
  return nvmath::nv_max(a, b);
}

template <class T>
inline std::vector<double> toDoubleVector(const T& v)
{
  return {static_cast<double>(v)};
}
inline std::vector<double> toDoubleVector(const nvmath::vec2f& v)
{
  return {v.x, v.y};
}
inline std::vector<double> toDoubleVector(const nvmath::vec3f& v)
{
  return {v.x, v.y, v.z};
}
inline std::vector<double> toDoubleVector(const nvmath::vec4f& v)
{
  return {v.x, v.y, v.z, v.w};
}

// Inserts src into dest and returns an iterator to the first inserted item
template <class Dest, class Src>
typename Dest::iterator appendRawData(Dest& dest, const Src& src)
{
  static_assert(alignof(typename Src::value_type) % alignof(typename Dest::value_type) == 0);
  return dest.insert(dest.end(), reinterpret_cast<typename Dest::value_type const*>(src.data()),
                     reinterpret_cast<typename Dest::value_type const*>(src.data() + src.size()));
}

// Inserts src into dest and returns an iterator to the first inserted item
template <class Dest, class T>
typename Dest::iterator appendRawElement(Dest& dest, const T& src)
{
  static_assert(alignof(T) % alignof(typename Dest::value_type) == 0);
  return dest.insert(dest.end(), reinterpret_cast<typename Dest::value_type const*>(&src),
                     reinterpret_cast<typename Dest::value_type const*>(&src + 1));
}

// Adds a buffer view to reference the given data. Returns the view ID.
int makeView(tinygltf::Model& model, int bufferID, size_t bufferOffsetBytes, size_t sizeBytes, size_t stride = 0, int target = TINYGLTF_TARGET_ARRAY_BUFFER);

// Adds a buffer view accessor with an offset. Returns the accessor ID.
int makeAccessor(tinygltf::Model&    model,
                 int                 viewID,
                 size_t              byteOffset,
                 size_t              elementCount,
                 int                 gltfComponentType,
                 int                 gltfType,
                 std::vector<double> minValues,
                 std::vector<double> maxValues);

template <typename T>
int makeAccessor(tinygltf::Model& model, meshops::ConstArrayView<T> data, int viewID, size_t viewOffsetBytes, size_t viewOffsetElements, size_t elementCount)
{
  // Compute vertex attrib bounds
  auto minMax = minmax_elements_op(data.begin() + viewOffsetElements, data.begin() + viewOffsetElements + elementCount,
                                   static_cast<BinaryOp<T>*>(&nv_min2<T>), static_cast<BinaryOp<T>*>(&nv_max2<T>));

  int componentType = tinygltfTypeInfo<T>::componentType;
  int type          = tinygltfTypeInfo<T>::type;

  size_t byteStride = model.bufferViews[viewID].byteStride;
  if(byteStride == 0)
    byteStride = static_cast<uint32_t>(sizeof(*data.data()));

  return makeAccessor(model, viewID, viewOffsetBytes + viewOffsetElements * byteStride, elementCount, componentType,
                      type, toDoubleVector(minMax.first), toDoubleVector(minMax.second));
}

/**
 * @brief Writes a MeshView to the gltf model, creating a new buffer that contains all attributes and positions.
 *
 * @param model Destination model to append the mesh data
 * @param meshView Input mesh data
 * @param writeDisplacementMicromapExt Uses NV_displacement_micromap if true, else NV_micromap_tooling.
 * @return Primitive structure with references to the added data. Not added to any of the model's meshes.
 */
tinygltf::Primitive tinygltfAppendPrimitive(tinygltf::Model&         model,
                                            const meshops::MeshView& meshView,
                                            bool                     writeDisplacementMicromapExt = false);

// List of gltf extensions created by or possibly conflicting with appendToTinygltfModel.
const std::set<std::string>& micromapExtensionNames();

/**
 * @brief Copies nodes, materials and extensions from one model to another, assuming matching mesh indices. This call is
 * intended to pair with appendToTinygltfModel(), to allow rewriting mesh data in a gltf model without affecting
 * materials etc.
 *
 * @param src Source model to copy non-mesh data from.
 * @param dst Source model to add non-mesh data to.
 * @param extensionFilter Set of extension names to ignore and not copy. Defaults to those created by or conflicting with appendToTinygltfModel.
 * @return true if the models were compatible, otherwise no changes are made.
 */
bool copyTinygltfModelExtra(const tinygltf::Model& src,
                            tinygltf::Model&       dst,
                            std::set<std::string>  extensionFilter = micromapExtensionNames());

void addTinygltfModelLinesMesh(tinygltf::Model&                  model,
                               const std::vector<uint32_t>&      indices,
                               const std::vector<nvmath::vec3f>& positions,
                               const std::string                 meshName,
                               const nvmath::mat4f               transform);
