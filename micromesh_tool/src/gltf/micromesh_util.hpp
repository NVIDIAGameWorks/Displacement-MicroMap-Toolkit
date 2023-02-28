//
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//

#include "NV_micromesh_extension_types.hpp"
#include <assert.h>
#include <tiny_gltf.h>
#include <nvh/gltfscene.hpp>

//--------------------------------------------------------------------------------------------------
//
// Utilities to get and set NV_barycentric_displacement information for a
// TinyGLTF primitive.
//

// Updates the extensionsUsed vector to include or not include the extensionName
// string depending on 'used'. Maintains order if the extension already exists.
void setExtensionUsed(std::vector<std::string>& extensionsUsed, const std::string& extensionName, bool used);

// Retrieves the NV_displacement_micromap extension from a TinyGLTF
// primitive. Returns whether the primitive had the NV_barycentric_displacement
// extension.
bool getPrimitiveDisplacementMicromap(const tinygltf::Primitive& primitive, NV_displacement_micromap& extension);

bool getMaterialsDisplacement(const tinygltf::Material& material, nvh::KHR_materials_displacement& extension);

void setMaterialsDisplacement(const nvh::KHR_materials_displacement& extension, tinygltf::Model& model, tinygltf::Material& material);

// Gets the previous version of NV_displacement_micromap,
// NV_barycentric_displacement. Automatically converts it to the new extensions,
// but returns the image in a separate field.
bool getPrimitiveLegacyBarycentricDisplacement(const tinygltf::Primitive& primitive,
                                               NV_displacement_micromap&  dispExt,
                                               NV_micromap_tooling&       toolExt,
                                               int32_t&                   image);

// Sets the NV_displacement_micromap for a TinyGLTF primitive, adding it
// if it doesn't exist.
void setPrimitiveDisplacementMicromap(tinygltf::Primitive& primitive, const NV_displacement_micromap& extension);

// Retrieves the NV_micromap_tooling extension from a tinygltf primitive.
// Returns whether the primitive had the NV_micromap_tooling extension.
bool getPrimitiveMicromapTooling(const tinygltf::Primitive& primitive, NV_micromap_tooling& extension);

// Sets the NV_micromap_tooling extension for a TinyGLTF primitive, adding it
// if it doesn't exist.
void setPrimitiveMicromapTooling(tinygltf::Primitive& primitive, const NV_micromap_tooling& extension);

// Retrieves the NV_micromaps extension as a vector of objects. Returns nullptr
// if it did not exist.
const tinygltf::Value::Array* getNVMicromapExtension(const tinygltf::Model& model);
tinygltf::Value::Array*       getNVMicromapExtensionMutable(tinygltf::Model& model);

// Returns number of micromaps in the glTF file. Returns false if the extension
// doesn't exist.
bool getGLTFMicromapCount(const tinygltf::Model& model, size_t& count);

// Retrieves the n'th micromap from a glTF file. Returns whether it succeeded.
bool getGLTFMicromap(const tinygltf::Model& model, int32_t n, NV_micromap& result);

// Sets data in the n'th micromap in a glTF file. Returns whether it succeeded.
bool setGLTFMicromap(tinygltf::Model& model, int32_t n, const NV_micromap& extension);

// Convert a NV_micromap object to tinygltf json
tinygltf::Value::Object createTinygltfMicromapObject(NV_micromap micromap);

// Adds a micromap to the NV_micromaps `micromaps` array, creating it if it
// doesn't exist. Returns its index. Does not update the extensionsUsed list;
// updateExtensions must be called manually.
int32_t addTinygltfMicromap(tinygltf::Model& model, const NV_micromap& nvMicromap);

// Add the given micromap URI to the model and returns its index, to be used in NV_displacement_micromap
int32_t addTinygltfMicromap(tinygltf::Model& model, const std::string& micromapUri);

void updateExtensionsUsed(tinygltf::Model& model);

// Updates a glTF file that uses the NV_barycentric_displacement extension to
// instead use the NV_displacement_micromap extension. The most significant
// change is that micromaps used to be stored in the `images` array, but now
// they use their own extension.
// Returns true on success. This can fail if the input glTF had invalid
// indices; in that case, it prints and returns false.
[[nodiscard]] bool updateNVBarycentricDisplacementToNVDisplacementMicromap(tinygltf::Model& model);
