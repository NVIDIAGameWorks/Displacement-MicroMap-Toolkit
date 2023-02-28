#pragma once

#include <string>

#define NV_GLTF_COMPONENT_TYPE_HALF_FLOAT 5131

#define NV_MICROMAPS "NV_micromaps"
#define BARY_MIME_TYPE "model/vnd.bary"
// References a micromap.
// https://gitlab-master.nvidia.com/barycentric_displacement/sdk/gltf_micromap_extensions/-/tree/main/extensions/2.0/Vendor/NV_micromaps
// Note: This is effectively the same as a simplified version of tinygltf::Image.
struct NV_micromap
{
  std::string uri{};  // A (percent-encoded) URI (or IRI) to a .bary file. (The extension allows formats other than BARY as well.)
  std::string mimeType{};  // Required if no URI: The MIME type of the embedded micromap. Should be BARY_MIME_TYPE for .bary format.
  int bufferView{-1};  // Required if no URI: The buffer view containing the embedded micromap.
};

#define NV_DISPLACEMENT_MICROMAP "NV_displacement_micromap"
// Specifies how to use a micromap for displacement.
// https://gitlab-master.nvidia.com/barycentric_displacement/sdk/gltf_micromap_extensions/-/blob/main/extensions/2.0/Vendor/NV_displacement_micromap/README.md
struct NV_displacement_micromap
{
  int32_t directionBounds{-1};  // Index of the VEC2 accessor for per-vertex (bias, scale).
  int32_t directionBoundsOffset{0};  // Overridden by directionBounds; start index into the eMeshDisplacementDirectionBounds property of the DM array.
  int32_t directions{-1};  // Index of the accessor for per-vertex displacement directions.
  int32_t directionsOffset{0};  // Overridden by directions; start index into the eMeshDisplacementDirections property of the DM array.
  int32_t groupIndex{0};   // Index of the group in the micromap.
  int32_t mapIndices{-1};  // Index of an accessor that provides a map from the triangle ID to the displacement map.
  int32_t mapIndicesOffset{0};  // Overridden by mapIndices; start index into the eMeshPrimitiveMappings property of the DM array.
  int32_t mapOffset{0};        // An unsigned integer that will be added to each of the mapIndices values.
  int32_t micromap{-1};        // Index of the micromap in the micromaps array.
  int32_t primitiveFlags{-1};  // Index of an accessor for per-primitive flags.
  int32_t primitiveFlagsOffset{0};  // Overriden by topologyFlags; start index into the MeshTriangleFlags property of the DM array.
};

#define NV_MICROMAP_TOOLING "NV_micromap_tooling"
// Used to transfer data between tools, when glTF is the main way to store data.
struct NV_micromap_tooling
{
  int32_t directionBounds{-1};    // Same as NV_displacement_micromap.
  int32_t directions{-1};         // Same as NV_displacement_micromap.
  int32_t mapIndices{-1};         // Same as NV_displacement_micromap.
  int32_t mapOffset{0};           // Same as NV_displacement_micromap.
  int32_t primitiveFlags{-1};     // Same as NV_displacement_micromap.
  int32_t subdivisionLevels{-1};  // Index of an accessor that gives an intended subdivision level for each triangle.
};

// Older version of the NV_displacement_micromap extension. This is here so
// that we can read files with the old extension, but we'll always write
// files with the new extension.
#define NV_LEGACY_BARYCENTRIC_DISPLACEMENT "NV_barycentric_displacement"
/*
struct Legacy_NV_barycentric_displacement
{
  int32_t directionBounds{-1};       // Same as NV_displacement_micromap.
  int32_t directionBoundsOffset{0};  // Same as NV_displacement_micromap.
  int32_t directions{-1};            // Same as NV_displacement_micromap.
  int32_t directionsOffset{0};       // Same as NV_displacement_micromap.
  int32_t groupOffset{0};            // Renamed to groupIndex in NV_displacement_micromap.
  int32_t image{-1};  // Micromaps used to be in the `images` array. In NV_displacement_micromap, they're in `micromaps`.
  int32_t mapIndices{-1};          // Same as NV_displacement_micromap.
  int32_t mapIndicesOffset{0};     // Same as NV_displacement_micromap.
  int32_t mapOffset{0};            // Same as NV_displacement_micromap.
  int32_t subdivisionLevels{-1};   // Moved to NV_micromap_tooling.
  int32_t topologyFlags{-1};       // Renamed to primitiveFlags in NV_displacement_micromap.
  int32_t topologyFlagsOffset{0};  // Renamed to primitiveFlagsOffset in NV_displacement_micromap.
};
*/
