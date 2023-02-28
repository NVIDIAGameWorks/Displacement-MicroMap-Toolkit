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

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

#include "bary/bary_types.h"
#include "meshops_api.h"
#include "meshops_types.h"
#include <baryutils/baryutils.h>
#include <micromesh/micromesh_displacement_compression.h>

// meshops-level core operations are implemented here
// unless mentioned otherwise all operations

namespace meshops {

// `meshops::Context` is the main object to do any `meshops` operation.
// For operations that require device support (GPU-baking etc.)
// it will host the Vulkan context. Furthermore it contains the `micromesh::OpContext` to
// drive low-level `micromesh` operations.
//
// Unless mentioned otherwise all operations are synchronous, i.e. results are
// directly visible after their execution. This is not ideal, nor recommended for
// GPU operations in an interactive or real-time scenario, but as this serves
// as research and sample platform, it is good enough.

typedef class Context_c* Context;

struct ContextConfig
{
  // some level of automatic threading (std::threads) can be used
  // by setting this to > 1
  uint32_t threadCount = 1;
  // warning / log verbosity level
  uint32_t verbosityLevel = 0;
  // general error and warning callback
  micromesh::MessageCallbackInfo messageCallback = {};
  // several operations require a device-side context (i.e. Vulkan)
  bool requiresDeviceContext = false;
};

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsContextCreate(const ContextConfig& config, Context* pContext);
MESHOPS_API void MESHOPS_CALL              meshopsContextDestroy(Context context);
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsContextGetConfig(Context context, ContextConfig* config);


//////////////////////////////////////////////////////////////////////////

// `meshops::DeviceMesh` object is optional representation
// of a Mesh, using the `meshhops::MeshView` interface.
// The main purpose is to cache result/ speed up operations in an interactive
// scenario.
//
// The vertex data is stored in a packed fashion on the device and
// therefore some precision compared to the fp32 storage in `MeshView` is lost.
// see `meshops::DeviceMeshVK` in the `meshops_vk.h` header.
//
// re-usable device representation of the mesh, so that
// in the very common scenario of remeshing an input mesh, that
// has no heightmaps, we can re-use the vbo, ibo, blas etc.
// in both remesh preparation (importance etc. tracing) and baker.
//
// All usages of `context` require device context support

typedef class DeviceMesh_c* DeviceMesh;

enum DeviceMeshUsageFlagBits : uint64_t
{
  DeviceMeshUsageBlasBit = 1ull << 0,
};
typedef uint64_t DeviceMeshUsageFlags;

struct DeviceMeshSettings
{
  DeviceMeshUsageFlags        usageFlags  = 0;
  meshops::MeshAttributeFlags attribFlags = 0;

  // Default values when not defined by MeshView.
  float directionBoundsBias  = 0.0f;
  float directionBoundsScale = 1.0f;
};

// creates new `DeviceMesh`
// `implictly runs meshopsDeviceMeshUpdate for the creation
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsDeviceMeshCreate(Context                  context,
                                                                   const meshops::MeshView& meshView,
                                                                   DeviceMeshSettings&      settings,
                                                                   DeviceMesh*              pDeviceMesh);

// updates the device buffers (must not be in-flight)
// based on `settings` and `meshView`.
// Uploads existing data, or leaves device buffers empty if
// `meshView` doesn't provide the content.
MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshUpdate(Context                  context,
                                                      DeviceMesh               deviceMesh,
                                                      const meshops::MeshView& meshView,
                                                      DeviceMeshSettings&      settings);

// reads back `deviceMesh` buffer content into `meshView`
// meshView must be properly sized
MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshReadback(Context context, DeviceMesh deviceMesh, meshops::MutableMeshView& meshView);

// reads back the `attributes` from `deviceMesh` buffer content into `meshView`
// meshView must be properly sized
MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshReadbackSpecific(Context                   context,
                                                                DeviceMesh                deviceMesh,
                                                                meshops::MutableMeshView& meshView,
                                                                DeviceMeshSettings        attributes);

// gets the current device mesh state showing which attributes and usages are currently available on the device
MESHOPS_API DeviceMeshSettings MESHOPS_CALL meshopsDeviceMeshGetSettings(Context context, DeviceMesh deviceMesh);


// destroys the `deviceMesh`
MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshDestroy(Context context, DeviceMesh deviceMesh);

// find extra details in `meshops_vk` header, how to access low-level vk buffers for rendering etc.

//////////////////////////////////////////////////////////////////////////

// `meshops::Texture` object represents 2D textures.
// The textures are either used as source or destination within
// other meshops operations. They may or may not exist
// as device resources, depending on their usage.
//
// All usages of `context` require device context support

typedef class Texture_c* Texture;

enum class TextureType : uint32_t
{
  eGeneric       = 0, /* 4-component RGBA color. Color space not specified. */
  eNormalMap     = 1, /* RGB components store 0.5 * (normal) + 0.5. */
  eQuaternionMap = 2, /* RGBA components store 0.5 * (quaternion) + 0.5. */
  eOffsetMap     = 3, /* RG components store 0.5 * (offset) + 0.5. */
  eHeightMap     = 4  /* R component stores displacement. */
};

enum TextureUsageFlagBit : uint64_t
{
  // used as input for the baker resampling
  eTextureUsageBakerResamplingSource = 1ull << 0,
  // used as output for the baker resampling
  eTextureUsageBakerResamplingDestination = 1ull << 1,
  // used as intermediate for the baker resampling
  // must be R32_sfloat storing closest distance per texel
  // must be initialized to float max, not zero
  eTextureUsageBakerResamplingDistance = 1ull << 2,
  // used as heightmap input for the baker
  // must be R32_sfloat
  eTextureUsageBakerHeightmapSource = 1ull << 3,
  // used as importance texture for the remesher
  // must be fetchable as R_sfloat
  eTextureUsageRemesherImportanceSource = 1ull << 4,
};

typedef uint64_t TextureUsageFlags;

struct TextureConfig
{
  uint32_t width  = 0;
  uint32_t height = 0;
  uint32_t mips   = 1;

  // this is an uncompressed format,
  // if a texture was created from file, then the format reported must not be BC compressed
  // but what the renderable destination format is, that way it's trivial to create
  // the appropriate resampled texture.

  micromesh::Format baseFormat = micromesh::Format::eUndefined;

  // actually used format
  // Any texture used as destination must be renderable, however
  // textures used as source may be compressed depending on what
  // usage flags they serve.
  uint32_t internalFormatVk = 0;
};

// create a new texture
// `clearColor` optional single value that the texture is filled with, only 32 bit values are legal
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreate(Context                         context,
                                                                TextureUsageFlags               usageFlags,
                                                                const TextureConfig&            config,
                                                                const micromesh::MicromapValue* clearColor,
                                                                Texture*                        pTexture);

// destroy texture
MESHOPS_API void MESHOPS_CALL meshopsTextureDestroy(Context context, Texture texture);

// retrieve basic information from texture
MESHOPS_API TextureConfig MESHOPS_CALL meshopsTextureGetConfig(Texture texture);

// copute byte size of mip, based on texture's `baseFormat`
MESHOPS_API size_t MESHOPS_CALL meshopsTextureGetMipDataSize(Texture texture, uint32_t mipLevel);


// The following functions describe a texture loader api

// on success writes to pHandle, the handles are passed into the other functions
// must be thread-safe
typedef std::function<micromesh::Result(const void* input, TextureConfig& config, void** pHandle, void* fnUserData)> FnTextureReadOpen;
// reads data into provided destination
// must be thread-safe
typedef std::function<micromesh::Result(void* handle, uint32_t mipLevel, size_t& outputSize, void* fnUserData)> FnTextureReadGetSize;
// reads data into provided destination
// must be thread-safe
typedef std::function<micromesh::Result(void* handle, uint32_t mipLevel, size_t size, void* destinationData, void* fnUserData)> FnTextureReadData;
// closes handle
// must be thread-safe
typedef std::function<void(void* handle, void* fnUserData)> FnTextureClose;

struct TextureDataLoader
{
  FnTextureReadOpen    fnOpen        = nullptr;
  FnTextureReadGetSize fnReadGetSize = nullptr;
  FnTextureReadData    fnReadData    = nullptr;
  FnTextureClose       fnClose       = nullptr;
  void*                fnUserData    = nullptr;
};

// loader api may be called in parallel and out of order to speed up loading of multiple textures.
// returns first non-success result
// `textures` will store the resulting texture objects (`count`-many elements)
// `textureInputs` provides the per-texture input to the loader's `fnOpen` function, typically filename (`count`-many elements)
// `textureUsageFlags` provides the per-texture usage flag (`count`-many elements)
// `results` will store the per-texture results (`count`-many elements)
// (typically file name)
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreateFromLoader(Context                  context,
                                                                          const TextureDataLoader* loader,
                                                                          size_t                   count,
                                                                          micromesh::Result*       results,
                                                                          Texture*                 textures,
                                                                          const TextureUsageFlags* textureUsageFlags,
                                                                          const void**             textureInputs);

// simplified loader for single uncompressed, mip 0 data only
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreateFromData(Context              context,
                                                                        TextureUsageFlags    usageFlags,
                                                                        const TextureConfig& config,
                                                                        size_t               dataSize,
                                                                        const void*          data,
                                                                        Texture*             pTexture);

// The following functions describe a texture saver api

// on success writes to pHandle, the handles are passed into the other functions
// `mipCount` return number of mip maps to retrieve
// must be thread-safe
typedef std::function<micromesh::Result(Texture texture, const void* input, uint32_t& mipCount, void** pHandle, void* fnUserData)> FnTextureWriteOpen;
// writes data into handle
// must be thread-safe
typedef std::function<micromesh::Result(void* handle, uint32_t mipLevel, size_t size, const void* textureData, void* fnUserData)> FnTextureWriteData;

struct TextureDataSaver
{
  FnTextureWriteOpen fnOpen      = nullptr;
  FnTextureWriteData fnWriteData = nullptr;
  FnTextureClose     fnClose     = nullptr;
  void*              fnUserData  = nullptr;
};

// saver api may be called in parallel and out of order to speed up saving of multiple textures.
// returns any non-success result
// `textures` provide the texture object (`count`-many elements)
// `textureInputs` provides the per-texture input to the savers's `fnOpen` function, typically filename (`count`-many elements)
// `results` will store the per-texture results (`count`-many elements)

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureToSaver(Context                 context,
                                                                 const TextureDataSaver* saver,
                                                                 size_t                  count,
                                                                 micromesh::Result*      results,
                                                                 const Texture*          textures,
                                                                 const void**            textureInputs);

// simple saver, first mip only
// `dataSize` must match `meshopsTextureGetMipDataSize(tex,0)`
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureToData(Context context, Texture tex, size_t dataSize, void* data);

struct Heightmap
{
  // Greyscale displacement values. Must be eR32_sfloat.
  Texture texture = nullptr;

  // Texture coordinate map index
  uint32_t textureCoord = 0;

  // Texture values are globally scaled by this value
  float scale = 1.0f;

  // Texture values are globally offset by this value in object space
  float bias = 0.0f;

  // If true, interpolated direction vectors will be normalized before being
  // used for displacement
  bool normalizeDirections = false;

  // Normals frequently have seams to make hard edges, which does not give
  // nice heightmap displacement results. Generating displacement direction
  // vectors is sometimes an improvement.
  bool usesVertexNormalsAsDirections = true;

  // Must be set to the maximum value in the mesh's triangleSubdivisionLevels
  // array.
  uint32_t maxSubdivLevel = 0xFFFFU;

  // Enables smoothed heightmap displacement using PN triangles [Vlachos et al. 2001].
  bool pnTriangles = false;
};

//////////////////////////////////////////////////////////////////////////
// Core Operations
// _do not_ require device context support
//////////////////////////////////////////////////////////////////////////

struct OpGenerateSubdivisionLevel_input
{
  uint32_t maxSubdivLevel = 0;

  bool useTextureArea = false;
  // if `useTextureArea == true` then set subdivlevel to somewhat match number of texels
  // and clamp by maxSubdivLevel.
  // if `useTextureArea == false` use longest edge of triangle and subdivide
  // according to the maximum edge length matching `maxSubdivLevel`

  // Manual adjustment of the factor choosing the subdivision
  float relativeWeight = 1.0f;

  // Manual adjustment of the output subdivision values when useTextureArea is
  // true. This is simply added to the result.
  int32_t subdivLevelBias = 0;

  uint32_t textureCoord  = 0;
  uint32_t textureWidth  = 0;
  uint32_t textureHeight = 0;

  // When !useTextureArea, the longest edge is normally computed automatically.
  // If this is non-zero it will be used instead, allowing the caller to choose
  // a maximum across multiple meshes. Note that this is in object space.
  float maxEdgeLengthOverride = 0.0f;
};

struct OpGenerateSubdivisionLevel_modified
{
  // modifies triangleSubdivisionLevel (must be properly sized)
  meshops::MutableMeshView meshView;

  uint32_t maxSubdivLevel = 0;
  uint32_t minSubdivLevel = 0;
};

// generate per-triangle subdivision levels for the target mesh based
// on texture coordinates or object space positions. See `OpGenerateSubdivisionLevel_input` for
// details
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateSubdivisionLevel(Context context,
                                                                             size_t  count,
                                                                             const OpGenerateSubdivisionLevel_input* inputs,
                                                                             OpGenerateSubdivisionLevel_modified* modifieds);

struct OpSanitizeSubdivisionLevel_input
{
  uint32_t                       maxSubdivLevel = 0;
  const micromesh::MeshTopology* meshTopology   = nullptr;
};

struct OpSanitizeSubdivisionLevel_modified
{
  // modifies triangleSubdivisionLevel (must be properly sized)
  meshops::MutableMeshView meshView;
  // updated after operation completed
  uint32_t minSubdivLevel = 0;
};

// alters per-triangle subdivision levels for the target mesh based
// on its mesh topology, so that one triangle's subdivision level can
// only have up to a difference of one level to its neighbors.
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpSanitizeSubdivisionLevel(Context context,
                                                                             size_t  count,
                                                                             const OpSanitizeSubdivisionLevel_input* inputs,
                                                                             OpSanitizeSubdivisionLevel_modified* modifieds);

struct OpBuildPrimitiveFlags_input
{
  const micromesh::MeshTopology* meshTopology = nullptr;
};

struct OpBuildPrimitiveFlags_modified
{
  // needs triangleSubdivisionLevel
  // modifies trianglePrimitiveFlags (must be properly sized)
  meshops::MutableMeshView meshView;
};

// build per triangle primitive flags that encode whether the current triangle
// has neighbors that have one subdivision level less. The encoding is per
// triangle edge, where the n-th bit in the flag is set if the n-th edge
// has a neighbor with such reduced subdivision level. The edges are
// {v0,v1},{v1,v2},{v2,v0}.
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpBuildPrimitiveFlags(Context                            context,
                                                                        size_t                             count,
                                                                        const OpBuildPrimitiveFlags_input* inputs,
                                                                        OpBuildPrimitiveFlags_modified*    modifieds);


struct OpReadSubdivisionLevel_input
{
  // pulls subdivision levels from bary
  const bary::BasicView* baryData = nullptr;
};

struct OpReadSubdivisionLevel_modified
{
  // modifies triangleSubdivisionLevel (must be properly sized)
  meshops::MutableMeshView meshView;
};

// fills the per triangle subdivision level from the micromap data supplied in the bary container
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpReadSubdivisionLevel(Context                             context,
                                                                         size_t                              count,
                                                                         const OpReadSubdivisionLevel_input* inputs,
                                                                         OpReadSubdivisionLevel_modified*    modifieds);

//////////////////////////////////////////////////////////////////////////
// Vertex Attributes

struct OpGenerateVertexDirections_input
{
  enum class Mode : uint32_t
  {
    eSmoothTriangleNormals,
  };

  // at the time of writing only smooth triangle directions can be generated
  Mode mode = Mode::eSmoothTriangleNormals;

  // how much to take the triangle area into account.
  // 0 means for a given vertex all triangle normals become averaged ignoring the area
  // of the triangle. 1 means the triangle area is used to compute the average,
  // so normal of large triangles have higher weight than others.
  float smoothTriangleAreaWeight = 1.0f;

  // a triangle index buffer that contains unique vertex positions
  // (the reguar triangle indices typically contain split vertices due to attributes
  //  like texcoords or normals)
  ArrayView<const micromesh::Vector_uint32_3> triangleUniqueVertexIndices;
};

struct OpGenerateVertexDirections_modified
{
  meshops::MutableMeshView meshView;
  // must be meshops::eMeshAttributeVertexNormalBit or meshops::eMeshAttributeVertexDirectionBit
  meshops::MeshAttributeFlagBits targetAttribute = meshops::eMeshAttributeVertexDirectionBit;
};

// generate the per vertex direction vectors for a mesh
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateVertexDirections(Context context,
                                                                             size_t  count,
                                                                             const OpGenerateVertexDirections_input* inputs,
                                                                             OpGenerateVertexDirections_modified* modifieds);

struct OpGenerateVertexTangentSpace_input
{
  // Tangent generation algorithm. Must not be eInvalid.
  TangentSpaceAlgorithm algorithm = TangentSpaceAlgorithm::eDefault;
};

struct OpGenerateVertexTangentSpace_modified
{
  // modifies vertexTangents/vertexBitangents (must be properly sized)
  meshops::MutableMeshView meshView;
};

// generate the per vertex tangent space for a mesh
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateVertexTangentSpace(Context context,
                                                                               size_t  count,
                                                                               const OpGenerateVertexTangentSpace_input* inputs,
                                                                               OpGenerateVertexTangentSpace_modified* modifieds);

//////////////////////////////////////////////////////////////////////////
// Tessellation

struct OpPreTessellate_input
{
  // pre-tessellates triangles so that afterwards,
  // the maximum subdivision level is the one provided here
  uint32_t maxSubdivLevel = 0;

  // must have triangleSubdivLevels
  // must have trianglePrimitiveFlags
  meshops::MeshView meshView;
};

struct OpPreTessellate_output
{
  // modifies everything, reallocates vertices etc.
  meshops::ResizableMeshView* meshView;
};

// pre-tessellate a mesh using the provided per triangle subdivision levels
// and primitive flags. Note pre-tessellation simply linearly interpolates all the vertex attributes,
// and therefore results in flat surfaces within the original input triangle. It is foremost
// meant to give the remesher some more triangles to play with, or to account for the fact
// that displaced micromeshes have an upper subdivision level of 5 in the current raytracing
// APIs, which could yield poor quality for a large input triangle.

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpPreTessellate(Context                      context,
                                                                  size_t                       count,
                                                                  const OpPreTessellate_input* inputs,
                                                                  OpPreTessellate_output*      outputs);

struct OpDisplacedTessellate_properties
{
  uint32_t maxHeightmapTessellateLevel;
};

struct OpDisplacedTessellate_input
{
  // the per-triangle subdivision levels are derived
  // either: baryDisplacement->triangles[meshTriangleIndex].subdivLevel
  // or:     baryTriangleIndex = meshView.triangleMappingIndex[meshTriangleIndex]
  //         baryDisplacement->triangles[baryTriangleIndex].subdivLevel

  // compressed or uncompressed displacement
  // required
  const bary::BasicView* baryDisplacement           = nullptr;
  uint32_t               baryDisplacementGroupIndex = 0;
  uint32_t               baryDisplacementMapOffset  = 0;
  // optional microvertex shading normal
  // must be eRG16_snorm octant encoding
  // must match subdivision level of displacement
  const bary::BasicView* baryNormal           = nullptr;
  uint32_t               baryNormalGroupIndex = 0;
  uint32_t               baryNormalMapOffset  = 0;

  // Optional heightmap displacement as an alternative to baryDisplacement.
  // meshView must have triangleSubdivisionLevels and trianglePrimitiveFlags.
  // These are intended to be generated with meshopsOpGenerateSubdivisionLevel,
  // setting useTextureArea to true. meshTopology is required for heightmap
  // tessellation.
  Heightmap                      heightmap;
  const micromesh::MeshTopology* meshTopology = nullptr;

  // must have vertexDirections
  // must have proper trianglePrimitiveFlags if required
  meshops::MeshView meshView;
};

struct OpDisplacedTessellate_output
{
  // modifies everything, reallocates vertices etc.
  meshops::ResizableMeshView* meshView;
};

MESHOPS_API void MESHOPS_CALL meshopsOpDisplacedGetProperties(Context context, OpDisplacedTessellate_properties& properties);

// tessellates an input mesh with a displacement micromap provided as bary container into
// a target mesh. Vertex attributes will be linearly interpolated, except for position
// and optionally shading normals can be provided as micromap as well.

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpDisplacedTessellate(Context                            context,
                                                                        size_t                             count,
                                                                        const OpDisplacedTessellate_input* inputs,
                                                                        OpDisplacedTessellate_output*      outputs);

//////////////////////////////////////////////////////////////////////////
// Mesh Topology

struct OpBuildTopology_input
{
  meshops::MeshView meshView;

  // optional, see `meshopsOpFindUniqueVertexIndices`
  // the topology's triangle vertex buffer will copy these if provided
  // otherwise generate them.
  ArrayView<const micromesh::Vector_uint32_3> triangleUniqueVertexIndices;
};

struct OpBuildTopology_output
{
  MeshTopologyData* meshTopology = nullptr;
};

// build the topology information for a mesh

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpBuildTopology(Context                      context,
                                                                  size_t                       count,
                                                                  const OpBuildTopology_input* inputs,
                                                                  OpBuildTopology_output*      outputs);

struct OpFindUniqueVertexIndices_input
{
  meshops::MeshView meshView;
};

struct OpFindUniqueVertexIndices_output
{
  ArrayView<micromesh::Vector_uint32_3> triangleUniqueVertexIndices;
};

// extract a triangle vertex index buffer with only unique vertices,
// which are found by matching raw float values.

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpFindUniqueVertexIndices(Context context,
                                                                            size_t  count,
                                                                            const OpFindUniqueVertexIndices_input* inputs,
                                                                            OpFindUniqueVertexIndices_output* outputs);

//////////////////////////////////////////////////////////////////////////
// Compression

struct OpCompressDisplacementMicromap_input
{
  micromesh::OpCompressDisplacement_settings settings;

  // values must be eR32_sfloat, eR8_unorm, eR16_unorm, or eR11_unorm_packed16
  const bary::BasicView* uncompressedDisplacement           = nullptr;
  uint32_t               uncompressedDisplacementGroupIndex = 0;

  // vertexDirections must be provided and vertexDirectionBounds may be used to
  // aid compression quality heuristic based on object space distance
  meshops::MeshView              meshView;
  const micromesh::MeshTopology* meshTopology = nullptr;
};

struct OpCompressDisplacementMicromap_output
{
  // mandatory, will be completely overwritten
  baryutils::BaryBasicData* compressedDisplacement = nullptr;
  // optional, if provided sets up uncompressed mips for typical rasterization use
  baryutils::BaryMiscData* compressedDisplacementRasterMips = nullptr;
};

// compress the provided displacement micromap for this mesh into a new bary container.
// The displacement subdivision levels must not exceed level 5.
// Optionally create meta information that speeds up rasterization.

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpCompressDisplacementMicromaps(Context context,
                                                                                  size_t  count,
                                                                                  const OpCompressDisplacementMicromap_input* inputs,
                                                                                  OpCompressDisplacementMicromap_output* outputs);

//////////////////////////////////////////////////////////////////////////
// Special Operations
// may require device context support as mention
//////////////////////////////////////////////////////////////////////////

// does require device
typedef class GenerateImportanceOperator_c* GenerateImportanceOperator;

// requires device context
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsGenerateImportanceOperatorCreate(Context context, GenerateImportanceOperator* pOp);
MESHOPS_API void MESHOPS_CALL meshopsGenerateImportanceOperatorDestroy(Context context, GenerateImportanceOperator op);

struct OpGenerateImportance_modified
{
  // expected to be filled with contents of input mesh
  // the operator will fill the vertexImportance field of
  // the meshView and the deviceMesh (if provided)
  meshops::MutableMeshView meshView;

  // texture coordinate for optional input vertex importance map
  uint32_t importanceTextureCoord = 0;
  // optional vertex importance map
  Texture importanceTexture = nullptr;

  // optional raytracing distance for curvature estimation
  // if the importance texture is not provided
  float rayTracingDistance = FLT_MAX;

  // optional power applied to the importance values
  float importancePower = 1.f;

  // optional input mesh object
  // if not passed, the operation will temporarily create and destroy one
  DeviceMesh deviceMesh = nullptr;
};


MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateImportance(Context                        context,
                                                                       GenerateImportanceOperator     op,
                                                                       size_t                         count,
                                                                       OpGenerateImportance_modified* inputs);


// does require device
typedef class RemeshingOperator_c* RemeshingOperator;

// requires device context
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsRemeshingOperatorCreate(Context context, RemeshingOperator* pOp);
MESHOPS_API void MESHOPS_CALL              meshopsRemeshingOperatorDestroy(Context context, RemeshingOperator op);

struct OpRemesh_input
{
  // maximum subdivision level generated during remeshing
  //
  // a triangle may not be further collapsed if its implicit
  // subdivision level reaches this limit.
  //
  // That is based on the greater of
  // either the heightmapTexture resolution of its area,
  // or     the number of source triangles that are represented by the output triangle

  uint32_t maxSubdivLevel = 0;

  // texture coordinate for optional height map, used to
  // limit decimation so the required subdivision level
  // for the final displaced geometry will not exceed
  // maxSubdivLevel
  uint32_t heightmapTextureCoord = 0;
  // size of the optional height map, in texels
  uint32_t heightmapTextureWidth  = 0;
  uint32_t heightmapTextureHeight = 0;

  // attributes the remesher must preserve,
  // such as texture coordinates, etc
  meshops::MeshAttributeFlags preservedVertexAttributeFlags = 0;

  // maximum error per edge incurred by the remesher
  float errorThreshold = 100.f;
  // maximum number of triangles after decimation
  // if nonzero, errorThreshold is ignored
  uint32_t maxOutputTriangleCount = 0;

  // multiplier of the vertex importance
  // in the error computation
  float importanceWeight = 200.f;

  //// texture coordinate for optional input curvature map
  //uint32_t importanceTextureCoord = 0;
  //// optional curvature map
  //Texture importanceTexture = nullptr;

  // maximum vertex valence yielded by the decimation
  uint32_t maxVertexValence = 20;

  // importance threshold [0,1] beyond which no decimation is allowed
  float importanceThreshold = 1.f;

  // generate micromesh information during remeshing:
  // subdivision levels, primitive flags, directions and displacement bounds
  bool generateMicromeshInfo = true;

  // if true the remeshing will stop after each iteration, requiring the
  // application to call meshopsOpRemesh multiple times. Each call returns
  // eContinue until all remeshing is finished, where meshopsOpRemesh returns eSuccess
  // This is typically useful for interactive previewing of the remeshing process.
  // Note the mesh data in the MeshView (Host side) will only be updated at the
  // end of the remeshing, and not during intermediate steps
  bool progressiveRemeshing = false;

  // Additional scale to the direction bounds to guarantee they contain the surface.
  float directionBoundsFactor = 1.02f;

  // If true the remesher may displace vertices along their displacement direction
  // to better fit the original surface
  bool fitToOriginalSurface = true;
};

struct OpRemesh_modified
{
  // expected to be filled with contents of input mesh
  //
  // remesher modifies everything, reallocates vertices etc.
  // The mesh must contain
  //  vertexNormals or vertexDirections
  //
  // If `generateMicromeshInfo` is true, also outputs
  //   triangleSubdivisionLevels
  //   vertexDirections
  //   vertexDirectionBounds
  meshops::ResizableMeshView* meshView;

  // optional output mesh object
  // if not passed, the operation will temporarily create and destroy one
  DeviceMesh deviceMesh = nullptr;
};

MESHOPS_API micromesh::Result MESHOPS_CALL
meshopsOpRemesh(Context context, RemeshingOperator op, size_t count, const OpRemesh_input* inputs, OpRemesh_modified* modifieds);

//////////////////////////////////////////////////////////////////////////

// `meshops::BakerOperator` handles ray-traced based baking
// of displacement and other micromap attributes. Also can
// do texture re-sampling.
//
// does require device context support

typedef class BakerOperator_c* BakerOperator;

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsBakeOperatorCreate(Context context, BakerOperator* pOp);
MESHOPS_API void MESHOPS_CALL              meshopsBakeOperatorDestroy(Context context, BakerOperator op);

typedef void (*PFN_meshopsDebugDisplacedReferenceMeshCallback)(const meshops::MeshView&           meshView,
                                                               const micromesh::Matrix_float_4x4* transform,
                                                               uint32_t                           batchIndex,
                                                               uint32_t                           batchTotal,
                                                               void*                              userPtr);

struct OpBake_properties
{
  uint32_t maxLevel;
  uint32_t maxResamplerTextures;
  uint32_t maxHeightmapTessellateLevel;
};

struct OpBake_requirements
{
  MeshAttributeFlags baseMeshAttribFlags;
  MeshAttributeFlags referenceMeshAttribFlags;
  bool               referenceMeshTopology;
};

struct OpBake_resamplerInput
{
  // Texture mapped to the reference mesh to sample from. Must be null for
  // generated texture types, i.e. eQuaternionMap, eOffsetMap and eHeightMap.
  Texture     texture     = nullptr;
  TextureType textureType = TextureType::eGeneric;

  // Floating point distance buffer, used to keep the closest samples from
  // tracing the reference mesh. This must be initialized to
  // std::numeric_limits<float>::max() as rays with longer distances will be
  // discarded.
  Texture distance = nullptr;

  // No MeshView support yet
  uint32_t textureCoord = 0;
};

typedef Heightmap OpBake_heightmap;

struct OpBake_settings
{
  // Output subdivision level to bake at. Each level produces 4x microtriangles.
  uint32_t level = 3;

  // If non-zero, overrides trace distance (in world space) otherwise defined by
  // baseMeshView.vertexDirections and baseMeshView.vertexDirectionBounds.
  float maxTraceLength = 0.0f;

  // Trace only in the direction of baseMeshView.vertexDirections if true.
  // Otherwise traces backwards too.
  bool uniDirectional = false;

  // If not null, called during baking for each batch of baking against the
  // reference mesh.
  PFN_meshopsDebugDisplacedReferenceMeshCallback debugDisplacedReferenceMeshCallback = nullptr;
  void*                                          debugDisplacedReferenceMeshUserPtr  = nullptr;

  bool fitDirectionBounds = false;

  // Rudimentary memory limit. Baking will be split into batches to maintain the
  // limit.
  uint64_t memLimitBytes = 4096ULL << 20;

  // Output displacement value layout
  bary::ValueLayout uncompressedLayout = bary::ValueLayout::eTriangleBirdCurve;

  // Output displacement value format
  bary::Format uncompressedDisplacementFormat = bary::Format::eR16_unorm;
  bary::Format uncompressedNormalFormat       = bary::Format::eRG16_snorm;

  // Factor applied to the maximum tracing distance, useful when the displacement bounds define a tight
  // shell around the original geometry, where floating-point approximations may create false misses.
  // A value of 1.02 typically provides satisfying results without resulting in performance/accuracy loss.
  float maxDistanceFactor = 1.0f;
};

struct OpBake_input
{
  OpBake_settings settings;

  meshops::MeshView baseMeshView;

  // Column-major object-to-world space transform
  micromesh::Matrix_float_4x4 baseMeshTransform;

  // Required if settings.fitDirectionBounds is true
  const micromesh::MeshTopology* baseMeshTopology;

  // May be the same as the base mesh
  meshops::MeshView referenceMeshView;

  // Column-major object-to-world space transform
  micromesh::Matrix_float_4x4 referenceMeshTransform;

  // only required for heightmaps
  const micromesh::MeshTopology* referenceMeshTopology = nullptr;

  // If populated with a texture, the reference mesh will be tessellated based
  // on its triangleSubdivisionLevels and triangleEdgeFlags, then displaced by
  // the heightmap values in the direction of its vertexNormals.
  OpBake_heightmap referenceMeshHeightmap;

  // Array of textures to resample. Resampling is required whenever the
  // displacement direction vectors to not exactly project UVs from the base to
  // the reference mesh. E.g. if the base and reference meshes are not the same,
  // or heightmap displacement is used and direction vectors do not match the
  // normals.
  ArrayView<OpBake_resamplerInput> resamplerInput;
};

struct OpBake_output
{
  // Output direction bounds. Must be allocated if settings.fitDirectionBounds
  // is true.
  ArrayView<nvmath::vec2f> vertexDirectionBounds;

  // Displacement values
  baryutils::BaryBasicData* uncompressedDisplacement = nullptr;
  baryutils::BaryBasicData* uncompressedNormal       = nullptr;

  ArrayView<Texture> resamplerTextures;
};

MESHOPS_API void MESHOPS_CALL meshopsBakeGetProperties(Context context, BakerOperator op, OpBake_properties& properties);

MESHOPS_API void MESHOPS_CALL meshopsBakeGetRequirements(Context                          context,
                                                         BakerOperator                    op,
                                                         const OpBake_settings&           settings,
                                                         ArrayView<OpBake_resamplerInput> resamplerInput,
                                                         bool                             uniformSubdivLevels,
                                                         bool                             referenceHasHeightmap,
                                                         bool                 heightmapUsesNormalsAsDirections,
                                                         OpBake_requirements& properties);

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpBake(Context context, BakerOperator op, const OpBake_input& input, OpBake_output& output);

//////////////////////////////////////////////////////////////////////////

#if 0
// TBD
// 
// `meshops::DisplacementMicromapOperator` deals with operations concerning
// the mesh and its displacement micromap data.
// It does not require a device context.
typedef class DisplacementOptimizationOperator_c* DisplacementOptimizationOperator;

MESHOPS_API micromesh::Result meshopsDisplacementMicromapOperatorCreate(Context context, DisplacementOptimizationOperator* pOp);
MESHOPS_API void meshopsDisplacementMicromapOperatorDestroy(Context context, DisplacementOptimizationOperator op);


struct OpOptimizeDisplacementMicromap_input
{
  float psnr          = 40.0;
  bool  validateEdges = false;

  const baryutils::BaryBasicData* uncompressedDisplacement = nullptr;
  const micromesh::MeshTopology*  meshTopology             = nullptr;
};

struct OpOptimizeDisplacementMicromap_output
{
  // subdivision levels may be altered through trimming
  meshops::ResizableMeshView meshView;
  // mandatory, will be completely overwritten
  baryutils::BaryBasicData* compressedDisplacement = nullptr;
  // optional sets up uncompressed mips for typical rasterization use
  baryutils::BaryMiscData* compressedDisplacementRasterMips = nullptr;
};


MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpOptimizeDisplacementMicromaps(Context context,
                                                                                  DisplacementOptimizationOperator op,
                                                                                  size_t count,
                                                                                  const OpOptimizeDisplacementMicromap_input* inputs,
                                                                                  OpOptimizeDisplacementMicromap_output* outputs);
#endif


}  // namespace meshops
