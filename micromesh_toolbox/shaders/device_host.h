#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvvkhl/shaders/dh_lighting.h"

#ifdef __cplusplus
#include <nvmath/nvmath.h>
namespace shaders {
using namespace nvvkhl_shaders;
using mat4 = nvmath::mat4f;
using vec4 = nvmath::vec4f;
using vec3 = nvmath::vec3f;
using vec2 = nvmath::vec2f;
#endif  // __cplusplus

struct PushConstant
{
  int  frame;            // For RTX
  int  maxDepth;         // For RTX
  int  maxSamples;       // For RTX
  int  materialID;       // For raster
  int  instanceID;       // For raster
  int  primMeshID;       // For raster
  int  baryInfoID;       // For micromesh
  int  microMax;         // For micromesh raster, different for basetri/subtri
  vec2 microScaleBias;   // For micromesh raster
  int  triangleCount;    // Valid for heightmaps only
  int  bakeSubdivLevel;  // ToolBox-global target subdivision level for baking
};


#define MAX_NB_LIGHTS 1
#define WORKGROUP_SIZE 16
#define HEIGHTMAP_MAX_SUBDIV_LEVEL 11

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
#define START_ENUM(a) enum a {
#define END_ENUM() }
#define INLINE inline
#else
#define START_ENUM(a)  const uint
#define END_ENUM()
#define INLINE
#endif


START_ENUM(DebugMethod)
eDbgMethod_none      = 0,
eDbgMethod_metallic  = 1,
eDbgMethod_roughness = 2,
eDbgMethod_normal    = 3,
eDbgMethod_basecolor = 4,
eDbgMethod_emissive  = 5,
eDbgMethod_txtcoord  = 6,
eDbgMethod_direction = 7
END_ENUM();


START_ENUM(RenderShading)
eRenderShading_default = 0,
eRenderShading_faceted = 1,
eRenderShading_phong = 2,
//
eRenderShading_anisotropy = 3,
//
eRenderShading_minMax = 4,
eRenderShading_subdivLevel = 5,
eRenderShading_baseTriangleIndex = 6,
eRenderShading_compressionFormat = 7,
eRenderShading_heightmapTexelFrequency = 8,
eRenderShading_opposingDirections = 9,
eRenderShading_sharedPositions = 10
END_ENUM();

// clang-format on


// The frame buffer, is a buffer that is updated at each frame.
// The information should be typically things that is changing
// often, like controlled by the UI
struct FrameInfo
{
  mat4  proj;                  // Camera projection matrix
  mat4  view;                  // Camera model view matrix
  mat4  projInv;               // Inverse of the projection
  mat4  viewInv;               // Inverse of the model view
  Light light[MAX_NB_LIGHTS];  // Support for multiple lights
  vec4  envColor;              // Environment color multiplier
  vec2  resolution;            // Size of the framebuffer in pixels
  int   useSky;                // Using sky of the HDR
  int   nbLights;              // Number of light used
  float envRotation;           // Rotation of the environment (around Y)
  float maxLuminance;          // For fireflies, cutoff the white pixels
  float metallic;              // Overriding the metallic with Flat shading
  float roughness;             // Overriding the roughness with Flat shading
  int   colormap;              // Choice of the color map (temperature, vivid, ...)
  int   overlayColor;          // Color RGBA8 of the wireframe overlay
  vec2  mouseCoord;            // Mouse coordinate when pressing down, else (-1,-1)
  float vectorLength;          // Visualization of normal and direction vectors
  int   heightmapSubdivLevel;  // Max. mesh shader heightmap tessellation
  float heightmapScale;        // Additional UI-exposed heightmap scale
  float heightmapOffset;       // Additional UI-exposed heightmap offset
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // HOST_DEVICE_H
