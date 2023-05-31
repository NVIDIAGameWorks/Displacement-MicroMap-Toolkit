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

#include "meshops/meshops_operations.h"
#include "tool_image.hpp"
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>  // Simplex noise
#include <gltf.hpp>
#include <gltf/micromesh_util.hpp>
#include <imageio/imageio.hpp>
#include <inputparser.hpp>
#include <memory>
#include <meshops/meshops_mesh_view.h>
#include <micromesh/micromesh_types.h>
#include <nvmath/nvmath.h>
#include <nvmath/nvmath_types.h>
#include <string>
#include <tool_generate.hpp>
#include <tool_meshops_objects.hpp>
#include <tool_scene.hpp>
#include <meshops/bias_scale.hpp>

// TODO: use public meshops API for attrib generation
#include <meshops_internal/umesh_util.hpp>

namespace tool_generate {

namespace fs = std::filesystem;

using ToolImageVector = std::vector<std::unique_ptr<micromesh_tool::ToolImage>>;

bool toolGenerateParse(int argc, char** argv, ToolGenerateArgs& args)
{
  bool              printHelp = false;
  std::string       geometry;
  CommandLineParser parser(
      "generate: Creates test meshes with textures. Use displacedtessellate to create real geometry from meshes with "
      "heightmaps.");
  parser.addArgument({"--help"}, &printHelp, "Print Help");
  parser.addArgument({"--geometry"}, &geometry,
                     "Kind of geometry to generate: plane, cube, terrain, sphere or rock. Default is cube");
  parser.addArgument({"--resolution"}, &args.resolution,
                     "Resolution in tessellation, texels or 3D cells, depending on --geometry. Should be a power of 2. "
                     "Default: 128.");

  if(!parser.parse(argc, argv) || printHelp)
  {
    parser.printHelp();
    return false;
  }

  if(args.resolution == 0)
  {
    LOGE("Error: --resolution must be positive.\n");
    return false;
  }

  if(geometry == "plane")
  {
    args.geometry = ToolGenerateArgs::GEOMETRY_PLANE;
  }
  else if(geometry == "cube")
  {
    args.geometry = ToolGenerateArgs::GEOMETRY_CUBE;
  }
  else if(geometry == "terrain")
  {
    args.geometry = ToolGenerateArgs::GEOMETRY_TERRAIN;
  }
  else if(geometry == "sphere")
  {
    args.geometry = ToolGenerateArgs::GEOMETRY_SPHERE;
  }
  else if(geometry == "rock")
  {
    args.geometry = ToolGenerateArgs::GEOMETRY_ROCK;
  }
  else if(!geometry.empty())
  {
    LOGE("Error: unknown --geometry '%s'.\n", geometry.c_str())
    return false;
  }

  return true;
}

static int defaultMaterial(tinygltf::Model& output)
{
  if(output.materials.empty())
  {
    tinygltf::Material material;
    material.pbrMetallicRoughness.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    material.doubleSided                          = true;
    output.materials.push_back(std::move(material));
  }
  return 0;
}

static void createGltfMesh(tinygltf::Model& output, const meshops::MeshView& meshView)
{
  // Single material
  int materialID = defaultMaterial(output);

  // Append mesh data to buffers, create buffer views and accessors
  tinygltf::Primitive primitive = tinygltfAppendPrimitive(output, meshView);
  primitive.material            = materialID;

  // Add the primitive to a mesh
  tinygltf::Mesh mesh;
  mesh.primitives.push_back(std::move(primitive));
  size_t meshId = output.meshes.size();
  output.meshes.push_back(std::move(mesh));

  // Instantiate the mesh
  tinygltf::Node node;
  node.mesh     = static_cast<int>(meshId);
  size_t nodeId = output.nodes.size();
  output.nodes.push_back(std::move(node));

  // Add the instance to a scene
  tinygltf::Scene scene;
  scene.nodes.push_back(static_cast<int>(nodeId));
  output.scenes.push_back(std::move(scene));

  // Metadata
  output.asset.copyright = "NVIDIA Corporation";
  output.asset.generator = "micromesh_tool";
  output.asset.version   = "2.0";  // glTF version 2.0
}

static int createImage(int resolution, size_t components, size_t componentBitDepth, std::string filename, ToolImageVector& images)
{
  int                             imageIndex = static_cast<int>(images.size());
  micromesh_tool::ToolImage::Info normalImageInfo;
  normalImageInfo.width = normalImageInfo.height = static_cast<size_t>(resolution);
  normalImageInfo.components                     = components;
  normalImageInfo.componentBitDepth              = componentBitDepth;
  images.push_back(micromesh_tool::ToolImage::create(normalImageInfo, filename));
  return imageIndex;
}

static meshops::ArrayView<uint32_t> createNormalMap(int resolution, std::string filename, tinygltf::Model& output, ToolImageVector& images)
{
  int imageIndex = createImage(resolution, 4, 8, filename, images);

  // Reference the normal map in the gltf
  int materialID = defaultMaterial(output);
  output.textures.emplace_back();
  output.textures.back().source                    = imageIndex;
  output.materials[materialID].normalTexture.index = 1;

  // avoid validation warnings from things checking the gltf before saving
  output.images.emplace_back();

  auto result = images[imageIndex]->array<uint32_t>();
  std::fill(result.begin(), result.end(), 0x000000ffu);
  return result;
}

static meshops::ArrayView<uint16_t> createHeightMap(int              resolution,
                                                    std::string      filename,
                                                    BiasScalef       biasScale,
                                                    tinygltf::Model& output,
                                                    ToolImageVector& images)
{
  int imageIndex = createImage(resolution, 1, 16, filename, images);

  // Add a texture
  int materialID = defaultMaterial(output);
  output.textures.emplace_back();
  output.textures.back().source = imageIndex;

  // avoid validation warnings from things checking the gltf before saving
  output.images.emplace_back();

  // Reference the heightmap in the gltf
  nvh::KHR_materials_displacement displacement;
  displacement.displacementGeometryTexture = 0;
  displacement.displacementGeometryFactor  = biasScale.scale;
  displacement.displacementGeometryOffset  = biasScale.bias;
  setMaterialsDisplacement(displacement, output, output.materials[materialID]);

  auto result = images[imageIndex]->array<uint16_t>();
  std::fill(result.begin(), result.end(), 0x0000u);
  return result;
}

bool generateSmoothNormals(micromesh_tool::ToolContext& context, meshops::MeshData& meshData)
{
  {
    meshData.vertexDirections.resize(meshData.vertexCount());
    meshops::MutableMeshView  meshView(meshData);
    meshops::MeshTopologyData topology;
    const micromesh::Result result = micromesh_tool::buildTopologyData(context.meshopsContext(), meshView, topology);
    if(micromesh::Result::eSuccess != result)
    {
      LOGE("Error: Failed to build topology data.\n");
      return false;
    }
    makeDisplacementDirections(meshView, *topology, meshView.vertexDirections, eNormalReduceNormalizedLinear);
    std::swap(meshData.vertexNormals, meshData.vertexDirections);
    meshData.vertexDirections = {};

#if 0
    // Replace original triangles with watertight triangles (leaving a few
    // unreferenced vertices). Currently the remesher must be given completely
    // watertight topology. This breaks UVs though.
    auto watertightTriangleVertices = meshops::ArrayView<nvmath::vec3ui>(meshops::ArrayView(topology.triangleVertices));
    std::copy(watertightTriangleVertices.begin(), watertightTriangleVertices.end(), meshData.triangleVertices.begin());
#endif
  }

  return true;
}

bool generateTangents(micromesh_tool::ToolContext& context, meshops::MeshData& meshData)
{
  meshData.vertexTangents.resize(meshData.vertexCount());
  meshops::MutableMeshView                       meshView(meshData);
  meshops::OpGenerateVertexTangentSpace_input    input{meshops::TangentSpaceAlgorithm::eMikkTSpace};
  meshops::OpGenerateVertexTangentSpace_modified modifieds{meshView};
  micromesh::Result result = meshops::meshopsOpGenerateVertexTangentSpace(context.meshopsContext(), 1, &input, &modifieds);
  if(result != micromesh::Result::eSuccess)
  {
    LOGE("Error: Failed to generate vertex tangents\n");
    return false;
  }

  return true;
}

// Creates a tessellated plane, covering the [-1, 1] range in the XZ plane.
// The plane is tessellated with tesellateQuads() to the given
// heightmap resolution; 1 or 2 produces no tessellation.
static bool generatePlane(uint32_t resolution, meshops::MeshData& meshData)
{
  // 0           1
  // +-----------+> u+
  // | \         |
  // |   \       |
  // |     +------> X+
  // |     | \   |
  // |     |   \ |
  // +-----|-----+
  // 3     v     2
  // v+    Z+
  meshData.triangleVertices = {{0, 3, 2}, {0, 2, 1}};
  meshData.vertexPositions  = {
       {-1, 0, -1},
       {1, 0, -1},
       {1, 0, 1},
       {-1, 0, 1},
  };
  meshData.vertexNormals = {
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
      {0, 1, 0},
  };
  meshData.vertexTexcoords0 = {
      {0, 0},
      {1, 0},
      {1, 1},
      {0, 1},
  };

  if(resolution > 2)
  {
    meshops::DynamicMeshSetView dynamicMeshView;
    dynamicMeshView.flat   = meshData;
    dynamicMeshView.slices = {meshops::MeshSlice(meshData.triangleCount(), meshData.vertexCount())};
    if(!tessellateQuads(0, {{resolution, resolution}}, dynamicMeshView))
    {
      LOGE("Error: plane tessellation failed\n");
      return false;
    }
  }
  return true;
}

static bool generatePlane(uint32_t resolution, tinygltf::Model& output)
{
  meshops::MeshData meshData;
  generatePlane(resolution, meshData);
  createGltfMesh(output, meshops::MeshView(meshData));
  return true;
}

// Creates a cube. Faces are not welded together, so there are 24 vertices.
static bool generateCube(meshops::MeshData& meshData, int resolution)
{
  meshData.triangleVertices = {
      {0, 1, 2},    {1, 3, 2},    {4, 5, 6},    {5, 7, 6},    {8, 9, 10},   {9, 11, 10},
      {12, 13, 14}, {13, 15, 14}, {16, 17, 18}, {17, 19, 18}, {20, 21, 22}, {21, 23, 22},
  };
  meshData.vertexPositions  = {{1.0, 1.0, -1.0},   {-1.0, 1.0, -1.0}, {1.0, 1.0, 1.0},    {-1.0, 1.0, 1.0},
                               {1.0, -1.0, 1.0},   {1.0, 1.0, 1.0},   {-1.0, -1.0, 1.0},  {-1.0, 1.0, 1.0},
                               {-1.0, -1.0, 1.0},  {-1.0, 1.0, 1.0},  {-1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0},
                               {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, -1.0, 1.0},  {1.0, -1.0, 1.0},
                               {1.0, -1.0, -1.0},  {1.0, 1.0, -1.0},  {1.0, -1.0, 1.0},   {1.0, 1.0, 1.0},
                               {-1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {1.0, -1.0, -1.0},  {1.0, 1.0, -1.0}};
  meshData.vertexNormals    = {{0.0, 1.0, 0.0},  {0.0, 1.0, 0.0},  {0.0, 1.0, 0.0},  {0.0, 1.0, 0.0},  {0.0, 0.0, 1.0},
                               {0.0, 0.0, 1.0},  {0.0, 0.0, 1.0},  {0.0, 0.0, 1.0},  {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
                               {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, -1.0, 0.0},
                               {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0},  {1.0, 0.0, 0.0},  {1.0, 0.0, 0.0},  {1.0, 0.0, 0.0},
                               {0.0, 0.0, -1.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, -1.0}, {0.0, 0.0, -1.0}};
  meshData.vertexTexcoords0 = {{0.625, 0.5},  {0.875, 0.5},  {0.625, 0.25}, {0.875, 0.25}, {0.375, 0.25}, {0.625, 0.25},
                               {0.375, 0.0},  {0.625, 0.0},  {0.375, 1.0},  {0.625, 1.0},  {0.375, 0.75}, {0.625, 0.75},
                               {0.125, 0.5},  {0.375, 0.5},  {0.125, 0.25}, {0.375, 0.25}, {0.375, 0.5},  {0.625, 0.5},
                               {0.375, 0.25}, {0.625, 0.25}, {0.375, 0.75}, {0.625, 0.75}, {0.375, 0.5},  {0.625, 0.5}};

  // Attempt to place texture coordinates on pixel centers. Without this, the
  // displaced cubes have clearly visible creases. Start with adding a half
  // pixel margin. This fixes image border values.
  float      resolutionf = static_cast<float>(resolution);
  BiasScalef transform(0.5f / resolutionf, (resolutionf - 1.0f) / resolutionf);
  for(nvmath::vec2f& coord : meshData.vertexTexcoords0)
  {
    coord = transform * coord;

    // Snap coordinates to the nearest pixel center so we have height values
    // exactly on axis aligned edges. This fixes interior/shared edge heights.
    coord.x = (std::round(coord.x * resolutionf - 0.5f) + 0.5f) / resolutionf;
    coord.y = (std::round(coord.y * resolutionf - 0.5f) + 0.5f) / resolutionf;
  }
  return true;
}

static bool generateCube(tinygltf::Model& output)
{
  meshops::MeshData meshData;
  if(!generateCube(meshData, 1024))
  {
    return false;
  }
  createGltfMesh(output, meshops::MeshView(meshData));
  return true;
}

static uint32_t ceillog2(uint32_t x)
{
  uint32_t orig   = x;
  uint32_t result = 0;  // bitlength - 1
  while(x > 1U)
  {
    x >>= 1;
    ++result;
  }
  if(orig != (1U << result) && result != 32)
  {
    ++result;
  }
  return result;
}

// Based on code from: https://github.com/imneme/pcg-c
/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014-2019 Melissa O'Neill <oneill@pcg-random.org>,
 *                     and the PCG Project contributors.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 * Licensed under the Apache License, Version 2.0 (provided in
 * LICENSE-APACHE.txt and at http://www.apache.org/licenses/LICENSE-2.0)
 * or under the MIT license (provided in LICENSE-MIT.txt and at
 * http://opensource.org/licenses/MIT), at your option. This file may not
 * be copied, modified, or distributed except according to those terms.
 *
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, either
 * express or implied.  See your chosen license for details.
 *
 * For additional information about the PCG random number generation scheme,
 * visit http://www.pcg-random.org/.
 */
static float pcg_noise3D(uint32_t x, uint32_t y, uint32_t z)
{
  uint32_t n = (x << 20) + (y << 10) + z;
  // Perform one iteration of pcg32i_random_t
  n = n * 747796405u + 1u;
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  n = ((n >> ((n >> 28) + 4u)) ^ n) * 277803737u;
  n = (n >> 22) ^ n;
  return float(n) / 4294967295.0f;
}

static float cosInterp(float y1, float y2, float mu)
{
  float              mu2 = (1 - cosf(mu * nv_pi)) / 2;
  return (y1 * (1 - mu2) + y2 * mu2);
}

static float smoothNoise3D(nvmath::vec3f p)
{
  nvmath::vec3i pI(nvmath::nv_floor(p));
  nvmath::vec3f pF(p - nvmath::nv_floor(p));
  float         n000 = pcg_noise3D(pI.x, pI.y, pI.z);
  float         n010 = pcg_noise3D(pI.x, pI.y + 1, pI.z);
  float         n100 = pcg_noise3D(pI.x + 1, pI.y, pI.z);
  float         n110 = pcg_noise3D(pI.x + 1, pI.y + 1, pI.z);
  float         n001 = pcg_noise3D(pI.x, pI.y, pI.z + 1);
  float         n011 = pcg_noise3D(pI.x, pI.y + 1, pI.z + 1);
  float         n101 = pcg_noise3D(pI.x + 1, pI.y, pI.z + 1);
  float         n111 = pcg_noise3D(pI.x + 1, pI.y + 1, pI.z + 1);
  float         n00  = cosInterp(n000, n001, pF.z);
  float         n01  = cosInterp(n010, n011, pF.z);
  float         n10  = cosInterp(n100, n101, pF.z);
  float         n11  = cosInterp(n110, n111, pF.z);
  float         n0   = cosInterp(n00, n01, pF.y);
  float         n1   = cosInterp(n10, n11, pF.y);
  return cosInterp(n0, n1, pF.x);
}

// Computes tiling fractal Brownian motion noise in 2D.
// Returns values in the range -1.0f ... 1.0f.
// Sums octaves over wavelengths from `resolution` texels to 0.5-1.0 texels.
static float fbm2D(nvmath::vec2f p, int resolution)
{
  const float gain       = 0.5f;
  int         octaves    = ceillog2(resolution);
  float       total      = 0.0f;
  float       frequency  = 1.0f / (float)resolution;
  float       amplitude  = gain;
  float       lacunarity = 2.0;
  for(int i = 0; i < octaves; ++i)
  {
    total += glm::perlin(glm::vec2(p.x, p.y) * frequency, glm::vec2(float(1 << i))) * amplitude;
    frequency *= lacunarity;
    amplitude *= gain;
  }

  return total;
}

// Computes fractal Brownian motion noise in 3D.
// Returns values in the range -1.0f ... 1.0f.
// Sums octaves over wavelengths from `resolution` texels to 0.5-1.0 texels.
static float fbm3D(nvmath::vec3f p, int resolution)
{
  const float gain       = 0.5f;
  int         octaves    = ceillog2(resolution);
  float       total      = 0.0f;
  float       frequency  = 1.0f / (float)resolution;
  float       amplitude  = gain;
  float       lacunarity = 2.0;
  for(int i = 0; i < octaves; ++i)
  {
    total += smoothNoise3D(nvmath::vec3f(p.x, p.y, p.z) * frequency) * amplitude;
    frequency *= lacunarity;
    amplitude *= gain;
  }

  return total;
}

static uint16_t floatToUnorm16(float x)
{
  return uint16_t(std::min(std::max(0.f, x), 1.0f) * 65535.0f + 0.5f);
}

static uint32_t vec3fToUnormRGBX8(nvmath::vec3f v)
{
  uint32_t x(static_cast<uint32_t>(std::min(std::max(0.0f, v.x), 1.0f) * 255.0f + 0.5f));
  uint32_t y(static_cast<uint32_t>(std::min(std::max(0.0f, v.y), 1.0f) * 255.0f + 0.5f));
  uint32_t z(static_cast<uint32_t>(std::min(std::max(0.0f, v.z), 1.0f) * 255.0f + 0.5f));
  return (x << 0) | (y << 8) | (z << 16) | (0xffu << 24);
}

static bool generateTerrain(micromesh_tool::ToolContext& context, uint32_t resolution, tinygltf::Model& output, ToolImageVector& images)
{
  meshops::MeshData meshData;
  generatePlane(2, meshData);

  // Generate tangents for the normal map
  if(!generateTangents(context, meshData))
  {
    return false;
  }

  createGltfMesh(output, meshops::MeshView(meshData));

  // Generate a heightmap from perlin noise
  auto heights = createHeightMap(resolution, "terrain_height.png", BiasScalef{}, output, images);
  for(uint32_t y = 0; y < resolution; ++y)
  {
    for(uint32_t x = 0; x < resolution; ++x)
    {
      heights[y * resolution + x] = floatToUnorm16(fbm2D(nvmath::vec2f{x, y} + 0.5f, resolution) + 0.5f);
    }
  }


  // Generate a normal map based on the sobel filter of the heights
  auto normals   = createNormalMap(resolution, "terrain_normal.png", output, images);
  auto addSample = [&heights, &resolution](nvmath::vec3f& sample, int x, int y, float weight) {
    nvmath::vec2f uv = (nvmath::vec2f{x, y} + 0.5f) / float(resolution);
    // Wrap the pixel coordinate. It's never less than -1.
    x = (x + resolution) % resolution;
    y = (y + resolution) % resolution;
    // Hard code the plane's x and z coordinates, which are in the range [-1, 1]
    nvmath::vec3f pos = {uv.x * 2.0f - 1.0f, heights[y * resolution + x] / 65535.0f, uv.y * 2.0f - 1.0f};
    sample += pos * weight;
  };
  for(int y = 0; y < static_cast<int>(resolution); ++y)
  {
    for(int x = 0; x < static_cast<int>(resolution); ++x)
    {
      // Sobel filter, finding the surface position derivative with respect to
      // pixel position.
      nvmath::vec3f gx;
      addSample(gx, x - 1, y - 1, -1.0f);
      addSample(gx, x - 1, y + 0, -2.0f);
      addSample(gx, x - 1, y + 1, -1.0f);
      addSample(gx, x + 1, y - 1, 1.0f);
      addSample(gx, x + 1, y + 0, 2.0f);
      addSample(gx, x + 1, y + 1, 1.0f);
      nvmath::vec3f gy;
      addSample(gy, x - 1, y - 1, -1.0f);
      addSample(gy, x + 0, y - 1, -2.0f);
      addSample(gy, x + 1, y - 1, -1.0f);
      addSample(gy, x - 1, y + 1, 1.0f);
      addSample(gy, x + 0, y + 1, 2.0f);
      addSample(gy, x + 1, y + 1, 1.0f);
      // Invert gy as eMikkTSpace aligns the bitangent with -UV.y
      nvmath::vec3f osNormal = nvmath::normalize(nvmath::cross(gx, -gy));
      // Hard coded TBN^-1 transform. The quad's +X is UV.x, +Y is up (surface
      // normal), +Z is +UV.y so it needs inverting to match the bitangent.
      nvmath::vec3f tsNormal      = {osNormal.x, -osNormal.z, osNormal.y};
      normals[y * resolution + x] = vec3fToUnormRGBX8(tsNormal * 0.5f + 0.5f);
    }
  }

  return true;
}

// Computes barycentric coordinates given a point `p` on a 2D triangle ABC.
// That is, returns {u, v, w} such that u + v + w = 1 and a*u + b*v + c*w = p.
static nvmath::vec3f triangleBaryCoord(const nvmath::vec2f& p, const nvmath::vec2f& a, const nvmath::vec2f& b, const nvmath::vec2f& c)
{
  // Substituting u = 1 - v - w into the second equation, we get
  // v0 * v + v1 * w = v2, where
  nvmath::vec2f v0 = b - a;
  nvmath::vec2f v1 = c - a;
  nvmath::vec2f v2 = p - a;
  // This is a 2x2 linear system:
  // [ |  |  ] [u]   [| ]
  // [ v0 v1 ] [ ] = [v2]
  // [ |  |  ] [v]   [| ]
  // which we solve by inverting the matrix and multiplying by v2:
  float den = v0.x * v1.y - v1.x * v0.y;
  float v   = (v2.x * v1.y - v1.x * v2.y) / den;
  float w   = (v0.x * v2.y - v2.x * v0.y) / den;
  return {1.0f - v - w, v, w};
}

// Returns positive values when `p` is inside the sphere circumscribing a centered 2x2x2 cube.
static float insideSphere(const nvmath::vec3f& p)
{
  return 3.0f - nvmath::dot(p, p);
}

static float cut(nvmath::vec3f p, nvmath::vec3f plane, float a, float b, float c)
{
  // Use c to instead shift the quadratic minimum to x=0 so it can't fold back
  // on itself
  float x = std::max(0.0f, nvmath::dot(p, plane) + c);
  return a * std::min(4.0f, x * x) + b * x;
}

static float insideRock(const nvmath::vec3f& p, int resolution)
{
  // Start with the distance from p to the surface of a sphere around the
  // 2x2x2 cube. This is the opposite of an SDF; this initial distance is
  // negative outside and positive inside.
  float n = 2.0f - nvmath::length(p);
  // Add random noise. The surface is no longer an SDF.
  n += fbm3D(p * float(resolution), resolution) * 2.0f;
  // Hollow inwards using a set of randomly chosen cutting planes. For each
  // plane, if p is further away from the origin, we make it more likely p is
  // outside the rock. This value is peturbed through a quadratic polynomial to
  // remove sharp edges from some cuts.
  n -= cut(p, {0.5f, -3.0f, -1.0f}, 0.0f, 4.0f, -1.5f);
  n -= cut(p, {1.0f, 0.0f, 1.0f}, 1.0f, 0.1f, -1.0f);
  n -= cut(p, {-0.5f, -1.0f, 2.0f}, 1.0f, 0.1f, -1.5f);
  n -= cut(p, {0.0f, 2.0f, 1.0f}, 1.0f, 3.0f, -1.5f);
  n -= cut(p, {-2.0f, 0.0f, -1.0f}, 0.0f, 1.0f, -1.0f);
  n -= cut(p, {1.0f, 2.0f, -2.0f}, 0.5f, 0.5f, -2.0f);
  return n;
}

// Finds a point on the isosurface (the rock if `rock` is set, the
// sphere otherwise) along the line between p0 and p1. p0 should be inside the
// isosurface; otherwise, there's a chance no intersection will be found.
static nvmath::vec3f intersectIsosurface(nvmath::vec3f p0, nvmath::vec3f p1, bool rock, int resolution)
{
  // Secant search with early termination
  float x0 = 0.25f;
  float x1 = 0.75f;
  float f0 = rock ? insideRock(nvmath::mix(p0, p1, x0), resolution) : insideSphere(nvmath::mix(p0, p1, x0));
  for(int i = 0; i < 16; ++i)
  {
    float f1 = rock ? insideRock(nvmath::mix(p0, p1, x1), resolution) : insideSphere(nvmath::mix(p0, p1, x1));
    float x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
    x0       = x1;
    f0       = f1;
    x1       = x2;

    // Stop after reaching 16 bit fixed point precision
    if(std::abs(x1 - x0) < 1.0f / float(1 << 16))
    {
      break;
    }
  }
  return nvmath::mix(p0, p1, x1);
}

// Generates a cube base mesh, plus a displacement map that turns it into
// either a sphere (if `!rock`) or a bumpy polyhedral rock (if `rock`).
static bool generateDisplacedCube(micromesh_tool::ToolContext& context, uint32_t resolution, bool rock, tinygltf::Model& output, ToolImageVector& images)
{
  meshops::MeshData meshData;
  if(!generateCube(meshData, resolution))
  {
    return false;
  }

  // Generate smooth normals for the cube
  if(!generateSmoothNormals(context, meshData))
  {
    return false;
  }

  // Generate tangents for the normal map
  if(!generateTangents(context, meshData))
  {
    return false;
  }

  // Define the heightmap and search displacement bounds:
  // from pos + bias * normal, to pos + (bias + scale) * normal.
  BiasScalef biasScale(-1.0, 3.0f);

  // Allow sampling a little outside triangles to avoid UVs interpolating to
  // empty data. This is in barycentric coordinate units, but ideally this
  // would be in units of pixels.
  const float baryExtrapolate = -0.1f;

  std::string texturePrefix(rock ? "rock" : "sphere");

  // Shoot rays from texels and set the displacement to match an intersection
  // with the isosurface. Criminally inefficient, but just for testing.
  auto heights = createHeightMap(resolution, texturePrefix + "_height.png", biasScale, output, images);
  std::vector<nvmath::vec3f> pixelPositions(resolution * resolution, nvmath::vec3f(NAN));
  std::vector<nvmath::mat3f> tbnInv(resolution * resolution, nvmath::mat3f(1));
  for(uint32_t y = 0; y < resolution; ++y)
  {
    for(uint32_t x = 0; x < resolution; ++x)
    {
      nvmath::vec2f uv   = (nvmath::vec2f{x, y} + 0.5f) / float(resolution);
      float         best = baryExtrapolate;
      for(const nvmath::vec3ui& triangle : meshData.triangleVertices)
      {
        nvmath::vec3f baryCoord =
            triangleBaryCoord(uv, meshData.vertexTexcoords0[triangle.x], meshData.vertexTexcoords0[triangle.y],
                              meshData.vertexTexcoords0[triangle.z]);
        float inside = std::min(std::min(baryCoord.x, baryCoord.y), baryCoord.z);
        if(inside > best)
        {
          uint32_t pixelIndex        = y * resolution + x;
          best                       = inside;
          nvmath::vec3f position     = baryInterp(meshData.vertexPositions, triangle, baryCoord);
          nvmath::vec3f normal       = baryInterp(meshData.vertexNormals, triangle, baryCoord);
          nvmath::vec4f tangent      = baryInterp(meshData.vertexTangents, triangle, baryCoord);
          nvmath::vec3f dir          = nvmath::normalize(normal);
          nvmath::vec3f start        = position + dir * biasScale.bias;
          dir                        = dir * biasScale.scale;
          nvmath::vec3f intersection = intersectIsosurface(start, start + dir, rock, resolution);
          float         height       = nvmath::dot(intersection - start, dir) / nvmath::dot(dir, dir);
          pixelPositions[pixelIndex] = intersection;
          heights[pixelIndex]        = floatToUnorm16(height);
          nvmath::mat3f tbn;
          tbn.set_col(0, nvmath::vec3f(tangent));
          tbn.set_col(1, nvmath::cross(normal, nvmath::vec3f(tangent)) * tangent.w);
          tbn.set_col(2, normal);
          tbnInv[pixelIndex] = nvmath::inverse(tbn);
        }
      }
    }
  }

  // Generate a normal map based on the sobel filter of the pixel positions
  auto                  normals   = createNormalMap(resolution, texturePrefix + "_normal.png", output, images);
  auto                  addSample = [&pixelPositions, &resolution](nvmath::vec3f& sample, int x, int y, float weight) {
    // Wrap the pixel coordinate. It's never less than -1.
    x                 = (x + resolution) % resolution;
    y                 = (y + resolution) % resolution;
    nvmath::vec3f pos = pixelPositions[y * resolution + x];
    // NAN is used to ignore empty pixels
    if(pos.x != NAN)
      sample += pos * weight;
  };
  for(int y = 0; y < static_cast<int>(resolution); ++y)
  {
    for(int x = 0; x < static_cast<int>(resolution); ++x)
    {
      // Sobel filter, finding the surface position derivative with respect to
      // pixel position.
      nvmath::vec3f gx;
      addSample(gx, x - 1, y - 1, -1.0f);
      addSample(gx, x - 1, y + 0, -2.0f);
      addSample(gx, x - 1, y + 1, -1.0f);
      addSample(gx, x + 1, y - 1, 1.0f);
      addSample(gx, x + 1, y + 0, 2.0f);
      addSample(gx, x + 1, y + 1, 1.0f);
      nvmath::vec3f gy;
      addSample(gy, x - 1, y - 1, -1.0f);
      addSample(gy, x + 0, y - 1, -2.0f);
      addSample(gy, x + 1, y - 1, -1.0f);
      addSample(gy, x - 1, y + 1, 1.0f);
      addSample(gy, x + 0, y + 1, 2.0f);
      addSample(gy, x + 1, y + 1, 1.0f);
      // Invert gy as eMikkTSpace aligns the bitangent with -UV.y
      nvmath::vec3f osNormal      = nvmath::normalize(nvmath::cross(gx, -gy));
      nvmath::vec3f tsNormal      = tbnInv[y * resolution + x] * osNormal;
      normals[y * resolution + x] = vec3fToUnormRGBX8(tsNormal * 0.5f + 0.5f);
    }
  }

  createGltfMesh(output, meshops::MeshView(meshData));
  return true;
}

void toolGenerateAddRequirements(meshops::ContextConfig& contextConfig)
{
  contextConfig.requiresDeviceContext = true;
}

bool toolGenerate(micromesh_tool::ToolContext& context, const ToolGenerateArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene)
{
  bool            result = false;
  auto            model  = std::make_unique<tinygltf::Model>();
  ToolImageVector images;
  switch(args.geometry)
  {
    case ToolGenerateArgs::GEOMETRY_PLANE:
      result = generatePlane(args.resolution, *model);
      break;
    case ToolGenerateArgs::GEOMETRY_CUBE:
      result = generateCube(*model);
      break;
    case ToolGenerateArgs::GEOMETRY_TERRAIN:
      result = generateTerrain(context, args.resolution, *model, images);
      break;
    case ToolGenerateArgs::GEOMETRY_SPHERE:
      result = generateDisplacedCube(context, args.resolution, false, *model, images);
      break;
    case ToolGenerateArgs::GEOMETRY_ROCK:
      result = generateDisplacedCube(context, args.resolution, true, *model, images);
      break;
  }
  if(result)
  {
    scene = micromesh_tool::ToolScene::create(std::move(model), std::move(images), {});
  }
  return result;
}

}  // namespace tool_generate
