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

#include <cstddef>
#include <meshops_internal/meshops_context.h>
#include "meshops_tangents_lengyel.hpp"
#include "meshops_tangents_liani.hpp"
#include <meshops/meshops_operations.h>
#include <mikktspace/mikktspace.h>
#include <atomic>
#include <thread>
#include <nvh/parallel_work.hpp>

namespace meshops {

//////////////////////////////////////////////////////////////////////////

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateVertexDirections(Context context,
                                                                             size_t  count,
                                                                             const OpGenerateVertexDirections_input* inputs,
                                                                             OpGenerateVertexDirections_modified* modifieds)
{
  assert(context && inputs && modifieds);

  for(size_t i = 0; i < count; ++i)
  {
    const OpGenerateVertexDirections_input& input    = inputs[i];
    OpGenerateVertexDirections_modified&    modified = modifieds[i];

    assert(input.mode == OpGenerateVertexDirections_input::Mode::eSmoothTriangleNormals);

    micromesh::OpSmoothMeshDirections_input opInput;
    opInput.triangleAreaWeight = input.smoothTriangleAreaWeight;
    arrayInfoTypedFromView(opInput.meshTriangleVertices, input.triangleUniqueVertexIndices);
    arrayInfoTypedFromView(opInput.meshVertexPositions, modified.meshView.vertexPositions);

    micromesh::OpSmoothMeshDirections_output opOutput;
    if(modified.targetAttribute == eMeshAttributeVertexDirectionBit)
    {
      arrayInfoTypedFromView(opOutput.meshVertexDirections, modified.meshView.vertexDirections);
    }
    else if(modified.targetAttribute == eMeshAttributeVertexNormalBit)
    {
      arrayInfoTypedFromView(opOutput.meshVertexDirections, modified.meshView.vertexNormals);
    }
    else
    {
      return micromesh::Result::eInvalidValue;
    }

    micromesh::Result result = micromesh::micromeshOpSmoothMeshDirections(context->m_micromeshContext, &opInput, &opOutput);

    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpApplyBounds(Context                    context,
                                                                size_t                     count,
                                                                const OpApplyBounds_input* inputs,
                                                                OpApplyBounds_modified*    modifieds)
{
  for(size_t i = 0; i < count; ++i)
  {
    const OpApplyBounds_input& input    = inputs[i];
    OpApplyBounds_modified&    modified = modifieds[i];
    if(input.meshView.vertexDirectionBounds.empty())
    {
      MESHOPS_LOGE(context, "meshops::OpApplyBounds_input[%zu].meshView.vertexDirectionBounds is empty", i);
      return micromesh::Result::eInvalidValue;
    }
    if(input.meshView.vertexCount() != modified.meshView->vertexCount())
    {
      MESHOPS_LOGE(context,
                   "meshops::OpApplyBounds_input[%zu] vertex count does not match meshops::OpApplyBounds_modified[%zu]", i, i);
      return micromesh::Result::eInvalidValue;
    }
    if(modified.meshView->vertexPositions.empty())
    {
      MESHOPS_LOGE(context, "meshops::OpApplyBounds_modified[%zu].meshView->vertexPositions is empty", i);
      return micromesh::Result::eInvalidValue;
    }
    if(modified.meshView->vertexDirections.empty())
    {
      MESHOPS_LOGE(context, "meshops::OpApplyBounds_modified[%zu].meshView->vertexDirections is empty", i);
      return micromesh::Result::eInvalidValue;
    }
  }
  for(size_t i = 0; i < count; ++i)
  {
    const OpApplyBounds_input& input      = inputs[i];
    OpApplyBounds_modified&    modified   = modifieds[i];
    auto&                      bounds     = input.meshView.vertexDirectionBounds;
    auto&                      positions  = modified.meshView->vertexPositions;
    auto&                      directions = modified.meshView->vertexDirections;
    nvh::parallel_ranges(
        input.meshView.vertexCount(),
        [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadIdx) {
          for(uint64_t i = idxBegin; i < idxEnd; i++)
          {
            // Add the bounds bias to the position
            positions[i] += directions[i] * bounds[i].x;

            // Multiply the direction vector by the bounds scale
            directions[i] *= bounds[i].y;
          }
        },
        micromesh::micromeshOpContextGetConfig(context->m_micromeshContext).threadCount);

    // Clear any vertexDirectionBounds on the output
    modified.meshView->resize(meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit, 0, 0);
  }
  return micromesh::Result::eSuccess;
}

MESHOPS_API TangentSpaceAlgorithm MESHOPS_CALL tangentAlgorithmFromName(const char* name)
{
  if(strcmp(name, "lengyel") == 0)
  {
    return TangentSpaceAlgorithm::eLengyel;
  }
  else if(strcmp(name, "liani") == 0)
  {
    return TangentSpaceAlgorithm::eLiani;
  }
  else if(strcmp(name, "mikktspace") == 0)
  {
    return TangentSpaceAlgorithm::eMikkTSpace;
  }
  else
  {
    return TangentSpaceAlgorithm::eInvalid;
  }
}

MESHOPS_API const char* MESHOPS_CALL getTangentAlgorithmName(TangentSpaceAlgorithm algorithm)
{
  switch(algorithm)
  {
    case TangentSpaceAlgorithm::eInvalid:
    default:
      return nullptr;
    case TangentSpaceAlgorithm::eLengyel:
      return "lengyel";
    case TangentSpaceAlgorithm::eLiani:
      return "liani";
    case TangentSpaceAlgorithm::eMikkTSpace:
      return "mikktspace";
  }
}

namespace {
class CalcMikktTangents
{
public:
  static inline MutableMeshView* getMeshView(const SMikkTSpaceContext* context)
  {
    MutableMeshView* meshView = reinterpret_cast<MutableMeshView*>(context->m_pUserData);
    return meshView;
  }

  CalcMikktTangents()
  {
    iface.m_getNumFaces = [](const SMikkTSpaceContext* context) -> int {
      return int(getMeshView(context)->triangleCount());
    };

    iface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* context, int iFace) { return 3; };

    iface.m_getNormal = [](const SMikkTSpaceContext* context, float outnormal[], int iFace, int iVert) {
      MutableMeshView* meshView = getMeshView(context);
      uint32_t         vertID   = meshView->triangleVertices[iFace][iVert];

      auto normal  = meshView->vertexNormals[vertID];
      outnormal[0] = normal.x;
      outnormal[1] = normal.y;
      outnormal[2] = normal.z;
    };

    iface.m_getPosition = [](const SMikkTSpaceContext* context, float outpos[], int iFace, int iVert) {
      MutableMeshView* meshView = getMeshView(context);
      uint32_t         vertID   = meshView->triangleVertices[iFace][iVert];
      auto             position = meshView->vertexPositions[vertID];
      outpos[0]                 = position.x;
      outpos[1]                 = position.y;
      outpos[2]                 = position.z;
    };

    iface.m_getTexCoord = [](const SMikkTSpaceContext* context, float outuv[], int iFace, int iVert) {
      MutableMeshView* meshView = getMeshView(context);
      uint32_t         vertID   = meshView->triangleVertices[iFace][iVert];
      auto             uv       = meshView->vertexTexcoords0[vertID];
      outuv[0]                  = uv.x;
      outuv[1]                  = uv.y;
    };

    iface.m_setTSpaceBasic = [](const SMikkTSpaceContext* context, const float tangentu[], float fSign, int iFace, int iVert) {
      MutableMeshView* meshView = getMeshView(context);
      uint32_t         vertID   = meshView->triangleVertices[iFace][iVert];

      auto& tangents = meshView->vertexTangents[vertID];
      tangents.x     = tangentu[0];
      tangents.y     = tangentu[1];
      tangents.z     = tangentu[2];
      tangents.w     = -fSign;

      if(tangents.x == 0 && tangents.y == 0 && tangents.z == 0)
        tangents.z = 1.0f;
    };

    context.m_pInterface = &iface;
  }

  void calc(MutableMeshView* data)
  {
    context.m_pUserData = data;
    genTangSpaceDefault(&this->context);
  }

private:
  SMikkTSpaceInterface iface{};
  SMikkTSpaceContext   context{};
};

}  // namespace

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateVertexTangentSpace(Context context,
                                                                               size_t  count,
                                                                               const OpGenerateVertexTangentSpace_input* inputs,
                                                                               OpGenerateVertexTangentSpace_modified* modifieds)
{
  assert(context && inputs && modifieds);

  CalcMikktTangents calcMikkt;

  // TODO could multi-thread this loop
  for(size_t i = 0; i < count; ++i)
  {
    const OpGenerateVertexTangentSpace_input& input    = inputs[i];
    OpGenerateVertexTangentSpace_modified&    modified = modifieds[i];
    MutableMeshView&                          mesh     = modified.meshView;

    micromesh::Result result = micromesh::Result::eSuccess;

    MeshAttributeFlags requiredFlags = eMeshAttributeVertexPositionBit | eMeshAttributeVertexNormalBit
                                       | eMeshAttributeVertexTexcoordBit | eMeshAttributeTriangleVerticesBit;

    // Test if the mesh has the required flags to proceed
    if(!modified.meshView.hasMeshAttributeFlags(requiredFlags))
    {
      const meshops::MeshAttributeFlags meshAttrNew = (~modified.meshView.getMeshAttributeFlags()) & requiredFlags;
      LOGE("Error: missing %s\n", meshops::meshAttribBitsString(meshAttrNew).c_str());
      return micromesh::Result::eInvalidValue;
    }


    if(input.algorithm == TangentSpaceAlgorithm::eLengyel)
    {
      createLengyelTangents(mesh);
    }
    else if(input.algorithm == TangentSpaceAlgorithm::eLiani)
    {
      // Space for tangents and bitangents. generateTangents generates facevarying
      // tangents - i.e. 1 tangent per index buffer element; we'll reduce this
      // at the end.
      std::vector<nvmath::vec4f> tangents(mesh.indexCount());
      std::vector<uint32_t>      adjacencyMap;
      assert(mesh.indexCount() <= std::numeric_limits<int>::max());
      assert(mesh.vertexCount() <= std::numeric_limits<int>::max());
      static_assert(alignof(nvmath::vec3i) == 4);
      createLianiTangents(context,                                                   /* context */
                          reinterpret_cast<uint32_t*>(mesh.triangleVertices.data()), /* varyingIndices */
                          nullptr,                                                   /* facevaryingIndices: not used */
                          nullptr, /* uniformIndices: not used; we assume source was a triangular mesh rather than a triangulated polygonal mesh */
                          uint32_t(mesh.vertexCount()),   /* numVarying */
                          uint32_t(mesh.indexCount()),    /* numFacevarying */
                          uint32_t(mesh.triangleCount()), /* numTriangles */
                          mesh.vertexNormals.data(),      /* inNormal */
                          mesh.vertexPositions.data(),    /* inPosition */
                          mesh.vertexTexcoords0.data(),   /* inUvs */
                          false,                          /* facevaryingN */
                          false,                          /* facevaryingTx */
                          adjacencyMap,                   /* adjacencyMap */
                          tangents.data()                 /* tangents */
      );

      // Turn vec3f tangents + bitangents into glTF-style vec4f tangents + handedness.
      // Like the MikkTSpace functions above, we handle welding relatively simply -
      // we ultimately take the tangent frame of the last time a vertex is referenced.
      // We can make use of the generateTangents adjacencyMap output to implement
      // the "vertex -> last index that referenced it" lookup.
      struct CombineFacevaryingArgs
      {
        const uint32_t*                   verticesOffset;
        const uint32_t*                   verticesValence;
        const uint32_t*                   verticesLists;
        const std::vector<nvmath::vec4f>* tangents;
        meshops::MutableMeshView*         mesh;
      } args{};
      args.verticesOffset  = &adjacencyMap[1];
      args.verticesValence = &adjacencyMap[1 + mesh.vertexCount()];
      args.verticesLists   = &adjacencyMap[1 + 2 * mesh.vertexCount()];
      args.tangents        = &tangents;
      args.mesh            = &mesh;
      micromesh::OpDistributeWork_input parallelInput{};
      parallelInput.userData                 = &args;
      parallelInput.pfnGenericSingleWorkload = [](uint64_t vtx, uint32_t /* unused */, void* userData) {
        CombineFacevaryingArgs* args    = reinterpret_cast<CombineFacevaryingArgs*>(userData);
        const uint32_t          valence = args->verticesValence[vtx];
        if(valence == 0)
        {
          return;  // This vertex wasn't referenced by any indices.
        }
        const uint32_t      lastIndex   = args->verticesLists[args->verticesOffset[vtx] + (valence - 1)];
        const nvmath::vec4f tangent     = args->tangents->at(lastIndex);
        args->mesh->vertexTangents[vtx] = tangent;
      };
      result = micromesh::micromeshOpDistributeWork(context->m_micromeshContext, &parallelInput, mesh.vertexCount());
    }
    else if(input.algorithm == TangentSpaceAlgorithm::eMikkTSpace)
    {
      calcMikkt.calc(&mesh);
    }
    else
    {
      MESHOPS_LOGE(context, "inputs->[%zu].algorithm (%u) must be one of eLengyel, eLiani, or eMikkTSpace.", i,
                   uint32_t(input.algorithm));
      result = micromesh::Result::eInvalidValue;
    }

    if(result != micromesh::Result::eSuccess)
    {
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

// Implementation of atomic floats for C++17 compatibility.
// Performance looks similar to C++20 std::atomic<float> on this use case.
struct AtomicFloat
{
  AtomicFloat() = default;
  AtomicFloat(float x)
      : f32(x)
  {
  }
  union
  {
    std::atomic_uint atomicU32;
    float            f32;
  };

  inline float fetch_add(float x)
  {
    static_assert(sizeof(std::atomic_uint) == sizeof(float));
    while(true)
    {
      uint32_t expected = atomicU32;
      if(atomicU32.compare_exchange_weak(expected, glm::floatBitsToUint(f32 + x)))
      {
        return glm::uintBitsToFloat(expected);
      }
    }
  }
  inline void store(float x) { f32 = x; }

  inline operator float() const { return f32; }
};

static bool checkVector(const nvmath::vec3f& d)
{
  if((d.x == 0.f && d.y == 0.f && d.z == 0.f) || std::isnan(d.x) || std::isnan(d.y) || std::isnan(d.z))
  {
    return false;
  }
  return true;
}

// Thread-safe unordered map
template <typename Key, typename Payload>
class ConcurrentHashmap
{
public:
  ConcurrentHashmap(size_t maxItems) { m_entries.resize(4ull * maxItems); }

  // Insert the payload p with key k in the map, and return the index of the entry
  inline size_t insert(const Key& k, const Payload& p)
  {
    uint32_t h               = hash(k);
    uint32_t c               = checksum(k);
    bool     found           = false;
    uint32_t searchIteration = 0u;
    while(!found)
    {
      uint32_t expected = 0u;
      found             = m_entries[h].checksum.compare_exchange_weak(expected, c) || (expected == c);
      if(found)
      {
        m_entries[h].payload = p;
      }
      else
      {
        // Combine linear search and regenerating a new hash key:
        // Linear search is more cache-friendly than regeneration, but tends
        // to create high-density zones in the hash map. Regeneration results in
        // a better spread, at the expense of cache coherence. Also in some cases
        // wangHash(wangHash(h)) == h, creating an infinite loop. Mixing linear
        // search and hash key regeneration offers a good compromise
        // between spread and cache coherence. If no free slot has been found
        // after 1024 iterations it becomes probable the search
        // hits an infinite loop. In this case we revert to simple linear search.
        if(searchIteration % 16 == 0 && searchIteration < 1024)
          h = wangHash(h) % static_cast<uint32_t>(m_entries.size());
        else
          h = (h + 1) % static_cast<uint32_t>(m_entries.size());
        searchIteration++;
      }
    }
    return h;
  }
  // Fetch a payload from an entry index
  inline Payload& operator[](size_t h) { return m_entries[h].payload; }

  void clear() { m_entries.clear(); }


private:
  struct Entry
  {
    Entry() = default;
    Entry(const Entry& e) {}
    std::atomic_uint32_t checksum = 0u;
    Payload              payload;
  };

  inline uint32_t wangHash(uint32_t seed)
  {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
  }

  inline uint32_t xorshift32(uint32_t x64)
  {
    x64 ^= x64 << 13;
    x64 ^= x64 >> 7;
    x64 ^= x64 << 17;
    return std::max(1u, x64);
  }

  inline uint32_t hash(const Key& k)
  {
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&k);
    uint32_t        h   = 0u;
    for(size_t i = 0; i < (sizeof(Key) / 4); i++)
      h = wangHash(h + ptr[i]);
    return h % static_cast<uint32_t>(m_entries.size());
  }
  inline uint32_t checksum(const Key& k)
  {
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&k);
    uint32_t        h   = 0u;
    for(size_t i = 0; i < (sizeof(Key) / 4); i++)
      h = xorshift32(h + ptr[i]);
    return h;
  }

  std::vector<Entry> m_entries;
};

// Generate per-vertex directions by averaging the face normals adjacent to each vertex
MESHOPS_API micromesh::Result MESHOPS_CALL
meshopsGenerateVertexDirections(Context context, meshops::ResizableMeshView& meshView)
{
  struct AtomicDirection
  {
    AtomicFloat      x{0.f};
    AtomicFloat      y{0.f};
    AtomicFloat      z{0.f};
    std::atomic_uint counter{0u};

    AtomicDirection& operator=(const AtomicDirection& d)
    {
      x.store(d.x);
      y.store(d.y);
      z.store(d.z);
      counter.store(d.counter);
      return *this;
    }

    inline void add(const nvmath::vec3f& d)
    {
      uint32_t existingCounter = counter.fetch_add(1);

      // Artificially change the magnitude of the first
      // contribution to the average direction.
      // This prevents the average from falling to 0
      // when a vertex is adjacent to 2 sets of triangles
      // with exactly opposite face normals
      if(existingCounter == 0)
      {
        x.fetch_add(d.x * 1.001f);
        y.fetch_add(d.y * 1.001f);
        z.fetch_add(d.z * 1.001f);
      }
      else
      {
        x.fetch_add(d.x);
        y.fetch_add(d.y);
        z.fetch_add(d.z);
      }
    }

    inline nvmath::vec3f get()
    {
      float         divider = 1.f / static_cast<float>(counter);
      nvmath::vec3f d;
      d.x = x * divider;
      d.y = y * divider;
      d.z = z * divider;
      return normalize(d);
    }
    inline bool isValid() { return counter > 0u; }
  };

  ConcurrentHashmap<nvmath::vec3f, AtomicDirection> uniqueDirs(meshView.vertexCount());


  std::vector<size_t> vertexIndices(meshView.vertexCount());


  nvh::parallel_batches(
      meshView.vertexCount(),
      [&](uint64_t threadIdx) {
        vertexIndices[threadIdx] = uniqueDirs.insert(meshView.vertexPositions[threadIdx], AtomicDirection{});
      },
      std::thread::hardware_concurrency());


  nvh::parallel_batches(
      meshView.triangleCount(),
      [&](uint64_t threadIdx) {
        const nvmath::vec3ui& indices = meshView.triangleVertices[threadIdx];
        const nvmath::vec3f&  v0      = meshView.vertexPositions[indices[0]];
        const nvmath::vec3f&  v1      = meshView.vertexPositions[indices[1]];
        const nvmath::vec3f&  v2      = meshView.vertexPositions[indices[2]];

        // Need to normalize everywhere to prevent small floating-point vertex coordinates
        // from rounding to 0 in the cross product
        nvmath::vec3f e0 = normalize(v1 - v0);
        nvmath::vec3f e1 = normalize(v2 - v0);
        nvmath::vec3f n  = normalize(cross(e0, e1));

        if(checkVector(n))
        {
          uniqueDirs[vertexIndices[indices[0]]].add(n);
          uniqueDirs[vertexIndices[indices[1]]].add(n);
          uniqueDirs[vertexIndices[indices[2]]].add(n);
        }
      },
      std::thread::hardware_concurrency());
  bool hasBadDirections = false;
  nvh::parallel_batches(
      meshView.vertexCount(),
      [&](uint64_t threadIdx) {
        nvmath::vec3f d = nvmath::vec3f(0.f, 1.f, 0.f);
        if(uniqueDirs[vertexIndices[threadIdx]].isValid())
        {
          d = uniqueDirs[vertexIndices[threadIdx]].get();
          if((d.x == 0.f && d.y == 0.f && d.z == 0.f) || std::isnan(d.x) || std::isnan(d.y) || std::isnan(d.z))
          {
            d                = nvmath::vec3f(0.f, 1.f, 0.f);
            hasBadDirections = true;
          }
        }
        meshView.vertexDirections[threadIdx]      = d;
        meshView.vertexDirectionBounds[threadIdx] = nvmath::vec2f(0.f);
      },
      std::thread::hardware_concurrency());
  
  if (!hasBadDirections)
  {
    return micromesh::Result::eSuccess;
  }
  else
  {
    return micromesh::Result::eFailure;
  }
}

template <typename T>
void atomicMax(std::atomic<T>& maximum_value, T const& value) noexcept
{
  T prev_value = maximum_value;
  while(prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value))
  {
  }
}
template <typename T>
void atomicMin(std::atomic<T>& maximum_value, T const& value) noexcept
{
  T prev_value = maximum_value;
  while(prev_value > value && !maximum_value.compare_exchange_weak(prev_value, value))
  {
  }
}

MESHOPS_API float MESHOPS_CALL
meshopsComputeMeshViewExtent(Context context, const meshops::MutableMeshView& meshview)
{
  uint32_t numThreads = std::thread::hardware_concurrency();

  nvmath::vec3f diagonal;

  std::atomic<float> bboxMin[3];
  std::atomic<float> bboxMax[3];

  for(uint32_t i = 0; i < 3; i++)
  {
    bboxMin[i].store(FLT_MAX);
    bboxMax[i].store(-FLT_MAX);
  }

  nvh::parallel_batches(
      meshview.vertexPositions.size(),
      [&](uint64_t vertIdx) {
        nvmath::vec3f v = meshview.vertexPositions[vertIdx];
        for(uint32_t i = 0; i < 3; i++)
        {
          atomicMax(bboxMax[i], v[i]);
          atomicMin(bboxMin[i], v[i]);
        }
      },
      numThreads

  );

  for(uint32_t i = 0; i < 3; i++)
    diagonal[i] = bboxMax[i] - bboxMin[i];

  return nvmath::length(diagonal);
}


}  // namespace meshops
