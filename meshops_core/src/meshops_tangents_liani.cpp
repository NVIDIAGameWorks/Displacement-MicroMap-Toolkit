// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "meshops_tangents_liani.hpp"
#include "micromesh/micromesh_operations.h"
#include <nvmath/nvmath.h>
#include <atomic>
#include <stdint.h>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X86

// Include the relevant intrinsic file for float3a for the target platform.
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#define TANGENTS_PROFILE_ZONE(x, y)

using float2 = nvmath::vec2f;
using float3 = nvmath::vec3f;
using float4 = nvmath::vec4f;

namespace meshops {
inline float2 fract(const float2& a)
{
  return float2(a.x - int(a.x), a.y - int(a.y));
}
inline float reduce_min(const float2& a)
{
  return std::min(a.x, a.y);
}
inline float reduce_max(const float2& a)
{
  return std::max(a.x, a.y);
}

inline float dot(float3 a, float3 b)
{
  return nvmath::dot(a, b);
}
inline float3 cross(float3 a, float3 b)
{
  return nvmath::cross(a, b);
}

inline float3 projectToPlane(float3 vector, float3 planeNormal)
{
  return vector - planeNormal * dot(planeNormal, vector);
}
inline float length(float3 a)
{
  return nvmath::length(a);
}
inline float length2(float3 a)
{
  return nvmath::dot(a, a);
}

// This isn't exactly the same as nvmath::normalize! nvmath use nv_eps as the
// threshold, while this uses exactly 0.
inline float3 normalize(float3 a)
{
  float floatLength = length(a);
  if(floatLength == 0.f)
  {
    return a;
  }

  return (1.0f / floatLength) * a;
}
inline float sqr(float a)
{
  return a * a;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Block-vectorized implementation of float3

#ifdef ARCH_X86
struct float3a
{
  union
  {
    __m128 m128;
    struct
    {
      float x, y, z, a;
    };
  };

  float3a() = default;

  float3a(float _x)
      : m128(_mm_set1_ps(_x))
  {
  }
  float3a(float _x, float _y, float _z)
      : m128(_mm_set_ps(0, _z, _y, _x))
  {
  }
  float3a(const float3& other)
      : m128(_mm_set_ps(0, other.z, other.y, other.x))
  {
  }
  float3a(const __m128& other)
      : m128(other)
  {
  }
  const float3a& operator=(const float3a& other)
  {
    m128 = other.m128;
    return *this;
  }
  const float3a& operator=(const float3& other)
  {
    m128 = _mm_set_ps(0, other.z, other.y, other.x);
    return *this;
  }
};

template <int l0, int l1, int l2, int l3>
inline __m128 shuffle(__m128 a)
{
  return _mm_shuffle_ps(a, a, _MM_SHUFFLE(l3, l2, l1, l0));
}

inline __m128 abs(__m128 a)
{
  return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));  //< mask out sign bit
}

inline float3a operator*(float lhs, const float3a& rhs)
{
  return float3a(_mm_mul_ps(_mm_set1_ps(lhs), rhs.m128));
}

inline float3a operator*(const float3a& lhs, float rhs)
{
  return float3a(_mm_mul_ps(lhs.m128, _mm_set1_ps(rhs)));
}
inline float3a operator/(const float3a& lhs, float rhs)
{
  return float3a(_mm_div_ps(lhs.m128, _mm_set1_ps(rhs)));
}
inline float3a& operator+=(float3a& lhs, const float3a& rhs)
{
  lhs.m128 = _mm_add_ps(lhs.m128, rhs.m128);
  return lhs;
}
inline float3a& operator-=(float3a& lhs, const float3a& rhs)
{
  lhs.m128 = _mm_sub_ps(lhs.m128, rhs.m128);
  return lhs;
}
inline float3a& operator*=(float3a& lhs, float rhs)
{
  lhs.m128 = _mm_mul_ps(lhs.m128, _mm_set1_ps(rhs));
  return lhs;
}
inline float3a& operator/=(float3a& lhs, float rhs)
{
  lhs.m128 = _mm_div_ps(lhs.m128, _mm_set1_ps(rhs));
  return lhs;
}

inline bool operator==(const float3a& lhs, const float3a& rhs)
{
  return (_mm_movemask_ps(_mm_cmpeq_ps(lhs.m128, rhs.m128)) & 7) == 7;
}
inline float3a operator+(const float3a& lhs, const float3a& rhs)
{
  return float3a(_mm_add_ps(lhs.m128, rhs.m128));
}
inline float3a operator-(const float3a& lhs, const float3a& rhs)
{
  return float3a(_mm_sub_ps(lhs.m128, rhs.m128));
}
inline float3a operator*(const float3a& lhs, const float3a& rhs)
{
  return float3a(_mm_mul_ps(lhs.m128, rhs.m128));
}
inline float3a operator/(const float3a& lhs, const float3a& rhs)
{
  return float3a(_mm_div_ps(lhs.m128, rhs.m128));
}

inline float dot(float3a a, float3a b)
{
#if defined(__SSE4_1__)
  return _mm_cvtss_f32(_mm_dp_ps(a.m128, b.m128, 0x7F));
#else
  const __m128 x = _mm_mul_ps(a.m128, b.m128);
  const __m128 y = shuffle<1, 1, 1, 1>(x);
  const __m128 z = shuffle<2, 2, 2, 2>(x);
  return _mm_cvtss_f32(_mm_add_ps(_mm_add_ps(x, y), z));  //< x+y+z, then extract lane for x.
#endif
}
inline float3a cross(float3a a, float3a b)
{
  // Reference implementation:
  //return float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };

  // Block vectorization:
  // Transpase xyz to yzx
  const __m128 bt = shuffle<1, 2, 0, 3>(b.m128);
  const __m128 at = shuffle<1, 2, 0, 3>(a.m128);
  const __m128 c  = _mm_sub_ps(_mm_mul_ps(a.m128, bt), _mm_mul_ps(at, b.m128));
  return float3a(shuffle<1, 2, 0, 3>(c));
}
inline float3a projectToPlane(float3a vector, float3a planeNormal)
{
  return vector - planeNormal * dot(planeNormal, vector);
}
inline float length(float3a a)
{
  return sqrtf(dot(a, a));
}
inline float length2(float3a a)
{
  return dot(a, a);
}
inline float rsqrt(const float x)
{
  const __m128 a = _mm_set_ss(x);
  const __m128 r = _mm_rsqrt_ss(a);
  const __m128 c = _mm_add_ss(_mm_mul_ss(_mm_set_ss(1.5f), r),
                              _mm_mul_ss(_mm_mul_ss(_mm_mul_ss(a, _mm_set_ss(-0.5f)), r), _mm_mul_ss(r, r)));
  return _mm_cvtss_f32(c);
}
inline float3a normalize(float3a a)
{
  // Reference implementation:
  //float floatLength = length(a);
  //if (floatLength == 0.f)
  //{
  //    return a;
  //}
  //
  //return (1.0f / floatLength) * a;

  float len2 = length2(a);
  if(len2 < 1e-19f)  //< rsqrt uses approximate math, values lower than 1e-19 are not resolved well
    return a;

  return a * rsqrt(len2);
}

#else  //< Other architecture
// No SIMD version for ARM yet
using float3a = float3;

#endif


// a naive implementation, we need to visit https://floating-point-gui.de/errors/comparison/
// if precision is not satified
inline bool approximatelySame(float2 f1, float2 f2, float epsilon = 1e-6f)
{
  float2 f = f1 - f2;
  return fabsf(f.x) <= epsilon && fabsf(f.y) <= epsilon;
}

inline bool approximatelySamePeriodic(float2 f1, float2 f2, float epsilon = 1e-6f)
{
#ifdef ARCH_X86
  // Reference implementation:
  // float2 f = fract(f1) - fract(f2);
  // return fabsf(f.x) <= epsilon && fabsf(f.y) <= epsilon;

  // Block vectorization:
  __m128 f_value    = _mm_set_ps(f2.y, f2.x, f1.y, f1.x);  //< load values to lanes[01], lanes[23]
  __m128 f_integer  = _mm_cvtepi32_ps(_mm_cvttps_epi32(f_value));
  __m128 f_fraction = _mm_sub_ps(f_value, f_integer);

  __m128 f = _mm_sub_ps(shuffle<0, 1, 0, 1>(f_fraction), shuffle<2, 3, 2, 3>(f_fraction));  //< subtract lanes[01] - lanes[23]
  __m128 abs_f = abs(f);
  return (_mm_movemask_ps(_mm_cmple_ps(abs_f, _mm_set1_ps(epsilon))) & 3) == 3;  //< mask 3 represents the first 2 lanes (xy)

#else  //< Other architecture
  float2 f = fract(f1) - fract(f2);
  return fabsf(f.x) <= epsilon && fabsf(f.y) <= epsilon;
#endif
}

inline bool approximatelySame(float3 f1, float3 f2, float epsilon = 1e-6f)
{
  float3 f = f1 - f2;
  return fabsf(f.x) <= epsilon && fabsf(f.y) <= epsilon && fabsf(f.z) <= epsilon;
}

#ifdef ARCH_X86
inline bool approximatelySame(float3a f1, float3a f2, float epsilon = 1e-6f)
{
  float3a f = f1 - f2;

  // Reference implementation:
  // return fabsf(f.x) <= epsilon && fabsf(f.y) <= epsilon && fabsf(f.z) <= epsilon;

  // Block vectorization:
  __m128 abs_f = abs(f.m128);
  return (_mm_movemask_ps(_mm_cmple_ps(abs_f, _mm_set1_ps(epsilon))) & 7) == 7;  //< mask 7 represents the first 3 lanes (xyz)
}
#endif


struct Args
{
  // Indices of points to define the topology. We don't need to know how many indices per face.
  const uint32_t* varyingIndices;
  // Optional triangulated indices, if not null this is an indirection buffer to read from varyingIndices.
  // If null then varyingIndices are of a topology already made of triangles.
  const uint32_t* facevaryingIndices;
  // Optional triangulated face indices. If set it allows us to know which authored face a triangle belongs to
  // This is only required to safely partition workload across threads.
  const uint32_t* uniformIndices;
  // In short, the number of points, or any other primvar defined with "varying" interpolation.
  int numVarying;
  // The number of values in facevarying primvars
  int           numFacevarying;
  int           numTriangles;
  const float3* inPosition;
  // normals and texture coordinates can be either varying or facevarying. Each defined by the bool flag below.
  const float3* inNormal;
  const float2* inUvs;
  const bool    facevaryingN;
  const bool    facevaryingTx;

  // Results:
  float4* tangentU;  //< The tangents. The w component is the bitangent sign, and is only set at the very end.
  float3* tangentV;  //< The bitangents. Normal, tangent and bitangent are guaranteed to be orthonormal at the end.

  // Temporaries and scratch space from here on:
  std::vector<uint32_t>& adjacencyMap;
  uint8_t*               tangentInit;

  int       maxValence = 0;
  uint32_t* verticesOffset;
  uint32_t* verticesValence;
  uint32_t* verticesLists;

  std::atomic<int> workload;
};

// Here we compute some form of adjacency table. More specific, we need the valence for each point, AKA how
// many faces each point is part of (faces, not triangles). We also need to know which facevarying indices
// these adjacent corners have. We store the data per point, so we have acess to each facevarying tangent
// for a point.
// Important: this adjacency map data is reused as-is in scenedb::Mesh::weldVertices. Any change to this data
//            affects the other algorithm.
void prepareAdjacencyMap(Args& args)
{
  TANGENTS_PROFILE_ZONE(kProfilerMask, "generateTangents_valence");

  // Make space in the adjacency map data.
  // Important: this adjacency map data is reused as-is in scenedb::Mesh::weldVertices. Any change to this data
  //            affects the other algorithm.
  args.adjacencyMap.resize(args.numVarying * 2 + args.numFacevarying + 1, 0u);

  uint32_t*       verticesOffset  = args.adjacencyMap.data() + 1;
  uint32_t*       verticesValence = verticesOffset + args.numVarying;
  uint32_t*       verticesLists   = verticesValence + args.numVarying;
  uint32_t        maxValence      = 0;
  const uint32_t* varyingIndices  = args.varyingIndices;
  for(int index = 0, size = args.numFacevarying; index < size; ++index)
  {
    const uint32_t entry = varyingIndices[index];
    ++verticesOffset[entry];  //< this is the valence
  }

  // Convert vertex valence to prefixSum to partition space where to store adjacency. We know exactly how
  // much space: args.numFacevarying, it's how it is partitioned that we need to determine.
  uint32_t prefixSum = 0;
  for(int entry = 0, size = args.numVarying; entry < size; ++entry)
  {
    const uint32_t valence = verticesOffset[entry];
    verticesOffset[entry]  = prefixSum;
    prefixSum += valence;
    // compute max valence, we need it to size scratch vectors.
    maxValence = std::max(maxValence, valence);
  }

  // This third pass we collect the facevarying indices.
  for(int index = 0, size = args.numFacevarying; index < size; ++index)
  {
    const uint32_t entry      = varyingIndices[index];
    const uint32_t i          = verticesValence[entry]++;  //< this is the valence, again.
    const uint32_t offset     = verticesOffset[entry];
    verticesLists[offset + i] = index;
  }

  args.adjacencyMap[0] = maxValence;
  args.maxValence      = maxValence;
  args.verticesOffset  = verticesOffset;
  args.verticesValence = verticesValence;
  args.verticesLists   = verticesLists;
}

constexpr int k_weightMode_angle = 0;
constexpr int k_weightMode_area  = 1;

template <int weightMode>
void prepareFacevarying(uint64_t index, uint32_t /* unused */, void* ud)
{
  Args&         args        = *static_cast<Args*>(ud);
  constexpr int k_stealSize = 1024;  //< work steal increments.

  // Only the first thread proceeds generating the adjacency map. Is seems convoluted done this way, but this
  // trick is a good performance win as it generates the data in parallel with the rest of the workload!
  if(index == 0)
  {
    prepareAdjacencyMap(args);
  }

  TANGENTS_PROFILE_ZONE(kProfilerMask, "generateTangents_prepareFacevaryingThread");
  while(true)
  {
    // Self-managed work stealing: work is partitioned between a number of threads, but then each thread
    // gets to advance at its own pace; due to prehemption, logical threads etc... threads rarely have
    // identical throughput. This strategy is on the line of CUDA persistent threads.

    // Begin with the assuption it is safe to consume batches of triangle and they are all independent.
    int start = args.workload.fetch_add(k_stealSize);
    int end   = std::min(start + k_stealSize, args.numTriangles);
    if(start >= args.numTriangles)
      break;

    // If uniformIndices are defined it means that the topology was made of arbitrary polygons then triangulated.
    // The prepareFacevarying pass is only thread safe to parallelize as far as the triangles originating from the
    // triangulation of one polygon are scheduled within the same thread. Those are stored consecutively by design
    // of HdMeshUtilX::ComputeTriangleIndices. To produce safe execution ranges, we need to scan the content of
    // args.uniformIndices and look for changes of values: differtent value means different originating face.
    if(args.uniformIndices)
    {
      // Align the range start and end to not straddle across triangles tessellated from a polygon. It is
      // easy to do so by looking for change of value in uniformIndices.
      auto rollForward = [](const uint32_t* uniformIndices, const int numTriangles, int index) -> int {
        uint32_t faceIndex = uniformIndices[index - 1];
        while(index < numTriangles && uniformIndices[index] == faceIndex)
        {
          ++index;
        }
        return index;
      };
      // Do not roll-forward the very first and last indices, those are exact.
      if(start != 0)
      {
        start = rollForward(args.uniformIndices, args.numTriangles, start);
      }
      if(end < args.numTriangles)
      {
        end = rollForward(args.uniformIndices, args.numTriangles, end);
      }
      // Protect for exceptinal condition where a single polygon got tessellated to more triangles than
      // k_stealSize, or the small reminder in the last range.
      if(start == end)
      {
        continue;
      }
    }

    // Loop over triangles, but produce tangents for the original non triangulated topology.
    // These tangents are not smooth yet, they are first otder approximation for each vertex of each face.
    for(int i = start; i < end; ++i)
    {
      // Facevarying indices of current triangle
      uint32_t f0 = (args.facevaryingIndices ? args.facevaryingIndices[i * 3 + 0] : i * 3 + 0);
      uint32_t f1 = (args.facevaryingIndices ? args.facevaryingIndices[i * 3 + 1] : i * 3 + 1);
      uint32_t f2 = (args.facevaryingIndices ? args.facevaryingIndices[i * 3 + 2] : i * 3 + 2);

      // Varying indices (think, the indices to fetch points...)
      uint32_t v0 = args.varyingIndices[f0];
      uint32_t v1 = args.varyingIndices[f1];
      uint32_t v2 = args.varyingIndices[f2];

      // Gather the initial quantities, points, normals, uvs. They are guaranteed to be available
      // otherwise we would not be here.
      const float3a p0 = args.inPosition[v0];
      const float3a p1 = args.inPosition[v1];
      const float3a p2 = args.inPosition[v2];

      const float2 t0 = args.inUvs[args.facevaryingTx ? f0 : v0];
      const float2 t1 = args.inUvs[args.facevaryingTx ? f1 : v1];
      const float2 t2 = args.inUvs[args.facevaryingTx ? f2 : v2];

      // Need normalized normals to project tangets to planes.
      const float3a n0 = normalize(args.inNormal[args.facevaryingN ? f0 : v0]);
      const float3a n1 = normalize(args.inNormal[args.facevaryingN ? f1 : v1]);
      const float3a n2 = normalize(args.inNormal[args.facevaryingN ? f2 : v2]);

      // Math inspired by Mikktspace: https://gitlab-master.nvidia.com/carbon/Carbonite-externals/mikktspace
      // In essence: compute the rate of change of the texcood over the triangle...
      // If we call uv the triangle baricentrics, and texcoord ST, these would be dSTdu and dSTdv
      const float2 dSTdu = t1 - t0;
      const float2 dSTdv = t2 - t0;

      // And the rate of change of the of P over barycentric u and v
      const float3a dPdu = p1 - p0;
      const float3a dPdv = p2 - p0;

      // Combine to obtain the rate of change of P over s and t: dPds and dPdt.
      const float jacobianDeterminant = dSTdu.x * dSTdv.y - dSTdu.y * dSTdv.x;
      float3a     dPds                = dSTdv.y * dPdu - dSTdu.y * dPdv;
      float3a     dPdt                = dSTdu.x * dPdv - dSTdv.x * dPdu;

      bool        orientation = (jacobianDeterminant > 0);
      const float absArea     = fabsf(jacobianDeterminant);
      const float lenS        = length(dPds);
      const float lenT        = length(dPdt);
      if(absArea > FLT_MIN)
      {
        const float flip = (orientation ? 1.0f : -1.0f);
        if(lenS > FLT_MIN)
          dPds *= flip / lenS;
        if(lenT > FLT_MIN)
          dPdt *= flip / lenT;

        // Use majorant of s and t to determine if we can accumulate this contribution
        float magS = lenS / absArea;
        float magT = lenT / absArea;
        if(magS < FLT_MIN || magT < FLT_MIN)
        {
          auto resetTangent = [&](uint32_t tangentIndex) {
            if(!args.tangentInit[tangentIndex])
            {  //< initialize
              args.tangentInit[tangentIndex] = true;
              args.tangentU[tangentIndex]    = float4(0, 0, 0, 0);
              args.tangentV[tangentIndex]    = float3(0, 0, 0);
            }
          };

          resetTangent(f0);
          resetTangent(f1);
          resetTangent(f2);
          continue;
        }
      }
      else if(lenS > FLT_MIN || lenT > FLT_MIN)
      {
        // At least one of the derivatives is zero, It may be that the triangle has zero area,
        // which is not a big problem as the triangle will not be hit by any ray. But it can also
        // happen if the texture coordinates have overlapping values, in which case we can try to
        // salvage if we have 1 good derivative.
        float3a n = cross(dPdu, dPdv);

        if(lenS > lenT)
        {
          dPds /= lenS;
          dPdt = normalize(cross(n, dPds));
        }
        else
        {
          dPdt /= lenT;
          dPds = normalize(cross(dPdt, n));
        }
      }

      if(weightMode == k_weightMode_angle)  //< Template predicated: this branch is optimized out.
      {
        // Use the angles at the triangle corner to weight the contribution of each corner. This is the method
        // used by mikktspace algorithm
        auto accumulateWeightedTangents = [&](uint32_t tangentIndex, const float3a& n, const float3a& d1,
                                              const float3a& d2, const float3a& dPds, const float3a& dPdt) {
          float3a e0 = d1;
          float3a e1 = d2;

          // Projected tangents. Need normalized vectors to calculate the angle in between
          e0 = projectToPlane(e0, n);
          e0 = normalize(e0);
          e1 = projectToPlane(e1, n);
          e1 = normalize(e1);

          // Weigh contribution by the angle between the two edge vectors
          float costTheta = dot(e0, e1);
          costTheta       = (costTheta > 1.0f ? 1.0f : (costTheta < -1.0f ? -1.0f : costTheta));  // clamp [-1, 1]
          float theta     = acosf(costTheta);

          // Projected tangents, and apply scaling factor
          float3a pOs = projectToPlane(dPds, n) * theta;
          float3a pOt = projectToPlane(dPdt, n) * theta;

          if(!args.tangentInit[tangentIndex])
          {  //< initialize
            args.tangentInit[tangentIndex] = true;
            args.tangentU[tangentIndex]    = float4(pOs.x, pOs.y, pOs.z, 0.0f);
            args.tangentV[tangentIndex]    = float3(pOt.x, pOt.y, pOt.z);
          }
          else
          {  //< accumulate
            args.tangentU[tangentIndex] += float4(pOs.x, pOs.y, pOs.z, 0.0f);
            args.tangentV[tangentIndex] += float3(pOt.x, pOt.y, pOt.z);
          }
        };

        float3a d3 = p2 - p1;
        accumulateWeightedTangents(f0, n0, dPdu, dPdv, dPds, dPdt);
        accumulateWeightedTangents(f1, n1, d3, dPdu * (-1.0f), dPds, dPdt);
        accumulateWeightedTangents(f2, n2, dPdv * (-1.0f), d3 * (-1.0f), dPds, dPdt);
      }
      else
      {
        // Use the triangle area to weight the contribution of each corner.
        auto accumulateWeightedTangents = [&](uint32_t tangentIndex, const float3a& n, float area, const float3a& dPds,
                                              const float3a& dPdt) {
          // Projected tangents, and apply scaling factor
          float3a pOs = projectToPlane(dPds, n) * area;
          float3a pOt = projectToPlane(dPdt, n) * area;

          if(!args.tangentInit[tangentIndex])
          {  //< initialize
            args.tangentInit[tangentIndex] = true;
            args.tangentU[tangentIndex]    = float4(pOs.x, pOs.y, pOs.z, 0.0f);
            args.tangentV[tangentIndex]    = float3(pOt.x, pOt.y, pOt.z);
          }
          else
          {  //< accumulate
            args.tangentU[tangentIndex] += float4(pOs.x, pOs.y, pOs.z, 0.0f);
            args.tangentV[tangentIndex] += float3(pOt.x, pOt.y, pOt.z);
          }
        };

        float area = 0.5f * length(cross(dPdu, dPdv));  //< triangle area
        accumulateWeightedTangents(f0, n0, area, dPds, dPdt);
        accumulateWeightedTangents(f1, n1, area, dPds, dPdt);
        accumulateWeightedTangents(f2, n2, area, dPds, dPdt);
      }
    }
  }
}

#define OUTPUT_BITANGENTS 0
#define NORMALIZE_OUTPUTS 1

// Each facevarying record represetns a slice of a manifold. In a close surface the combination of all records
// describes the whole manifold (radially). Here we attempt at clustering records into shells. The logic is
// based on the observation that it should be safe to merge records that shares the same normal and texcoords.
// It is the same criteria that is use to weld topology elsewhere in the code. However, this is not enough,
// because texture seams in periodic surfaces are supposed to be smooth. We form shells by looking at normals
// + texcoords, and normals alone. The first accounts for texture seams, the second merges more and not always
// what it should. We measure how much skew in the tangents is accumulated in the second compared to the first,
// we decide if that is the smooth result we want. With periodic texture seams, one tangend tends to line up well
// while the other may have more skew. When both are skewed it is a sign these are not ment to be smooth.
// One last corner case to handle: a sphere apex vertex may have a single tex coord, and one normal, yet not a
// single tangent frame is valid for a singularity. That can be spotted by looking at how much the averaged tangents
// canceled out in the sum of the fanning out values.
void combine(uint64_t /*index*/, uint32_t /* unused */, void* ud)
{
  TANGENTS_PROFILE_ZONE(kProfilerMask, "generateTangents_combiningThread");
  Args&           args               = *static_cast<Args*>(ud);
  constexpr int   k_chunkSize        = 1024;
  constexpr float k_qualityThreshold = 0.999848f;  //< Aproximatively cos(1 degrees)
  // constexpr float k_qualityThreshold   = 0.996f; //< Aproximatively cos(5 degrees)
  constexpr float k_rejectionThreshold = 0.17f;  //< Aproximatively cos(80 degrees)

  struct Record
  {
    uint32_t index{};             //< the facevarying index
    uint32_t count{};             //< how many records are accumulated to contribute to this
    float3a  n{};                 //< the facevarying normal (not normalized)
    float2   tx{};                //< the facevarying texcoord
    float3a  tanU{}, tanV{};      //< tangets (either from the facevarying slice or accumulated)
    float    lenU{}, lenV{};      //< accumulated tangent lenght (different that length of accumulatd tanget!)
    uint32_t nIndex{}, tIndex{};  //< for the faevarying records, which shell do they belong to.
    uint32_t xIndex{};

    void operator+=(const Record& other)
    {
      tanU += other.tanU;
      tanV += other.tanV;
      lenU += other.lenU;
      lenV += other.lenV;
      count += other.count;
    }
  };
  std::vector<Record> scratchSpace(args.maxValence * 4);
  Record*             entries = scratchSpace.data();  //< facevarying records one point
  Record* shellsT = entries + args.maxValence;        //< shells where texture coords and normals are safe to merge
  Record* shellsN = shellsT + args.maxValence;        //< shells where only normals are safe to merge
  Record* shellsX = shellsN + args.maxValence;        //< shells where we detect singularities (tangents canceling out)

  while(true)
  {
    // Self-managed work stealing: work is partitioned between a number of threads, but then each thread
    // gets to advance at its own pace; due to prehemption, logical threads etc... threads rarely have
    // identical throughput. This strategy is on the line of CUDA persistent threads.
    int index = args.workload++;
    int start = index * k_chunkSize;
    int end   = std::min<int>(args.numVarying, (index + 1) * k_chunkSize);
    if(start >= (int)args.numVarying)
      break;

    // Loop over vertices, each entry is a point in the topology
    for(int entry = start; entry < end; ++entry)
    {
      const uint32_t offset  = args.verticesOffset[entry];
      const uint32_t valence = args.verticesValence[entry];
      if(valence == 0)
      {
        continue;
      }

      // For each point, loop over the facevarying slices of the surface manifold.
      // Here accumulate the tangents in each each slice to form shells.
      uint32_t numShellsT = 0, numShellsN = 0, numShellsX = 0;
      for(uint32_t i = 0; i < valence; ++i)
      {
        Record record;
        record.index = args.verticesLists[offset + i];
        record.count = 1;
        record.n     = args.inNormal[args.facevaryingN ? record.index : entry];
        record.tx    = args.inUvs[args.facevaryingTx ? record.index : entry];
        record.tanU  = float3(args.tangentU[record.index]);
        record.tanV  = args.tangentV[record.index];
        record.lenU  = length(record.tanU);
        record.lenV  = length(record.tanV);

        constexpr uint32_t k_notFound = 0xffffffffu;
        // Prouce shells acording to normals and texcoord
        {
          uint32_t candidateIndex = k_notFound;
          for(uint32_t j = 0; j < numShellsT; ++j)
          {
            Record& shell = shellsT[j];
            if(approximatelySame(record.n, shell.n) && approximatelySamePeriodic(record.tx, shell.tx))
            {
              float2 cosTheta{dot(shell.tanU / shell.lenU, record.tanU / record.lenU),
                              dot(shell.tanV / shell.lenV, record.tanV / record.lenV)};

              // Reject if tangents are too far apart to merge, don't want to merge vectors that
              // are pointing in opposite directions.
              if(reduce_min(cosTheta) <= -0.75f)
                continue;

              // Accumulate
              shell += record;
              candidateIndex = j;
              break;
            }
          }

          if(candidateIndex == k_notFound)
          {  //< Initialize
            candidateIndex          = numShellsT++;
            shellsT[candidateIndex] = record;
          }
          record.tIndex = candidateIndex;
        }
        // Prouce shells acording to normals alone, these may be fewer or equal to those above.
        {
          uint32_t candidateIndex = k_notFound;
          for(uint32_t j = 0; j < numShellsN; ++j)
          {
            Record& shell = shellsN[j];
            if(approximatelySame(record.n, shell.n))
            {
              float2 cosTheta{dot(shell.tanU / shell.lenU, record.tanU / record.lenU),
                              dot(shell.tanV / shell.lenV, record.tanV / record.lenV)};

              // Reject if tangents are too far apart to merge, don't want to merge vectors that
              // are pointing in opposite directions.
              if(reduce_min(cosTheta) <= -0.75f)
                continue;

              // Accumulate
              shell += record;
              candidateIndex = j;
              break;
            }
          }

          if(candidateIndex == k_notFound)
          {  //< Initialize
            candidateIndex          = numShellsN++;
            shellsN[candidateIndex] = record;
          }
          record.nIndex = candidateIndex;
        }
        // Produce shells acording to normals and texcoord without apply any rejection. The shell from
        // by this step may contain singularities, for example the apex of a sphere where tangents may
        // cancel out if not taken care of. We rely on this shell to filter out these singularities.
        {
          uint32_t candidateIndex = k_notFound;
          for(uint32_t j = 0; j < numShellsX; ++j)
          {
            Record& shell = shellsX[j];
            if(approximatelySame(record.n, shell.n) && approximatelySamePeriodic(record.tx, shell.tx))
            {
              // Accumulate
              shell += record;
              candidateIndex = j;
              break;
            }
          }

          if(candidateIndex == k_notFound)
          {  //< Initialize
            candidateIndex          = numShellsX++;
            shellsX[candidateIndex] = record;
          }
          record.xIndex = candidateIndex;
        }
        entries[i] = record;
      }

      // Precompute shell tangent length ratio
      //uint32_t numShellsT = 0, numShellsN = 0, numShellsX = 0;
      for(uint32_t i = 0; i < numShellsT; ++i)
      {
        shellsT[i].lenU = length2(shellsT[i].tanU) / sqr(shellsT[i].lenU);
        shellsT[i].lenV = length2(shellsT[i].tanV) / sqr(shellsT[i].lenV);
      }
      for(uint32_t i = 0; i < numShellsN; ++i)
      {
        shellsN[i].lenU = length2(shellsN[i].tanU) / sqr(shellsN[i].lenU);
        shellsN[i].lenV = length2(shellsN[i].tanV) / sqr(shellsN[i].lenV);
      }
      for(uint32_t i = 0; i < numShellsX; ++i)
      {
        shellsX[i].lenU = length2(shellsX[i].tanU) / sqr(shellsX[i].lenU);
        shellsX[i].lenV = length2(shellsX[i].tanV) / sqr(shellsX[i].lenV);
      }

      // Last loop over valence to decide for each facevarying corner which tangent we are going to use
      for(uint32_t i = 0; i < valence; ++i)
      {
        Record record = entries[i];
        // float flip = (handedness[record.index] ? 1.0f : -1.0f);

        const Record& shellT = shellsT[record.tIndex];
        const Record& shellN = shellsN[record.nIndex];
        const Record& shellX = shellsX[record.xIndex];

        // Note: for performance we carry forward and compare squared values.
        bool singularity = false;
        {
          // Compare the length of the mean vector, with the summed  vector lengths that contributed to
          // the mean. The ratio must be 1 or lower.  The closer to one, the higher the confidence that
          // merging facevarying records is the right thing to do.  In other words, the tangents across
          // the corners were already in a consisten direction.  Small ratios means tangents are faning
          // out.  Sometimes  the faning happens  in one direction, say tangentU,  while tangentV stays
          // consistent; so, here we take the maximum of the ratios as a measure of quality of merge,
          // and the min of the ratios as a measure to check for cancellation.
          //float2 lenRatio{ length2(shellX.tanU) / sqr(shellX.lenU), length2(shellX.tanV) / sqr(shellX.lenV) };
          float2 lenRatio{shellX.lenU, shellX.lenV};
          singularity = reduce_min(lenRatio) < sqr(k_rejectionThreshold);
        }
        bool merged = false;
        if((shellT.count > 1 || shellN.count > 1) && !singularity)
        {
          // We have two shells to choose from: the merged shell (normal and texture coordinate), which
          // may produce tangent discontinuities at the texture seams, and the one where only the normals
          // were considered.
          float2  lenRatio{shellT.lenU, shellT.lenV};
          float3a tanU = shellT.tanU, tanV = shellT.tanV;

          // Select best shell
          if(shellT.count != shellN.count)
          {
            float2 lenRatioN{shellN.lenU, shellN.lenV};

            // The length ratio is comparable to to the cosine of the wedge of vector (approximatively).
            // We prefer to smooth tangents across texture seams, if the faning is within some tight
            // treshold.
            if(reduce_max(lenRatioN) > reduce_max(lenRatio) * sqr(k_qualityThreshold)
               && reduce_min(lenRatioN) > sqr(k_rejectionThreshold))  //< check for cancellation
            {
              tanU = shellN.tanU, tanV = shellN.tanV;
              lenRatio = lenRatioN;
            }
          }

          // Check for cancellation. Accept the new tangent!
          if(reduce_min(lenRatio) > sqr(k_rejectionThreshold))
          {
            merged = true;
            // Flip note:    dot(N, cross(tanU, tanV))
            //            == det(N, tanU, tanV)
            //            == det(tanV, N, tanU)
            //            == dot(tanV, cross(N, tanU))
            // which matches the glTF definition.
            float flip = (dot(record.n, cross(tanU, tanV)) >= 0 ? 1.0f : -1.0f);
#if OUTPUT_BITANGENTS
            tanV = cross(record.n, tanU) * flip;
#endif
#if NORMALIZE_OUTPUTS
            tanU = normalize(tanU);
#endif

            // To keep this consitent with the previous algorithm, we produce a binormal that is
            // orthogonal the tangent and normal. It is clearly biased towards one of the tangents.
            args.tangentU[record.index] = float4(tanU.x, tanU.y, tanU.z, flip);
#if OUTPUT_BITANGENTS
            args.tangentV[record.index] = float3(tanV.x, tanV.y, tanV.z);
#endif
          }
        }

        if(!merged)
        {
          // No progress was made, keep the tangents, just orthogonalize tangentV.
          float flip = (dot(record.n, cross(record.tanU, record.tanV)) >= 0 ? 1.0f : -1.0f);
#if OUTPUT_BITANGENTS
          float3a tanV = cross(record.n, record.tanU) * flip;
#endif
#if NORMALIZE_OUTPUTS
          record.tanU = normalize(record.tanU);
#endif

          args.tangentU[record.index] = float4(record.tanU.x, record.tanU.y, record.tanU.z, flip);
#if OUTPUT_BITANGENTS
          args.tangentV[record.index] = float3(tanV.x, tanV.y, tanV.z);
#endif
        }
      }
    }
  }
};

// Tangent generation algorithm. This is a new method, designed my Max Liani from first principles, and compared to
// Mikktspace to validate the result is equivalent or better. Tangents are generated over the base arbitrary mesh
// topology. It relies on triangulated indices to not make any assumption on how the mesh will eventually be divided
// to triangles, however the algorithm produces a number of tangent frames matching facevarying quantities over the
// input topology. The algorithm work in three steps:
// 1. Produce a set of discontinuous tangent frames facevarying fragments: dPds, dPdt. These are first order derivs
//    for how position changes in respect to the texture coord (s, t). Each fragement is projected to their normal
//    plane.
// 2. Prepare an adjacency map by which we can easily loop over each facevarying fragment associated to a point in the
//    mesh.
// 3. Loop over points, and for each point over their facevarying fragments and combine them when possible, to produce
//    smooth tangets across faces. These smooth tangets are then made orthonormal, in respect to tangentU.
void createLianiTangents(Context         context,
                         const uint32_t* varyingIndices,
                         const uint32_t* facevaryingIndices,  //< optional
                         const uint32_t* uniformIndices,      //< optional

                         uint32_t numVarying,
                         uint32_t numFacevarying,
                         uint32_t numTriangles,

                         const nvmath::vec3f* inNormal,
                         const nvmath::vec3f* inPosition,
                         const nvmath::vec2f* inUvs,

                         const bool facevaryingN,
                         const bool facevaryingTx,

                         std::vector<uint32_t>& adjacencyMap,

                         // Results:
                         nvmath::vec4f* tangent)
{
  // clang-format off

    // It is much faster to initialize tangents by letting each thread take care of their chunk of memory. We use a
    // byte flag to control where we initialize with tangentU[index] = x, vs accumulate with tangentU[index] += x.
    // We do have to initialize the flag though...
    std::vector<uint8_t> tangentInit(numFacevarying, 0);

    // To exactly match the results produced by Omniverse's implementation of
    // this algorithm, we must hold both tangents and bitangents instead of
    // only tangents + a bitangent sign (since tangentU and tangentV aren't
    // orthogonalized until the very end). At the moment, we hide the
    // bitangent buffer allocation inside the implementation, so we provide
    // the desired API.
    std::vector<nvmath::vec3f> bitangents(numFacevarying, nvmath::vec3f(0.0f));

    Args args{ varyingIndices,    facevaryingIndices, uniformIndices,    (int)numVarying, (int)numFacevarying,
               (int)numTriangles, inPosition,         inNormal,          inUvs,           facevaryingN,
               facevaryingTx,     tangent,            bitangents.data(), adjacencyMap,    tangentInit.data() };

    constexpr int k_chunkSize = 1024 * 16; //< thread chunking size
    {
        TANGENTS_PROFILE_ZONE(kProfilerMask, "generateTangents_prepareFacevarying");

        args.workload.store(0); //< Reset the workload before every parallel run.

        if (args.numTriangles > k_chunkSize)
        {
            // This workload scales well with threads schedule. Because of the self-managed work stealing, we push
            // to the scheduler a number of threads we cosider effcient to wake up, not how much work units we have.
            uint32_t numThreads = std::min<uint32_t>((args.numTriangles + k_chunkSize -1) / k_chunkSize,
                                                 std::thread::hardware_concurrency());
            micromesh::OpDistributeWork_input parallelInput{};
            parallelInput.pfnGenericSingleWorkload = prepareFacevarying<k_weightMode_angle>;
            parallelInput.userData = &args;
            parallelInput.batchSize = 1;
            [[maybe_unused]] micromesh::Result result = micromesh::micromeshOpDistributeWork(context->m_micromeshContext, &parallelInput, numThreads);
            assert(result == micromesh::Result::eSuccess);
        }
        else
        {
            // Nice and easy, and sequential...
            prepareFacevarying<k_weightMode_angle>(0, 0, &args);
        }
    }

    // Inform static analysis that adjacency information has been created:
    assert(args.verticesLists && args.verticesOffset && args.verticesValence);

    {
        TANGENTS_PROFILE_ZONE(kProfilerMask, "generateTangents_combining");
        args.workload.store(0);
        if (args.numVarying > k_chunkSize)
        {
            uint32_t numThreads = std::min<uint32_t>((args.numVarying + k_chunkSize -1) / k_chunkSize,
                                                 std::thread::hardware_concurrency());
            micromesh::OpDistributeWork_input parallelInput{};
            parallelInput.pfnGenericSingleWorkload = combine;
            parallelInput.userData = &args;
            parallelInput.batchSize = 1;
            [[maybe_unused]] micromesh::Result result = micromesh::micromeshOpDistributeWork(context->m_micromeshContext, &parallelInput, numThreads);
        }
        else
        {
            combine(0, 0, &args);
        }
    }
  // clang-format on
}
}  // namespace meshops