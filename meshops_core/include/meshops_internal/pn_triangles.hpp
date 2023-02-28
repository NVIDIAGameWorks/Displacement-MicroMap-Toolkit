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

#include "nvmath/nvmath.h"

using nvmath::vec3f;

// https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
// https://ogldev.org/www/tutorial31/tutorial31.html
// https://www.nvidia.com/content/PDF/GDC2011/John_McDonald.pdf
class PNTriangles
{
public:
  PNTriangles()  = default;
  ~PNTriangles() = default;

  vec3f projectToPlane(vec3f p, vec3f plane, vec3f planeNormal)
  {
    vec3f delta          = p - plane;
    vec3f deltaProjected = dot(delta, planeNormal) * planeNormal;
    return (p - deltaProjected);
  }

  PNTriangles(const vec3f& v0, const vec3f& v1, const vec3f& v2, const vec3f& n0, const vec3f& n1, const vec3f& n2)
  {
    // Naming conventions:
    // (this code)   (Vlachos paper) (ogldev.org)
    // v0---v2 v+    P1---P3 v+      v2---v1 v+
    // |   /         |   /           |   /
    // |  /          |  /            |  /
    // | /           | /             | /
    // v1            P2              v0
    // u+            u+              u+
    //
    // Three-digit indices correspond to powers of W, U, and V, in that order.
    // v0, v1, and v2 are also in W, U, V order.
    // Other than that, the code most closely matches the ogldev.org tutorial.

    vB030 = v1;
    vB003 = v2;
    vB300 = v0;

    vec3f edgeB300 = vB003 - vB030;
    vec3f edgeB030 = vB300 - vB003;
    vec3f edgeB003 = vB030 - vB300;

    vB021 = vB030 + edgeB300 / 3.0f;
    vB012 = vB030 + edgeB300 * 2.0f / 3.0f;
    vB102 = vB003 + edgeB030 / 3.0f;
    vB201 = vB003 + edgeB030 * 2.0f / 3.0f;
    vB210 = vB300 + edgeB003 / 3.0f;
    vB120 = vB300 + edgeB003 * 2.0f / 3.0f;

    vB021 = projectToPlane(vB021, vB030, n1);
    vB012 = projectToPlane(vB012, vB003, n2);
    vB102 = projectToPlane(vB102, vB003, n2);
    vB201 = projectToPlane(vB201, vB300, n0);
    vB210 = projectToPlane(vB210, vB300, n0);
    vB120 = projectToPlane(vB120, vB030, n1);

    vec3f vCenter = (vB003 + vB030 + vB300) / 3.0f;
    vB111         = (vB021 + vB012 + vB102 + vB201 + vB210 + vB120) / 6.0f;
    vB111 += (vB111 - vCenter) / 2.0f;

    // Quadratic normal interpolation from Vlachos paper. Normalization is
    // skipped as normals can be used as direction vectors with magnitude.
    float v01 = 2.0f * dot(edgeB003, n0 + n1) / dot(edgeB003, edgeB003);
    float v12 = 2.0f * dot(edgeB300, n1 + n2) / dot(edgeB300, edgeB300);
    float v20 = 2.0f * dot(edgeB030, n2 + n0) / dot(edgeB030, edgeB030);
    vN200     = n0;
    vN020     = n1;
    vN002     = n2;
    vN110     = n0 + n1 - v01 * edgeB003;
    vN011     = n1 + n2 - v12 * edgeB300;
    vN101     = n2 + n0 - v20 * edgeB030;
  }

  vec3f position(const vec3f& wuv) const
  {
    float w = wuv.x;
    float u = wuv.y;
    float v = wuv.z;

    float uPow3 = powf(u, 3);
    float vPow3 = powf(v, 3);
    float wPow3 = powf(w, 3);
    float uPow2 = powf(u, 2);
    float vPow2 = powf(v, 2);
    float wPow2 = powf(w, 2);

    return vec3f{vB300 * wPow3 +             //
                 vB030 * uPow3 +             //
                 vB003 * vPow3 +             //
                 vB210 * 3.0f * wPow2 * u +  //
                 vB120 * 3.0f * w * uPow2 +  //
                 vB201 * 3.0f * wPow2 * v +  //
                 vB021 * 3.0f * uPow2 * v +  //
                 vB102 * 3.0f * w * vPow2 +  //
                 vB012 * 3.0f * u * vPow2 +  //
                 vB111 * 6.0f * w * u * v};
  }

  vec3f normal(const vec3f& wuv) const
  {
    float w = wuv.x;
    float u = wuv.y;
    float v = wuv.z;
    return vec3f{vN200 * w * w +  //
                 vN020 * u * u +  //
                 vN002 * v * v +  //
                 vN110 * w * u +  //
                 vN011 * u * v +  //
                 vN101 * w * v};
  }

  float apply(const vec3f& pos, const vec3f& dir, const vec3f& wuv) const
  {
    vec3f pnPos = position(wuv);
    return dot(pnPos - pos, dir) / dot(dir, dir);
  }

private:
  vec3f vB021;
  vec3f vB012;
  vec3f vB102;
  vec3f vB201;
  vec3f vB210;
  vec3f vB120;

  vec3f vB111;
  vec3f vB030;
  vec3f vB003;
  vec3f vB300;

  vec3f vN200;
  vec3f vN020;
  vec3f vN002;
  vec3f vN110;
  vec3f vN011;
  vec3f vN101;
};
