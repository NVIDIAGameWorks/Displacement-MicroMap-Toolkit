/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RASTER_ANISOTROPY
#define RASTER_ANISOTROPY

float anisotropyMetric(vec3 v0, vec3 v1, vec3 v2)
{
  // Compute edge vectors between vertices. We'll catch NaNs at the end.
  const vec3 e01 = normalize(v0 - v1);
  const vec3 e02 = normalize(v0 - v2);
  const vec3 e12 = normalize(v1 - v2);

  // Get cosines of angles. This doesn't depend on the triangle's winding order.
  const float c0 = abs(dot(e01, e02));  // <-e01, -e02>
  const float c1 = abs(dot(e01, e12));  // <e01, -e12>
  const float c2 = abs(dot(e02, e12));

  // const float bigM = acos(min(min(c0, c1), c2));
  // const float ltlM = acos(max(max(c0, c1), c2));

  // This denominator should always be at least 2pi/3.
  // const float e = (bigM - ltlM) / (bigM + ltlM);
  // Catch NaNs in case of undefined normalize behavior. If two vertices were
  // equal, the triangle's degenerate, so return 1.
  //return isnan(e) ? 1.0 : e;
  return smoothstep(0.85f, 1.0f, max(max(c0, c1), c2));
}

#endif
