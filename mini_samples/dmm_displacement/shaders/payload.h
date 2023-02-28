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

#ifndef PAYLOAD_H
#define PAYLOAD_H

#ifdef __cplusplus
using vec3 = nvmath::vec3f;
#endif  // __cplusplus

#define MISS_DEPTH 1000

struct HitPayload
{
  vec3 color;
  vec3 weight;
  int  depth;
};

HitPayload initPayload()
{
  HitPayload p;
  p.color  = vec3(0, 0, 0);
  p.depth  = 0;
  p.weight = vec3(1);
  return p;
}

#endif  // PAYLOAD_H
