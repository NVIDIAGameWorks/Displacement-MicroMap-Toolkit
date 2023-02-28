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


#ifndef UTILITY_H
#define UTILITY_H


#define IM_COL32_R_SHIFT 0
#define IM_COL32_G_SHIFT 8
#define IM_COL32_B_SHIFT 16
#define IM_COL32_A_SHIFT 24

vec4 intColorToVec4(int rgba)
{
  vec4  value;
  float sc = 1.0f / 255.0f;
  value.x  = float((rgba >> IM_COL32_R_SHIFT) & 0xFF) * sc;
  value.y  = float((rgba >> IM_COL32_G_SHIFT) & 0xFF) * sc;
  value.z  = float((rgba >> IM_COL32_B_SHIFT) & 0xFF) * sc;
  value.w  = float((rgba >> IM_COL32_A_SHIFT) & 0xFF) * sc;
  return value;
}


#endif