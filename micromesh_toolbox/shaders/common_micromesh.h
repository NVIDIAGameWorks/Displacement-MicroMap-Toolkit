
/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// this file is included by C++ and GLSL

#ifndef _COMMON_MICROMESH_H_
#define _COMMON_MICROMESH_H_

#define MICRO_GROUP_SIZE SUBGROUP_SIZE
#define MICRO_TRI_PER_TASK SUBGROUP_SIZE

// how many groups per flat mesh shader invocations
#define MICRO_FLAT_GROUPS 2

#endif