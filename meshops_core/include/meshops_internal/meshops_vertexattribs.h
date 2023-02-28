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

#include "meshops/meshops_api.h"
#include "meshops/meshops_types.h"

namespace meshops {

// Generate per-vertex directions by averaging the face normals adjacent to each vertex
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsGenerateVertexDirections(Context context, meshops::ResizableMeshView& meshView);

// Compute extent of meshView
MESHOPS_API float MESHOPS_CALL meshopsComputeMeshViewExtent(Context context, const meshops::MutableMeshView& meshView);

}  // namespace meshops