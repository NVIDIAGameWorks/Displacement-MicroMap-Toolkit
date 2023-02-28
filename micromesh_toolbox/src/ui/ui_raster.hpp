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

#pragma once

class ToolboxViewer;
struct ViewerSettings;

///
/// Implementation of the UI setting dialog section of for Raster rendering
///
class UiRaster
{
  ToolboxViewer* _v;

public:
  UiRaster(ToolboxViewer* v);
  bool onUI(ViewerSettings& settings);
};
