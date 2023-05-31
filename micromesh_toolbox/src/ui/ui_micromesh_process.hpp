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
#include "settings.hpp"
#include <future>


class ToolboxViewer;
class ToolboxScene;


///
/// Implementation of the UI setting dialog section of for Raster rendering
///
class UiMicromeshProcess
{
  ToolboxViewer*    m_toolboxViewer;
  std::future<bool> m_toolOperation;
  std::future<bool> m_loadingScene;

public:
  UiMicromeshProcess(ToolboxViewer* v)
      : m_toolboxViewer(v)
  {
  }
  bool onUI();

private:
  int  loadSaveDelLine(std::string name, std::unique_ptr<ToolboxScene>& tboxscene, ViewerSettings::RenderViewSlot view, bool& dispbaked);
  void runSourceToTarget();
  void attributesOperations(std::unique_ptr<ToolboxScene>& scene_base);
};

///
/// Implementation of the UI setting dialog section of for Raster rendering
///
class UiMicromeshProcessPipeline
{
  ToolboxViewer*    m_toolboxViewer;
  std::future<bool> m_toolOperation;
  std::future<bool> m_loadingScene;

public:
  UiMicromeshProcessPipeline(ToolboxViewer* v)
      : m_toolboxViewer(v)
  {
  }
  bool onUI();

private:
  void loadLine(std::string name, ViewerSettings::RenderViewSlot view);
  bool toolHeader(const char* name, bool& use);
  void attributesOperations(std::unique_ptr<ToolboxScene>& scene);
};
