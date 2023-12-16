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
#include "nvmath/nvmath.h"

#include "shaders/device_host.h"
#include "hbao/hbao_pass.hpp"
#include "imgui.h"
#include "meshops/meshops_types.h"
#include "nvh/nvprint.hpp"

struct HbaoSettings
{
  bool               active{true};
  HbaoPass::Frame    frame{};
  HbaoPass::Settings settings{};
  float              radius{0.05f};
};

enum SceneVersion
{
  eReference = 0,
  eBase      = 1,
  eScratch   = 2
};

enum RasterPipelines : int
{
  eRasterPipelineSolid,
  eRasterPipelineBlend,
  eRasterPipelineWire,
  eRasterPipelineShell,
  eRasterPipelineVector,
  eRasterPipelineMicromeshSolid,
  eRasterPipelineMicromeshWire,
  eRasterPipelineHeightmapSolid,
  eRasterPipelineHeightmapWire,
  eRasterPipelineNum
};


// This is to handle the busy state of a tool
class ActivityStatus
{
public:
  enum State
  {
    eIdle,    // Nothing - Application can run
    eActive,  // Something is running
    eDone,    // The activity is done, but yet not Idle
  };

  void activate(const std::string& reason)
  {
    m_busyReason = reason;
    m_busy       = eActive;
    LOGI("Active: %s\n", reason.c_str());
  }
  void stop()
  {
    m_busy       = eDone;
    m_busyReason = "";
  }
  bool isBusy() const { return m_busy != eIdle; }
  bool updateState()  // This have to be called ONLY once, at the beginning of a frame
  {
    if(m_busy == eDone)
    {
      m_busy = eIdle;
      return true;
    }
    return false;
  }
  const std::string& status() const { return m_busyReason; }

private:
  State       m_busy = eIdle;  // UI blocker
  std::string m_busyReason;
};

struct ViewerSettings
{
  ViewerSettings()
  {
    // Default light
    lights.resize(1);
    lights[0] = nvvkhl_shaders::defaultLight();
  }

  enum EnvSystem
  {
    eSky,
    eHdr,
  };

  enum RenderSystem
  {
    ePathtracer,
    eRaster,
  };

  enum RenderViewSlot
  {
    eNone,
    eReference,
    eBase,
    eScratch,
    eNumSlots,
  };
#define NUMSCENES (ViewerSettings::RenderViewSlot::eNumSlots - 1)


  enum ColormapMode
  {
    eColormap_temperature,
    eColormap_viridis,
    eColormap_plasma,
    eColormap_magma,
    eColormap_inferno,
    eColormap_turbo,
    eColormap_batlow,
  };

  struct RenderView
  {
    RenderView()
    {
      slot  = eNone;
      baked = false;
    }
    RenderView(RenderViewSlot _slot, bool _baked)
    {
      slot  = _slot;
      baked = _baked;
    }
    RenderViewSlot slot;
    bool           baked;
  };

  int           maxFrames    = 200000;                 // Maximum number of frames for ray tracing
  int           maxSamples   = 1;                      // Number of samples in a single frame
  int           maxDepth     = 5;                      // Number of bouncing rays
  EnvSystem     envSystem    = EnvSystem::eSky;        // Background environment HDR or Sky
  RenderSystem  renderSystem = RenderSystem::eRaster;  // Rendering engine
  nvmath::vec4f envColor     = {1.F, 1.F, 1.F, 1.F};   // Environment color multiplier
  float         envRotation  = 0.F;                    // Rotating the environment in degrees

  RenderView             geometryView = {RenderViewSlot::eReference, false};
  RenderView             overlayView  = {RenderViewSlot::eNone, false};
  RenderView             shellView    = {RenderViewSlot::eNone, false};
  shaders::RenderShading shading      = shaders::RenderShading::eRenderShading_default;
  shaders::DebugMethod   debugMethod  = shaders::DebugMethod::eDbgMethod_none;
  ColormapMode           colormap     = ColormapMode::eColormap_temperature;

  // Override for RenderShading::eFaceted
  float metallic{0.2F};
  float roughness{0.4F};

  float vectorLength{1.0F};

  ImVec4 overlayColor = ImVec4(118.F / 255.F, 185.F / 255.F, 0, 1);

  bool forceDoubleSided = false;

  bool showAxis       = false;
  bool showStats      = false;
  bool showAdvancedUI = false;
  bool nonpipelineUI  = false;

  // Global tool options
  struct GlobalToolSettings
  {
    int32_t subdivLevel                 = 5;
    int32_t pretessellateBias           = 0;
    int32_t decimateRateFromSubdivLevel = 0;
  } tools;

  // Heightmaps
  int   heightmapSubdivLevel{HEIGHTMAP_MAX_SUBDIV_LEVEL};
  int   heightmapRTXSubdivLevel{5};
  float heightmapScale{1.0f};
  float heightmapOffset{0.0f};

  std::vector<nvvkhl_shaders::Light> lights;

  ActivityStatus activtyStatus;  // UI blocker

  HbaoSettings hbao;
};
