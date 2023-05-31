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

#include "imgui.h"
#include "imgui_internal.h"
#include "nvvkhl/application.hpp"
#include "nvml_monitor.hpp"
#include "imgui_helper.h"
#include <numeric>
#include <type_traits>

namespace nvvkhl {

// Unsafe. Temporary workaround for ImGui_Extra
template <class EnumTo, EnumTo ToDefault, class EnumFrom, EnumFrom FromValue>
struct FallbackEnumCast
{
  using ToType   = std::underlying_type_t<EnumTo>;
  using FromType = std::underlying_type_t<EnumFrom>;
  static EnumTo value()
  {
    ToType result = ToDefault;
    if constexpr(FromValue <= std::numeric_limits<ToType>::max())
    {
      result = static_cast<ToType>(FromValue);
    }
    return static_cast<EnumTo>(result);
  }
};

//extern SampleAppLog g_logger;
struct ElementNvml : public nvvkhl::IAppElement
{
  explicit ElementNvml(bool show = false)
      : m_showWindow(show)
  {
#if defined(NVP_SUPPORTS_NVML)
    m_nvmlMonitor = std::make_unique<NvmlMonitor>();
#endif
    addSettingsHandler();
  }

  virtual ~ElementNvml() = default;

  void onUIRender() override
  {
#if defined(NVP_SUPPORTS_NVML)
    m_nvmlMonitor->refresh();
#endif
    if(!m_showWindow)
      return;

    ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
    ImGui::SetNextWindowSize({400, 200}, ImGuiCond_Appearing);
    ImGui::SetNextWindowBgAlpha(0.7F);
    if(ImGui::Begin("NVML Monitor", &m_showWindow))
    {
      guiGpuMeasures();
    }
    ImGui::End();
  }

  void onUIMenu() override
  {
    if(ImGui::BeginMenu("Help"))
    {
      ImGui::MenuItem("NVML Monitor", nullptr, &m_showWindow);
      ImGui::EndMenu();
    }
  }  // This is the menubar to create


  //--------------------------------------------------------------------------------------------------
  //
  //
  bool guiGpuMeasures()
  {
    static const std::vector<const char*> t{"", "KiB", "MiB", "GiB", "TiB"};

#if defined(NVP_SUPPORTS_NVML)
    if(m_nvmlMonitor->isValid() == false)
    {
      ImGui::Text("NVML wasn't loaded");
      return false;
    }

    int offset = m_nvmlMonitor->getOffset();

    for(uint32_t g = 0; g < m_nvmlMonitor->nbGpu(); g++)  // Number of gpu
    {
      const NvmlMonitor::GpuInfo& i = m_nvmlMonitor->getInfo(g);
      const NvmlMonitor::Measure& m = m_nvmlMonitor->getMeasures(g);
      char                        progtext[64];
      size_t                      level =
          std::min(static_cast<size_t>(log2(static_cast<double>(std::max(UINT64_C(1), i.max_mem))) / 10), t.size() - 1);
      double divider = static_cast<double>(UINT64_C(1) << (level * 10));
      sprintf(progtext, "%3.2f/%3.2f %s", static_cast<double>(m.last_memory) / divider,
              static_cast<double>(i.max_mem) / divider, t[level]);

      // Load
      ImGui::Text("GPU: %s", i.name.c_str());
      ImGuiH::PropertyEditor::begin();
      ImGuiH::PropertyEditor::entry("Load", [&] {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, (ImVec4)ImColor::HSV(0.3F, 0.5F, 0.5F));
        ImGui::ProgressBar(m.load[offset] / 100.F);
        ImGui::PopStyleColor();
        return false;
      });
      // Memory
      ImGuiH::PropertyEditor::entry("Memory", [&] {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, (ImVec4)ImColor::HSV(0.6F, 0.5F, 0.5F));
        float memUsage = static_cast<float>((m.last_memory * 1000) / i.max_mem) / 1000.0F;
        ImGui::ProgressBar(memUsage, ImVec2(-1.f, 0.f), progtext);
        ImGui::PopStyleColor();
        return false;
      });

      //ImGui::Unindent();
      ImGuiH::PropertyEditor::end();
    }

    // CPU - refreshing only every second and average the last 5 values
    static float  average      = 0;
    static double refresh_time = ImGui::GetTime();
    if(refresh_time < ImGui::GetTime() - 1)  // Create data at fixed 60 Hz rate for the demo
    {
      average = 0;
      for(int i = 0; i < 5; i++)
      {
        size_t sampleCount   = static_cast<int>(m_nvmlMonitor->getSysInfo().cpu.size());
        size_t values_offset = (offset - i + sampleCount) % sampleCount;
        average += m_nvmlMonitor->getSysInfo().cpu[values_offset];
      }
      average /= 5.0F;
      refresh_time = ImGui::GetTime();
    }

    ImGuiH::PropertyEditor::begin();
    ImGuiH::PropertyEditor::entry("CPU", [&] {
      ImGui::ProgressBar(average / 100.F);
      return false;
    });
    ImGuiH::PropertyEditor::end();

    // Display Graphs
    for(uint32_t g = 0; g < m_nvmlMonitor->nbGpu(); g++)  // Number of gpu
    {
      const NvmlMonitor::GpuInfo& i = m_nvmlMonitor->getInfo(g);
      const NvmlMonitor::Measure& m = m_nvmlMonitor->getMeasures(g);

      if(ImGui::TreeNode("Graph", "Graph: %s", i.name.c_str()))
      {
        ImGui::ImPlotMulti datas[2] = {};
        datas[0].plot_type = FallbackEnumCast<ImGuiPlotType, ImGuiPlotType_Lines, ImGui_Extra, ImGuiPlotType_Area>::value();
        datas[0].name          = "Load";
        datas[0].color         = ImColor(0.07f, 0.9f, 0.06f, 1.0f);
        datas[0].thickness     = 1.5;
        datas[0].data          = m.load.data();
        datas[0].values_count  = (int)m.load.size();
        datas[0].values_offset = offset + 1;
        datas[0].scale_min     = 0;
        datas[0].scale_max     = 100;

        datas[1].plot_type     = ImGuiPlotType_Histogram;
        datas[1].name          = "Mem (KiB)";
        datas[1].color         = ImColor(0.06f, 0.6f, 0.97f, 0.8f);
        datas[1].thickness     = 2.0;
        datas[1].data          = m.memoryKB.data();
        datas[1].values_count  = (int)m.memoryKB.size();
        datas[1].values_offset = offset + 1;
        datas[1].scale_min     = 0;
        datas[1].scale_max     = float(i.max_mem / 1024);


        std::string overlay = "Load: " + std::to_string((int)m.load[offset]) + " %";
        ImGui::PlotMultiEx("##NoName", 2, datas, overlay.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 100));
        ImGui::TreePop();
      }
    }
#else
    ImGui::Text("NVML wasn't loaded");
#endif
    return false;
  }


  // This goes in the .ini file and remember the state of the window [open/close]
  void addSettingsHandler()
  {
    // Persisting the window
    ImGuiSettingsHandler ini_handler{};
    ini_handler.TypeName   = "ElementNvml";
    ini_handler.TypeHash   = ImHashStr("ElementNvml");
    ini_handler.ClearAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    ini_handler.ApplyAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* { return (void*)1; };
    ini_handler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      ElementNvml* s = (ElementNvml*)handler->UserData;
      int          x;
      if(sscanf(line, "ShowLoader=%d", &x) == 1)
      {
        s->m_showWindow = (x == 1);
      }
    };
    ini_handler.WriteAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      ElementNvml* s = (ElementNvml*)handler->UserData;
      buf->appendf("[%s][State]\n", handler->TypeName);
      buf->appendf("ShowLoader=%d\n", s->m_showWindow ? 1 : 0);
      buf->appendf("\n");
    };
    ini_handler.UserData = this;
    ImGui::AddSettingsHandler(&ini_handler);
  }

private:
  bool m_showWindow{false};
#if defined(NVP_SUPPORTS_NVML)
  std::unique_ptr<NvmlMonitor> m_nvmlMonitor;
#endif
};


}  // namespace nvvkhl
