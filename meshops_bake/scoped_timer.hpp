//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <nvh/nvprint.hpp>
#include <nvh/timesampler.hpp>
#include <string>

// Simple scoped timer that prints a label and logs the duration
// between its construction and destruction.
struct ScopedTimer
{
  ScopedTimer(const std::string& str) { LOGI("%s", str.c_str()); }
  ~ScopedTimer() { LOGI(" %.3f ms\n", sw_.elapsed()); }
  nvh::Stopwatch sw_;
};

struct NestingScopedTimer
{
  NestingScopedTimer(const std::string& str)
      : name(str)
  {
    LOGI("%sBegin %s:\n", indent().c_str(), name.c_str());
    ++s_depth;
  }
  ~NestingScopedTimer()
  {
    --s_depth;
    LOGI("%sEnd %s (%.3f ms)\n", indent().c_str(), name.c_str(), sw_.elapsed());
  }
  std::string                indent() { return std::string(s_depth * 2, ' '); }
  std::string                name;
  nvh::Stopwatch             sw_;
  static thread_local size_t s_depth;
};
