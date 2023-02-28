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

#include <stdarg.h>

#include "nvh/nvprint.hpp"
#include "nvh/timesampler.hpp"

//--------------------------------------------------------------------------------------------------
// Print the time a function takes and indent nested functions
//
struct NestingScopedTimer
{
  explicit NestingScopedTimer(std::string str)
      : m_name(std::move(str))
  {
    LOGI("%s%s:\n", indent().c_str(), m_name.c_str());
    ++s_depth;
  }

  ~NestingScopedTimer()
  {
    --s_depth;
    LOGI("%s|-> (%.3f ms)\n", indent().c_str(), m_sw.elapsed());
  }

  // Comment indented comments
  template <typename... Args>
  void print(const char* fmt, Args... args)
  {
    LOGI("%s", indent().c_str());
    nvprintf(fmt, args...);
  }

  static std::string indent();

  std::string                m_name;
  nvh::Stopwatch             m_sw;
  static thread_local size_t s_depth;
};
