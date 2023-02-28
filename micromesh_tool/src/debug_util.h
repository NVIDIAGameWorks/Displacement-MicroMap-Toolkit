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

#ifdef _WIN32

#include <windows.h>

static int AbortReportHook(int reportType, char* message, int* returnValue)
{
  const char* typeStr;
  switch(reportType)
  {
    case _CRT_WARN:
      typeStr = "Warning";
      break;
    case _CRT_ERROR:
      typeStr = "Error";
      break;
    case _CRT_ASSERT:
      typeStr = "Assertion";
      break;
    default:
      typeStr = "<invalid report type>";
      break;
  }
  printf("Abort (%s): %s\n", typeStr, message);
  fflush(stdout);
  *returnValue = 1;
  return true;  // no popup!
}

inline void fixAbortOnWindows()
{
  // Disable assert popups on windows that can hang automated testing when no debugger is attached
  if(!IsDebuggerPresent())
  {
    _CrtSetReportHook(AbortReportHook);
  }
}

#endif  // _WIN32
