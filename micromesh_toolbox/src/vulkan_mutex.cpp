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

#include "vulkan_mutex.h"

static std::recursive_mutex m_vkQueueOrAllocatorMutex;

std::unique_lock<std::recursive_mutex> GetVkQueueOrAllocatorLock() noexcept
{
#pragma warning(push)
  // It looks like MSVC 16.11.13 doesn't quite figure out the lifetime of the
  // unique_lock here:
#pragma warning(disable : 26115)
  return std::unique_lock<std::recursive_mutex>(m_vkQueueOrAllocatorMutex);
#pragma warning(pop)
}