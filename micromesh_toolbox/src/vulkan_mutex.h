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

// This file contains ways to protect Vulkan resources that require external
// synchronization from write hazards on multiple threads. In particular,
// this defines a mutex that *must* be locked whenever using the viewer's
// GCT queue 0, or its nvvk::ResourceAllocator.
//
// Although the viewer ensures that tasks such as micromesh_gen and the
// remesher get their own Vulkan queues and have their own memory allocators,
// the viewer's window thread, scene-loading thread, and HDR-loading thread
// can all make allocations and submit work to queues. They currently share
// the GCT queue 0 and nvvk::ResourceAllocator. Vulkan requires queues to be
// externally synchronized; without a mutex, it's easy for multiple threads to
// try to submit work and wait for work on a queue at once, which breaks.
// Similarly, the NVVK allocators generally aren't thread-safe, even though
// vkAllocateMemory is (see nvpro_core issue #7 for more on that); if two
// threads use an NVVK allocator at once, the result is generally undefined.
//
// For more information, please see JIRA issue MICROSDK-181.

#include <mutex>

// This function must be called to get an exclusive lock whenever using the
// viewer's GCT queue 0, its nvvk::ResourceAllocator, or both.
//
// Here's some examples of how to use it. It returns a scoped lock:
// ```
// {
//   auto lock = GetQueueOrAllocatorLock();
//	 vkQueueSubmit(device.m_queueGCT, ...);
//   vkQueueWaitIdle(device.m_queueGCT);
// } // lock goes out of scope here; other threads can now obtain a lock
// ```
// If we used vkWaitForFences, we no longer have a dependency on the queue, so
// we can unblock other threads earlier:
// ```
// {
//   auto lock = GetQueueOrAllocatorLock();
//   vkQueueSubmit(device.m_queueGCT, ..., fence);
// } // lock goes out of scope here; other threads can now proceed
// vkWaitForFences(device, 1, &fence, ...);
// ```
// It should be used whenever using either or both resources:
// ```
// {
//   auto lock = GetQueueOrAllocatorLock();
//   m_alloc.createBuffer(...);
//   ...
//   vkQueueSubmit(device.m_queueGCT, ...);
//   vkQueueWaitIdle(device.m_queueGCT);
// }
// ```
// Sometimes NVVK functions implicitly use an allocator. Watch out for those
// cases!
// ```
// // RaytracingBuilderKHR's setup() stores a pointer to an allocator, but
// // doesn't access it itself:
// nvvk::RaytracingBuilderKHR raytracingBuilder;
// raytracingBuilder.setup(device, &m_alloc, 0);
// // Later operations require locking, though:
// {
//   auto lock = GetQueueOrAllocatorLock();
//   raytracingBuilder.buildBlas(...);
//   ...
//   raytracingBuilder.destroy();
// }
// ```
// If you ever need a resource lock in a constructor, here's how to do it:
// ```
// auto lock = GetQueueOrAllocatorLock();
// ObjectWhoseConstructorAllocates obj(m_alloc);
// lock.unlock(); // Other threads are now unblocked
// ```
// This is a recursive mutex, so you can lock it again if you already have
// ownership of it:
// ```
// void foo()
// {
//   auto lock = GetQueueOrAllocatorLock();
//   m_alloc.createAcceleration(...)
// }
//
// void bar()
// {
//   foo();
//   {
//     auto lock = GetQueueOrAllocatorLock();
//     vkQueueWaitIdle(device.m_queueGCT);
//     foo();
//   }
// }
// ```
// However, it's possible to deadlock if you lock, then create threads that
// also lock, like this:
// ```
// void foo()
// {
//   parallel_batches([](){ auto lock = GetQueueOrAllocatorLock(); ...}, ...);
// }
//
// auto lock = GetQueueOrAllocatorLock();
// foo();
// ```
// Also, note that it's only safe to call finalizeAndReleaseStaging once no
// existing command buffers reference any staging textures. As a result, locks'
// lifetimes will often need to wrap around the lifetimes of command buffers
// with data uploads.
//
// We currently use one mutex for both resources, to avoid deadlocks like this:
// ```
// Thread 1:                      Thread 2:
//   Lock on queue mutex            Lock on allocator mutex
//   Lock on allocator mutex        Lock on queue mutex
// ```
// However, a design based on std::shared_lock could handle this.
// Locking and unlocking a mutex should be around 50 clock cycles.
std::unique_lock<std::recursive_mutex> GetVkQueueOrAllocatorLock() noexcept;