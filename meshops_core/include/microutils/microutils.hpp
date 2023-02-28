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

#include <bary/bary_core.h>
#include <micromesh/micromesh_types.h>
#include <micromesh/micromesh_utils.h>
#include <micromesh/micromesh_operations.h>
#include <vector>

// microutils is designed to work with micromesh data
// which is represented in bary files, hence `baryutils.hpp`
// dependency exists.
// We split away the functions dealing with compression
// in `microutils_compression.h`, which adds the dependency
// on `micromesh_displacement_compression`

namespace microutils {

//////////////////////////////////////////////////////////////////////////

micromesh::MessageCallbackInfo makeDefaultMessageCallback();

// extended regular ScopedOpContext by a default message callback
struct ScopedOpContextMsg : public micromesh::ScopedOpContext
{
public:
  ScopedOpContextMsg(uint32_t numThreads, micromesh::MessageCallbackInfo callbackInfo = makeDefaultMessageCallback())
      : micromesh::ScopedOpContext(numThreads, callbackInfo)
  {
  }

  // Disable copying
  ScopedOpContextMsg(const ScopedOpContextMsg& other) = delete;
  ScopedOpContextMsg& operator=(const ScopedOpContextMsg& other) = delete;
};

//////////////////////////////////////////////////////////////////////////

// returns MicromapType based on basic.valueInfo.valueFormat
micromesh::MicromapType micromapTypeFromBasic(const bary::BasicView& basic);

// These assume the returned micromap is compatible, otherwise asserts.
// Incompatibility can be caused by valueFormat mismatch and non-standard valueLayout.
// If basic pointers are not properly set the returning value & triangle pointers can be null
// or wrong and are intentionally not range checked. Use `baryBasicContentToMicromap` if you
// prefer safer option.
micromesh::Micromap           micromapFromBasicGroup(const bary::BasicView& basic, uint32_t groupIndex);
micromesh::MicromapPacked     micromapPackedFromBasicGroup(const bary::BasicView& basic, uint32_t groupIndex);
micromesh::MicromapCompressed micromapCompressedFromBasicGroup(const bary::BasicView& basic, uint32_t groupIndex);

// Setup accessor to data typically found in bary files and
// does some basic checks if result is plausible.
// We are using bary::BasicContent directly to make it a bit more generic usable,
// as bary::BasicContent can be extract from memory mapped file pointers as well.
// Warning this function changes const'ness of pointers within basic
// to possible non-const access in destinations.
bary::Result baryBasicViewToMicromap(const bary::BasicView& basic, uint32_t groupIndex, micromesh::MicromapGeneric& micromap);
bary::Result baryBasicViewToMinMaxs(const bary::BasicView& basic, uint32_t groupIndex, micromesh::ArrayInfo& arrayInfo);
bary::Result baryBasicViewToBlockFormatUsage(const bary::BasicView& basic, uint32_t groupIndex, micromesh::MicromapBlockFormatUsage& mapUsage);

//////////////////////////////////////////////////////////////////////////
bary::Format             getBaryFormat(micromesh::Format microFormat);
bary::ValueFrequency     getBaryFrequency(micromesh::Frequency microFrequency);
bary::ValueLayout        getBaryValueLayout(micromesh::StandardLayoutType microStandardLayout);
bary::BlockFormatDispC1  getBaryBlockFormatDispC1(micromesh::BlockFormatDispC1 microBlockFormat);
bary::BlockFormatOpaC1   getBaryBlockFormatOpaC1(micromesh::BlockFormatOpaC1 microBlockFormat);
bary::HistogramEntry     getBaryHistogramEntry(micromesh::BlockFormatUsage microBlockFormatUsage);
bary::MeshHistogramEntry getBaryMeshHistogramEntry(micromesh::BlockFormatUsage microBlockFormatUsage);

micromesh::Format             getMicromeshFormat(bary::Format baryFormat);
micromesh::Frequency          getMicromeshFrequncy(bary::ValueFrequency baryFrequency);
micromesh::StandardLayoutType getMicromeshLayoutType(bary::ValueLayout baryLayout);
micromesh::BlockFormatDispC1  getMicromeshBlockFormatDispC1(bary::BlockFormatDispC1 baryBlockFormat);
micromesh::BlockFormatOpaC1   getMicromeshBlockFormatOpaC1(bary::BlockFormatOpaC1 baryBlockFormat);
micromesh::BlockFormatUsage   getMicromeshBlockFormatUsage(bary::HistogramEntry baryHistoEntry);

}  // namespace microutils
