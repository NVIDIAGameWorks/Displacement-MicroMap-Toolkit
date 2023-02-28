/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "generate_importance_operator.hpp"
#include "meshops_internal/meshops_device_mesh.h"

using namespace meshops;

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsGenerateImportanceOperatorCreate(Context context, GenerateImportanceOperator* pOp)
{
  (*pOp) = new GenerateImportanceOperator_c;
  if((*pOp)->create(context))
  {
    return micromesh::Result::eSuccess;
  }
  else
  {
    return micromesh::Result::eFailure;
  }
}
MESHOPS_API void MESHOPS_CALL meshopsGenerateImportanceOperatorDestroy(Context context, GenerateImportanceOperator op)
{
  op->destroy(context);
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpGenerateImportance(Context                        context,
                                                                       GenerateImportanceOperator     op,
                                                                       size_t                         count,
                                                                       OpGenerateImportance_modified* inputs)
{
  if(op->generateImportance(context, count, inputs))
  {
    return micromesh::Result::eSuccess;
  }
  else
  {
    return micromesh::Result::eFailure;
  }
}