/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_CUDNN_UTILS_H_
#define TRANSFORMER_ENGINE_CUDNN_UTILS_H_

#ifndef __HIP_PLATFORM_AMD__
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>
#endif

#include <cstdint>
#include <mutex>

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
  
#ifndef __HIP_PLATFORM_AMD__
cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t);

cudnn_frontend::DataType_t get_cudnn_fe_dtype(const transformer_engine::DType t);

class cudnnExecutionPlanManager {
 public:
  static cudnnExecutionPlanManager &Instance() {
    static thread_local cudnnExecutionPlanManager instance;
    return instance;
  }

  cudnnHandle_t GetCudnnHandle() {
    static thread_local std::once_flag flag;
    std::call_once(flag, [&] { cudnnCreate(&handle_); });
    return handle_;
  }

  ~cudnnExecutionPlanManager() {}

 private:
  cudnnHandle_t handle_ = nullptr;
};
#endif

}  // namespace transformer_engine

#endif
