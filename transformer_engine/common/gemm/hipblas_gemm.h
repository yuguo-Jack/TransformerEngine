/*************************************************************************
 * Copyright (c) 2022-2024, S3000 qianyj. All rights reserved.
 ************************************************************************/

/*! \file hipblas_gemmn.h
 *  \brief Functions for blas instead blaslt in pure gemm
 */

#ifndef TRANSFORMER_ENGINE_COMMON_HIPBLAS_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_HIPBLAS_GEMM_H_

#include <hip/hip_runtime.h>
#ifdef USE_HIPBLASLT
#include <hipblas/hipblas.h>
#include <mutex>
#else
#include <rocblas/rocblas.h>
#endif
#include <stdexcept>
#include "../common_hip.h"
#include <iostream>


#ifdef USE_HIPBLASLT
class HipblasHandleManager {
public:
    HipblasHandleManager() {}

    ~HipblasHandleManager() {
        // Release all handles when the manager is destroyed
        for (auto& device_pair : handles_map_) {
            hipblasDestroy(device_pair.second);  // Only one handle per device
        }
    }

    // Get a handle for the given device (creates if necessary)
    hipblasHandle_t get(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if the handle for this device exists
        auto device_it = handles_map_.find(device_id);
        if (device_it != handles_map_.end()) {
            return device_it->second;
        }

        // Create a new handle for this device if it doesn't exist
        hipblasHandle_t handle;
        hipblasStatus_t status = hipblasCreate(&handle);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create HIPBLAS handle");
        }

        // Store the handle in the map for this device
        handles_map_[device_id] = handle;
        return handle;
    }

private:
    std::unordered_map<int, hipblasHandle_t> handles_map_;  // Map from device_id to hipblasHandle
    std::mutex mutex_;
};

namespace transformer_engine {
    void hipblas_gemm(const Tensor *inputA,
                 const Tensor *inputB,
                 Tensor *outputD,
                 const Tensor *inputBias,
                 Tensor *outputPreGelu,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 hipblasOperation_t transa,
                 hipblasOperation_t transb,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool accumulate,
                 bool use_split_accumulator,
                 int math_sm_count,
                 int m_split,
                 int n_split,
                 bool gemm_producer,
                 const Tensor *inputCounter,
                 hipStream_t stream);

    void hipblas_batchgemm(const Tensor *inputA,
                 const Tensor *inputB,
                 Tensor *outputD,
                 const Tensor *inputBias,
                 Tensor *outputPreGelu,
                 int m, int n, int k,
                 int lda, int ldb, int ldd,
                 hipblasOperation_t transa,
                 hipblasOperation_t transb,
                 bool grad,
                 void* workspace,
                 size_t workspaceSize,
                 bool accumulate,
                 bool use_split_accumulator,
                 int math_sm_count,
                 int m_split,
                 int n_split,
                 bool gemm_producer,
                 const Tensor *inputCounter,
                 int batch_count,
                 hipStream_t stream);
}
#else

class HipblasHandleManager {
public:
    HipblasHandleManager() : handle_(nullptr) {}

    ~HipblasHandleManager() {
        // Release the handle in the destructor to ensure cleanup when it's no longer needed
        if (handle_ != nullptr) {
            rocblas_destroy_handle(handle_);
        }
    }

    // Get a handle to make sure it's valid every time
    rocblas_handle get() {
        if (handle_ == nullptr) {
            createHandle();
        }

        // Check whether the handle is created successfully
        assert(handle_ != nullptr && "hipblasHandle should not be null after creation");
        return handle_;
    }

private:
    rocblas_handle handle_;

    // 
    void createHandle() {
        // A private method that creates a handle
        rocblas_status status = rocblas_create_handle(&handle_);
        if (status != rocblas_status_success) {
            // If initialization fails, an exception is thrown
            throw std::runtime_error("Failed to create HIPBLAS handle");
        }
    }

    // Copy construct and assignment operations are prohibited
    HipblasHandleManager(const HipblasHandleManager&) = delete;
    HipblasHandleManager& operator=(const HipblasHandleManager&) = delete;
};
#endif // #ifdef USE_HIPBLASLT
#endif // TRANSFORMER_ENGINE_COMMON_HIPBLAS_GEMM_H_