/*************************************************************************
 * Copyright (c) 2022-2024, S3000 qianyj. All rights reserved.
 ************************************************************************/

#include <hip/hip_runtime.h>
#include "hipblas_gemm.h"
#include "../common_hip.h"
#include "../util/logging.h"

namespace {

hipblasDatatype_t get_hip_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return HIPBLAS_R_16F;
    case DType::kFloat32:
      return HIPBLAS_R_32F;
    case DType::kBFloat16:
      return HIPBLAS_R_16B;     
    default:
      NVTE_ERROR("Invalid type");
  }
}

}  // namespace

// Define a static handle manager
static HipblasHandleManager handleManager;

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
                 hipStream_t stream) {
    // Use static handles
    int device_id;
    hipGetDevice(&device_id);
    hipblasHandle_t handle = handleManager.get(device_id);
    void *A = inputA->data.dptr;
    // void *A_scale_inverse = inputA->scale_inv.dptr;
    void *B = inputB->data.dptr;
    // void *B_scale_inverse = inputB->scale_inv.dptr;
    void *C = outputD->data.dptr;
    void *D = outputD->data.dptr;


    // Select the calculation accuracy
    hipblasDatatype_t A_type = get_hip_dtype(inputA->data.dtype);
    hipblasDatatype_t B_type = get_hip_dtype(inputB->data.dtype);
    hipblasDatatype_t D_type = get_hip_dtype(outputD->data.dtype);
    hipblasDatatype_t computeType = HIPBLAS_R_32F; // default acc is float32

    // setting computetype
    // if (/* condition for mixed precision */) {
    //     computeType = HIPBLAS_R_16F; // 
    // }
    // hipblasComputeType_t gemm_compute_type = HIPBLAS_COMPUTE_32F;
    // const char *env_tf32 = std::getenv("NVTE_BLASLT_TF32");
    // if (env_tf32 != nullptr && env_tf32[0] == '1') {
    // if (A_type == HIPBLAS_R_32F && B_type == HIPBLAS_R_32F && D_type == HIPBLAS_R_32F) {
    //     gemm_compute_type = HIPBLAS_COMPUTE_32F_FAST_TF32;
    // }

    float one = 1.0f;
    float zero = 0.0f;
    float beta = accumulate ? one : zero;
  
    hipblasSetStream(handle, stream);
    // execute multiply
    hipblasStatus_t status = hipblasGemmEx(
                                       handle,
                                       transa,   // transa
                                       transb,   // transb
                                       m,
                                       n,
                                       k,
                                       static_cast<const void*>(&one), 
                                       A,
                                       A_type,
                                       lda,
                                       B,
                                       B_type,
                                       ldb,
                                       static_cast<const void*>(&beta), 
                                       D,
                                       D_type,
                                       ldd,
                                       computeType,
                                       HIPBLAS_GEMM_DEFAULT);

    if (status != HIPBLAS_STATUS_SUCCESS) {
        NVTE_ERROR("hipblasGemmEx execution failed");
    }
}

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
                 hipStream_t stream) {
    // Use static handles
    int device_id;
    hipGetDevice(&device_id);
    hipblasHandle_t handle = handleManager.get(device_id);
    void *A = inputA->data.dptr;
    // void *A_scale_inverse = inputA->scale_inv.dptr;
    void *B = inputB->data.dptr;
    // void *B_scale_inverse = inputB->scale_inv.dptr;
    void *C = outputD->data.dptr;
    void *D = outputD->data.dptr;

    // Select the calculation accuracy
    hipblasDatatype_t A_type = get_hip_dtype(inputA->data.dtype);
    hipblasDatatype_t B_type = get_hip_dtype(inputB->data.dtype);
    hipblasDatatype_t D_type = get_hip_dtype(outputD->data.dtype);
    hipblasDatatype_t computeType = HIPBLAS_R_32F; // default acc is float32

    float one = 1.0f;
    float zero = 0.0f;
    float beta = accumulate ? one : zero;
  
    hipblasSetStream(handle, stream);
    // execute multiply
    // calculate stride

    const long long int strideA = m*k;
    const long long int strideB = k*n;
    const long long int strideD = m*n;
    hipblasStatus_t status = hipblasGemmStridedBatchedEx(
                                       handle,
                                       transa,   // transa
                                       transb,   // transb
                                       m,
                                       n,
                                       k,
                                       static_cast<const void*>(&one), 
                                       A,
                                       A_type,
                                       lda,
                                       strideA,
                                       B,
                                       B_type,
                                       ldb,
                                       strideB,
                                       static_cast<const void*>(&beta), 
                                       D,
                                       D_type,
                                       ldd,
                                       strideD,
                                       batch_count,
                                       computeType,
                                       HIPBLAS_GEMM_DEFAULT);
  
    if (status != HIPBLAS_STATUS_SUCCESS) {
        NVTE_ERROR("hipblasGemmEx execution failed");
    }
}

}  // namespace transformer_engine