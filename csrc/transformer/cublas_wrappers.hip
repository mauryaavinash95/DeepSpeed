// !!! This is a file automatically generated by hipify!!!
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cublas_wrappers_hip.h"

#ifdef __HIP_PLATFORM_HCC__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   rocblas_gemm_algo algo)
#else
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   cublasGemmAlgo_t algo)
#endif
{
#ifdef __HIP_PLATFORM_HCC__
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_datatype_f32_r,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_datatype_f32_r,
                                            (transb == rocblas_operation_none) ? k : n,
                                            (const void*)beta,
                                            C,
                                            rocblas_datatype_f32_r,
                                            m,
                                            C,
                                            rocblas_datatype_f32_r,
                                            m,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            0,
                                            0);
#else
    rocblas_status status = rocblas_gemmex(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         HIP_R_32F,
                                         (transa == rocblas_operation_none) ? m : k,
                                         (const void*)B,
                                         HIP_R_32F,
                                         (transb == rocblas_operation_none) ? k : n,
                                         (const void*)beta,
                                         C,
                                         HIP_R_32F,
                                         m,
                                         HIP_R_32F,
                                         algo);
#endif

#ifdef __HIP_PLATFORM_HCC__
    if (status != rocblas_status_success) {
#else
    if (status != rocblas_status_success) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

#ifdef __HIP_PLATFORM_HCC__
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   rocblas_gemm_algo algo)
#else
int cublas_gemm_ex(rocblas_handle handle,
                   rocblas_operation transa,
                   rocblas_operation transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   cublasGemmAlgo_t algo)
#endif
{
#ifdef __HIP_PLATFORM_HCC__
    rocblas_status status = rocblas_gemm_ex(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            (const void*)alpha,
                                            (const void*)A,
                                            rocblas_datatype_f16_r,
                                            (transa == rocblas_operation_none) ? m : k,
                                            (const void*)B,
                                            rocblas_datatype_f16_r,
                                            (transb == rocblas_operation_none) ? k : n,
                                            (const void*)beta,
                                            (void*)C,
                                            rocblas_datatype_f16_r,
                                            m,
                                            (void*)C,
                                            rocblas_datatype_f16_r,
                                            m,
                                            rocblas_datatype_f32_r,
                                            algo,
                                            0,
                                            0);
#else
    rocblas_status status = rocblas_gemmex(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         HIP_R_16F,
                                         (transa == rocblas_operation_none) ? m : k,
                                         (const void*)B,
                                         HIP_R_16F,
                                         (transb == rocblas_operation_none) ? k : n,
                                         (const void*)beta,
                                         (void*)C,
                                         HIP_R_16F,
                                         m,
                                         HIP_R_32F,
                                         algo);
#endif

#ifdef __HIP_PLATFORM_HCC__
    if (status != rocblas_status_success) {
#else
    if (status != rocblas_status_success) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

#ifdef __HIP_PLATFORM_HCC__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo)
#else
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
#endif
{
#ifdef __HIP_PLATFORM_HCC__
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_datatype_f32_r,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_datatype_f32_r,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_datatype_f32_r,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_datatype_f32_r,
                                        m,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        algo,
                                        0,
                                        0);
#else
    rocblas_status status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       HIP_R_32F,
                                                       (op_A == rocblas_operation_none) ? m : k,
                                                       stride_A,
                                                       B,
                                                       HIP_R_32F,
                                                       (op_B == rocblas_operation_none) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       HIP_R_32F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       HIP_R_32F,
                                                       algo);
#endif

#ifdef __HIP_PLATFORM_HCC__
    if (status != rocblas_status_success) {
#else
    if (status != rocblas_status_success) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
                batch,
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

#ifdef __HIP_PLATFORM_HCC__
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                rocblas_gemm_algo algo)
#else
int cublas_strided_batched_gemm(rocblas_handle handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                rocblas_operation op_A,
                                rocblas_operation op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
#endif
{
#ifdef __HIP_PLATFORM_HCC__
    rocblas_status status =
        rocblas_gemm_strided_batched_ex(handle,
                                        op_A,
                                        op_B,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        rocblas_datatype_f16_r,
                                        (op_A == rocblas_operation_none) ? m : k,
                                        stride_A,
                                        B,
                                        rocblas_datatype_f16_r,
                                        (op_B == rocblas_operation_none) ? k : n,
                                        stride_B,
                                        beta,
                                        C,
                                        rocblas_datatype_f16_r,
                                        m,
                                        stride_C,
                                        C,
                                        rocblas_datatype_f16_r,
                                        m,
                                        stride_C,
                                        batch,
                                        rocblas_datatype_f32_r,
                                        algo,
                                        0,
                                        0);
#else
    rocblas_status status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       HIP_R_16F,
                                                       (op_A == rocblas_operation_none) ? m : k,
                                                       stride_A,
                                                       B,
                                                       HIP_R_16F,
                                                       (op_B == rocblas_operation_none) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       HIP_R_16F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       HIP_R_32F,
                                                       algo);
#endif

#ifdef __HIP_PLATFORM_HCC__
    if (status != rocblas_status_success) {
#else
    if (status != rocblas_status_success) {
#endif
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }

    return 0;
}
