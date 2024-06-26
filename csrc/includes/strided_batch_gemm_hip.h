// !!! This is a file automatically generated by hipify!!!
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include "context_hip.h"

template <typename T>
class StridedBatchGemm {
public:
    struct Config {
        int batch_size;
        int m;
        int n;
        int k;
        float alpha;
        float beta;
        rocblas_operation op_A;
        rocblas_operation op_B;
        std::array<int, 3> gemm_algos;

        Config(int batch,
               int mm,
               int nn,
               int kk,
               float param_alpha,
               float param_beta,
               rocblas_operation opA,
               rocblas_operation opB,
               const std::array<int, 3>& algos)
            : batch_size(batch),
              m(mm),
              n(nn),
              k(kk),
              alpha(param_alpha),
              beta(param_beta),
              op_A(opA),
              op_B(opB),
              gemm_algos(algos)
        {
        }
        void SetConfig(int mm, int nn, int kk)
        {
            m = mm;
            n = nn;
            k = kk;
        }
    };

    StridedBatchGemm(const Config& config) : _config(config) {}

    virtual ~StridedBatchGemm() {}

    void Forward(int bsz, T* output, const T* _buffer_a, const T* _buffer_b, rocblas_handle handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        cublas_strided_batched_gemm(handle,
                                    _config.m,
                                    _config.n,
                                    _config.k,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a,
                                    _buffer_b,
                                    output,
                                    _config.op_A,
                                    _config.op_B,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
#ifdef __HIP_PLATFORM_HCC__
                                    rocblas_gemm_algo(_config.gemm_algos[0]));
#else
                                    cublasGemmAlgo_t(_config.gemm_algos[0]));
#endif
    }

    void ForwardPlusSave(T* output, const T* _buffer_a, const T* _buffer_b, rocblas_handle handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        cublas_strided_batched_gemm(handle,
                                    _config.m,
                                    _config.n,
                                    _config.k,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a,
                                    _buffer_b,
                                    output,
                                    _config.op_A,
                                    _config.op_B,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    _config.batch_size,
#ifdef __HIP_PLATFORM_HCC__
                                    rocblas_gemm_algo(_config.gemm_algos[0]));
#else
                                    cublasGemmAlgo_t(_config.gemm_algos[0]));
#endif

        k_buf = _buffer_a;
        q_buf = _buffer_b;
    }

    void Backward(int bsz,
                  const T* d_output,
                  const T* _buffer_a,
                  const T* _buffer_b,
                  rocblas_handle handle,
                  T* inpGradA = nullptr,
                  T* inpGradB = nullptr)
    {
        int mb = (_config.op_A == rocblas_operation_transpose ? _config.k : _config.m);
        int kb = (_config.op_A == rocblas_operation_transpose ? _config.m : _config.k);

        int stride_a = mb * _config.n;
        int stride_b = _config.n * kb;
        int stride_c = _config.m * _config.k;

        // B need to transpose.
        rocblas_operation op_b = (_config.op_B == rocblas_operation_transpose ? rocblas_operation_none : rocblas_operation_transpose);

        // Calculate d_A.
        cublas_strided_batched_gemm(handle,
                                    mb,
                                    kb,
                                    _config.n,
                                    &_config.alpha,
                                    &_config.beta,
                                    (_config.op_A == rocblas_operation_transpose ? _buffer_b : d_output),
                                    (_config.op_A == rocblas_operation_transpose ? d_output : _buffer_b),
                                    inpGradA,
                                    rocblas_operation_none,
                                    op_b,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
#ifdef __HIP_PLATFORM_HCC__
                                    rocblas_gemm_algo(_config.gemm_algos[1]));
#else
                                    cublasGemmAlgo_t(_config.gemm_algos[1]));
#endif

        // A need to transpose.
        rocblas_operation op_a = (_config.op_A == rocblas_operation_transpose ? rocblas_operation_none : rocblas_operation_transpose);

        stride_a = _config.m * _config.k;
        stride_b = _config.m * _config.n;
        stride_c = _config.n * _config.k;

        // Calculate d_B.
        cublas_strided_batched_gemm(handle,
                                    _config.k,
                                    _config.n,
                                    _config.m,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a,
                                    d_output,
                                    inpGradB,
                                    op_a,
                                    rocblas_operation_none,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
#ifdef __HIP_PLATFORM_HCC__
                                    rocblas_gemm_algo(_config.gemm_algos[2]));
#else
                                    cublasGemmAlgo_t(_config.gemm_algos[2]));
#endif
    }

    inline int GetN() const { return _config.k; }

    inline const T* GetBufferA() const { return k_buf; }

    inline const T* GetBufferB() const { return q_buf; }

    inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

private:
    Config _config;
    const T* q_buf;
    const T* k_buf;
};
