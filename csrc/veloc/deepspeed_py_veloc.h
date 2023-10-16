#ifndef __DEEPSPEED_PY_VELOC_HPP
#define __DEEPSPEED_PY_VELOC_HPP

#include <stdlib.h>
#include <deque>
#include <mutex>
// #include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <condition_variable>
#include <cuda_runtime.h>
#include <tuple>
#include <iostream>
#include <exception>
// #include <ATen/core/Dict.h>
// #include <ATen/core/ivalue.h>
// #include <ATen/core/Tensor.h>
// #include <ATen/core/grad_mode.h>
// #include <c10/util/ArrayRef.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <pybind11/stl.h>
#include <thread>
#include <tuple>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "memory_cache.hpp"

namespace py = pybind11;



class veloc_ckpt_t {    
    // Tuple contains: <version, unique_region_id, path, tensor, size, file_offset>
    std::deque<std::tuple<int, uint64_t, std::string, torch::Tensor, size_t, size_t>> _pending_d2h;
    std::mutex _mutex_d2h;
    std::condition_variable _cv_d2h;
    std::thread _thread_d2h;

    std::deque<std::tuple<int, uint64_t, std::string, torch::Tensor, size_t, size_t>> _pending_h2f;
    std::mutex _mutex_h2f;
    std::condition_variable _cv_h2f;
    std::thread _thread_h2f;

    bool is_active = true;
    int _gpu_id = 0;
    cudaStream_t _cpy_stream;    
    memory_cache_t *mem;
    std::stringstream s_stream;
    
    public:
    veloc_ckpt_t(size_t host_cache, int g) {
        try {
            _gpu_id = g;
            checkCuda(cudaSetDevice(_gpu_id));
            checkCuda(cudaStreamCreate(&_cpy_stream));
            is_active = true;
            _thread_d2h = std::thread([&] { _d2h_trf(); });
            _thread_h2f = std::thread([&] { _h2f_trf(); });
            mem = new memory_cache_t(_gpu_id, host_cache);
            _pending_d2h.clear();
            _pending_h2f.clear();
            DBG("Inited veloc_ckpt_t on GPU ID " << g << " for host cache size of (MB) " << (host_cache >> 20));
        } catch(std::exception& e) {
            FATAL("Standard exception caught in veloc init: " << e.what());
        } catch (...) {
            FATAL("Unknown exception caught in veloc init.");
        }
    };

    // void begin_ckpt_version(int version);
    void ckpt_header_size(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t value, std::string path);
    void ckpt_pickle(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, py::bytes value, std::string path);
    void ckpt_obj(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t ptr, const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path);
    void ckpt_tensor(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const torch::Tensor &t, 
        const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path);
    // void end_ckpt_version(int version);
    void wait(int version = -1);
    void _d2h_trf();
    void _h2f_trf();
    void shutdown();
};

#endif // __DEEPSPEED_PY_VELOC_HPP