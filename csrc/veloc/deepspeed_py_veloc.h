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
// #include "pybind11/numpy.h"
#include <thread>
#include <tuple>
#include <iostream>
#include <cstdint>
#include <fstream>

namespace py = pybind11;

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class veloc_ckpt_t {
    char*_start_ptr;
    size_t _curr_offset = 0;
    size_t _total_host_cache = 0;
    
    std::deque<std::tuple<int, std::string, torch::Tensor, size_t, size_t>> _pending_d2h;
    std::mutex _mutex_d2h;
    std::condition_variable _cv_d2h;
    // std::unique_lock<std::mutex> _lock_d2h;
    std::thread _thread_d2h;

    std::deque<std::tuple<int, std::string, torch::Tensor, size_t, size_t>> _pending_h2f;
    // std::deque<std::tuple<int, std::string, char*, size_t, size_t>> _pending_h2f_obj;
    std::mutex _mutex_h2f;
    std::condition_variable _cv_h2f;
    // std::unique_lock<std::mutex> _lock_h2f;
    std::thread _thread_h2f;

    bool is_active = true;
    int _gpu_id = 0;
    cudaStream_t _cpy_stream;    

    public:
    veloc_ckpt_t(size_t host_cache, int g) {
    // veloc_ckpt_t(size_t host_cache, int g) {
        try {
            std::cout << "Initing veloc_ckpt_t" << std::endl;
            // _lock_d2h(_mutex_d2h, std::defer_lock);
            // _lock_h2f(_mutex_h2f, std::defer_lock);
            _gpu_id = g;
            checkCuda(cudaSetDevice(_gpu_id));
            _total_host_cache = host_cache;
            checkCuda(cudaMallocHost(&_start_ptr, _total_host_cache));
            checkCuda(cudaStreamCreate(&_cpy_stream));
            _curr_offset = 0;
            _pending_d2h.clear();
            _pending_h2f.clear();
            // _lock_d2h.unlock();
            // _lock_h2f.unlock();
            is_active = true;
            _thread_d2h = std::thread([&] { _d2h_trf(); });
            _thread_h2f = std::thread([&] { _h2f_trf(); });
            std::cout << "Inited veloc_ckpt_t" << std::endl;
        } catch(std::exception& e) {
            std::cerr << "Standard exception caught in veloc init: " << e.what() << std::endl;
            std::abort();
        } catch (...) {
            std::cerr << "Unknown exception caught in veloc init." << std::endl;
            throw std::runtime_error("Unknown exception");
            std::abort();
        }
    }

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