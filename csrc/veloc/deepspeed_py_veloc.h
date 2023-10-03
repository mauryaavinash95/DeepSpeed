#include <stdlib.h>

#include <deque>
#include <mutex>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <condition_variable>
#include <cuda_runtime.h>
#include <tuple>
#include <iostream>
#include <exception>
#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include <thread>

namespace py = pybind11;

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class veloc_ckpt_t {
    void *_start_ptr;
    size_t _curr_offset = 0;
    size_t _total_host_cache = 0;
    
    std::deque<std::pair<std::string, py::dict>> _pending_d2h;
    std::mutex _mutex_d2h;
    std::condition_variable _cv_d2h;
    // std::unique_lock<std::mutex> _lock_d2h;
    std::thread _thread_d2h;

    std::deque<std::pair<std::string, c10::IValue*>> _pending_h2f;
    std::mutex _mutex_h2f;
    std::condition_variable _cv_h2f;
    // std::unique_lock<std::mutex> _lock_h2f;
    std::thread _thread_h2f;

    bool is_active = true;
    // py::object argparse_module = py::module::import("argparse");
    // py::object torch_module = py::module::import("torch");
    py::object argparse_module, torch_module;

    public:
    // veloc_ckpt_t(size_t host_cache): _lock_d2h(_mutex_d2h, std::defer_lock), _lock_h2f(_mutex_h2f, std::defer_lock) {
    veloc_ckpt_t(size_t host_cache) {
        _total_host_cache = host_cache;
        checkCuda(cudaMallocHost(&_start_ptr, _total_host_cache));
        _curr_offset = 0;
        _pending_d2h.clear();
        _pending_h2f.clear();
        // _lock_d2h.unlock();
        // _lock_h2f.unlock();
        is_active = true;
        argparse_module = py::module::import("argparse");
        torch_module = py::module::import("torch");
        _thread_d2h = std::thread([&] { _d2h_trf(); });
        _thread_h2f = std::thread([&] { _h2f_trf(); });
    }

    c10::IValue convert_to_ivalue(const py::handle &obj);
    void ckpt(py::dict &m, std::string path);
    void wait();
    void _d2h_trf();
    void _h2f_trf();

};