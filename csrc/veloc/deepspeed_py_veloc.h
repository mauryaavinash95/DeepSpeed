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


namespace py = pybind11;

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class veloc_ckpt_t {
    // typedef std::unordered_map<std::string, c10::IValue> ckpt_map_t;
    void *_start_ptr;
    size_t _curr_offset = 0;
    size_t _total_host_cache = 0;
    struct ckpt_t {
        void *ptr;
        size_t size;
        size_t offset;
    };
    std::deque<ckpt_t*> _pending;
    std::mutex _mutex;
    std::condition_variable _cv;
    // std::unique_lock<std::mutex> _lock(_mutex, std::defer_lock);
    std::thread copy_thread;

    public:
    veloc_ckpt_t(size_t host_cache) {
        _total_host_cache = host_cache;
        checkCuda(cudaMallocHost(&_start_ptr, _total_host_cache));
        _curr_offset = 0;
        _pending.clear();
    }

    // void ckpt(ckpt_map_t &m, std::string path);
    at::IValue convert_to_ivalue(const py::handle &obj);
    void ckpt(py::dict &m, std::string path);
    void wait(const size_t tensor_id = 0);

};