#include <stdlib.h>
#include <torch/extension.h>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <tuple>
#include <iostream>
#include <ATen/core/Dict.h>

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct ckpt_t {
    void *ptr;
    size_t size;
    size_t offset;
};

class veloc_ckpt_t {
    typedef std::unordered_map<std::string, c10::IValue> ckpt_map_t;
    void *_start_ptr;
    size_t _curr_offset = 0;
    size_t _total_host_cache = 0;
    std::deque<ckpt_t> _pending;
    std::mutex _mutex;
    std::condition_variable _cv;
    // std::unique_lock<std::mutex> _lock(_mutex, std::defer_lock);
    std::thread copy_thread;

    public:
    veloc_ckpt_t(size_t host_cache) {
        _total_host_cache = host_cache;
        checkCuda(cudaMallocHost(&_start_ptr, host_cache));
        _curr_offset = 0;
        _pending.clear();
    }

    void ckpt(ckpt_map_t &m, std::string path);
    void wait(const size_t tensor_id = 0);

};