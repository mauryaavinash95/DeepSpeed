#include "memory_cache.hpp"

memory_cache_t::memory_cache_t(int d, size_t t): _device_id(d), _total_memory(t), _curr_size(0), _head(0), _tail(0) {
    TIMER_START(alloc_start);
    std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
    checkCuda(cudaSetDevice(d));
    checkCuda(cudaFree(0));
    checkCuda(cudaMallocHost(&_start_ptr, _total_memory));
    _mem_lock.unlock();
    TIMER_STOP(alloc_start, "Host memory allocation time", _total_memory);
}

memory_cache_t::~memory_cache_t() {
    checkCuda(cudaFreeHost(_start_ptr));
}

mem_region_t* memory_cache_t::_assign(size_t h, size_t s) {
    if (h+s > _total_memory) {
        std::runtime_error("Exception in assign exceeding total memory size");
        // _print_trace();
        std::abort();
    }
    char *ptr = _start_ptr + h;
    mem_region_t *m = new mem_region_t(local_uid, ptr, h, h+s);
    local_uid++;
    _head = (h + s)%_total_memory;
    _curr_size += s;
    _mem_q.push_back(m);
    // _print_trace();
    return m;
}

mem_region_t* memory_cache_t::allocate(size_t s) {
    if (s > _total_memory) 
        FATAL("Cannot allocate size " << s << " larger than the pool of " << _total_memory);
    mem_region_t* ptr = nullptr;
    std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
    while(_curr_size + s > _total_memory)
        _mem_cv.wait(_mem_lock);
    if (_tail == _head)
        _tail = _head = 0;
    if (_tail <= _head) {
        if (_total_memory - _head >= s) {
            ptr = _assign(_head, s);
        } else {
            _head = 0;
        }
    } 
    if (ptr == nullptr) {
        // Now the tail is greater than head
        while (_tail - _head < s)
            _mem_cv.wait(_mem_lock);
        ptr = _assign(_head, s);
    }
    _mem_lock.unlock();
    _mem_cv.notify_all();
    return ptr;
}

void memory_cache_t::deallocate(uint64_t _uid, size_t s) {
    try {
        if (_mem_q.empty() || _uid < 1)
            return;
        mem_region_t *m = _mem_q.front();
        if (_uid != m->uid || s != (m->end_offset-m->start_offset)) {
            std::cout << "Should deallocate the tail first. Only FIFO eviction allowed" << std::endl;
            std::cout << "Tried deleting " << _uid << " of size " << s <<  " at offset " << m->start_offset 
                << " but front element was " << (void *)m->ptr << " of size " << m->end_offset-m->start_offset << std::endl;
            // _print_trace();
            std::abort();
            return;
        }
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
        _tail += s;
        _curr_size -= s;
        _mem_q.pop_front();
        _mem_lock.unlock();
        _mem_cv.notify_all();
    } catch (std::exception &e) {
        FATAL("Exception caught in deallocate operation ." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in deallocate operation.");
    }
}

size_t memory_cache_t::size() {
    return _curr_size;
}

void memory_cache_t::_print_trace() {
    DBG("===================================================");
    for (size_t i = 0; i < _mem_q.size(); ++i) {
        const auto e = _mem_q[i];
        DBG(e->uid << (void*)e->ptr << " : " << e->start_offset << " - " << e->end_offset);
    }
    auto e = _mem_q.front();
    DBG("First element " << e->uid << " ptr " << (void *)e->ptr << " at start offset " << e->start_offset);
    DBG("Head " << _head << ", Tail " << _tail);
    DBG("===================================================");
}
