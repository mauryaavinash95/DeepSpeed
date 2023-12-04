#include "memory_cache.hpp"

memory_cache_t::memory_cache_t(int d, size_t t): _device_id(d), _total_memory(t), _curr_size(0), _head(0), _tail(0) {
    try {
        is_active = true;
        max_allocated = 0;
        malloc_thread = std::thread([&] { allocate_pin_mem(); });
        // malloc_thread.detach();
        DBG("Returned from the memory cache function..... ");
    } catch (std::exception &e) {
        FATAL("Exception caught in memory cache constructor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in memory cache constructor.");
    }
}

memory_cache_t::~memory_cache_t() {
    try {
        is_active = false;
        _mem_cv.notify_all();
        checkCuda(cudaFreeHost(_start_ptr));
        malloc_thread.join();
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in memory cache destructor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in memory cache destructor.");
    }
}

void memory_cache_t::allocate_pin_mem() {
    try{
        TIMER_START(alloc_start);
        
        checkCuda(cudaSetDevice(_device_id));
        checkCuda(cudaFree(0));
        int posix_memalign_result = posix_memalign((void **)&_start_ptr, HUGEPAGES_SIZE, _total_memory);
        madvise(_start_ptr, _total_memory, MADV_HUGEPAGE);
        if (posix_memalign_result != 0) {
            FATAL("Error allocating hugepages: " << posix_memalign_result);
        }
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex, std::defer_lock);
        omp_set_num_threads(MALLOC_THREADS);
        TIMER_START(alloc_starting);
        size_t rem = _total_memory;
        while (max_allocated != _total_memory) {
            size_t chunk = MIN_CHUNK_SIZE < rem ? MIN_CHUNK_SIZE : rem;
            #pragma omp parallel
            {
                char *buf = (char *)_start_ptr+max_allocated;
                int id = omp_get_thread_num();
                int num = omp_get_num_threads();
                size_t my_size = chunk/num;
                size_t my_start = id*my_size;
                // if (touch_pages) {
                // #pragma omp parallel for
                // {
                // for(size_t i = 0; i < my_size; i+=HUGEPAGES_SIZE)
                for(size_t i = 0; i < my_size; i+=CACHE_LINE_SIZE)
                    buf[my_start + i] = 0x00;
                // }
                // } else {
                // memset(buf+my_start, 0, my_size);
                // }
            }
            _mem_lock.lock();
            max_allocated += chunk;
            _mem_lock.unlock();
            _mem_cv.notify_all();
            rem -= chunk;
        }
        if (max_allocated != _total_memory) 
            FATAL("Max allocated is not same as total " << max_allocated << " total was " << _total_memory);
        TIMER_STOP(alloc_starting, "Simple malloc and touch done", _total_memory);
        TIMER_START(pinning);
        _mem_lock.lock();
        checkCuda(cudaHostRegister(_start_ptr, _total_memory, cudaHostRegisterPortable));   
        _mem_lock.unlock();
        _mem_cv.notify_all();
        TIMER_STOP(pinning, "Time to pin memory", _total_memory);
        TIMER_STOP(alloc_start, "Host memory allocation time on device " << _device_id, _total_memory);
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate pin memory." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in allocate pin memory.");
    }
}

mem_region_t* memory_cache_t::_assign(const uint64_t uid, size_t h, size_t s) {
    try {
        if (h+s > _total_memory) {
            std::runtime_error("Exception in assign exceeding total memory size");
            // _print_trace();
            std::abort();
        }
        char *ptr = _start_ptr + h;
        mem_region_t *m = new mem_region_t(uid, ptr, h, h+s);
        _head = (h + s)%_total_memory;
        _curr_size += s;
        _mem_q.push_back(m);
        // _print_trace();
        return m;
    } catch (std::exception &e) {
        FATAL("Exception caught in _assign." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in _assign.");
    }
}

mem_region_t* memory_cache_t::allocate(const uint64_t uid, size_t s) {
    try {
        // DBG("Attempting to allocate of size " << s << " when current memory is " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
        if (s > _total_memory) 
            FATAL("Cannot allocate size " << s << " larger than the pool of " << _total_memory);
        mem_region_t* ptr = nullptr;
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
        while(((max_allocated < _total_memory) || (_curr_size + s > _total_memory)) && is_active) {
            _mem_cv.wait(_mem_lock);
        }
        if (!is_active) {
            return ptr;
        }
        if (_tail == _head)
            _tail = _head = 0;
        if (_tail <= _head) {
            if (_total_memory - _head >= s) {
                ptr = _assign(uid, _head, s);
            } else {
                _head = 0;
            }
        } 
        if (ptr == nullptr) {
            // Now the tail is greater than head
            while(((_tail > _head) && (_tail - _head < s)) && is_active) {
                // DBG("Waiting in second wait _tail -head < s, head " << _head << " tail: " << _tail << " s " << s);
                TIMER_START(wait_time);
                _mem_cv.wait(_mem_lock);
                TIMER_STOP(wait_time, "Waiting for more memory " << _tail - _head << " uid " << uid << " size " << s, 10);
            }
            if (!is_active)
                return ptr;
            ptr = _assign(uid, _head, s);
        }
        _mem_lock.unlock();
        _mem_cv.notify_all();
        return ptr;
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate function." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in allocate function.");
    }
}

void memory_cache_t::deallocate(uint64_t _uid, size_t s) {
    try {
        // DBG("Attempting to deallocate " << _uid << " of size " << s << " cur size " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
        if (_mem_q.empty() || _uid < 1)
            return;
        mem_region_t *m = _mem_q.front();
        if (_uid != m->uid || s != (m->end_offset-m->start_offset)) {
            std::cout << "Should deallocate the tail first. Only FIFO eviction allowed" << std::endl;
            std::cout << "Tried deleting " << _uid << " of size " << s <<  " at offset " << m->start_offset 
                << " but front element was " << (void *)m->ptr << " of size " << m->end_offset-m->start_offset << std::endl;
            _print_trace();
            // std::abort();
            return;
        }
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
        _tail += s;
        if (_tail > _total_memory)
            _tail = 0;
        _curr_size -= s;
        _mem_q.pop_front();
        _mem_lock.unlock();
        _mem_cv.notify_all();
        // DBG("Deallocated from host " << _uid << " of size " << s << " cur size " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
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
    try {
        DBG("===================================================");
        for (size_t i = 0; i < _mem_q.size(); ++i) {
            const auto e = _mem_q[i];
            DBG(e->uid << (void*)e->ptr << " : " << e->start_offset << " - " << e->end_offset);
        }
        auto e = _mem_q.front();
        DBG("First element " << e->uid << " ptr " << (void *)e->ptr << " at start offset " << e->start_offset);
        DBG("Head " << _head << ", Tail " << _tail);
        DBG("===================================================");
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate _print_trace." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in _print_trace.");
    }
}
