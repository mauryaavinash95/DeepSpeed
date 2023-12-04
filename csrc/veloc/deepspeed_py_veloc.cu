#include <deepspeed_py_veloc.h>

void veloc_ckpt_t::_d2h_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
        try {
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            while(_pending_d2h.empty() && is_active)
                _cv_d2h.wait(_lock_d2h);
            if (!is_active) {
                // _pending_d2h.clear();
                _lock_d2h.unlock();
                _cv_d2h.notify_all();
                // DBG("==== Returning from d2h_trf");
                return;
            }
            // TIMER_START(sl_time);
            // std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // TIMER_STOP(sl_time, "Time for which it is sleeping ", 10);
            TIMER_START(d2h_time);
            auto e = _pending_d2h.front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();

            int version = std::get<0>(e);
            uint64_t uid = std::get<1>(e);
            std::string path = std::get<2>(e);
            torch::Tensor t = std::get<3>(e);
            size_t size = std::get<4>(e);
            size_t file_offset = std::get<5>(e);
            uint64_t enqueued_time = std::get<6>(e);
            DBG("[D2H][" << _gpu_id << "] transfer of tensor " << uid << " version " << version << " delta " << get_current_ts()-enqueued_time 
            << " enqueued at " << enqueued_time << " started at " << get_current_ts());
            mem_region_t* m = mem->allocate(uid, size);
            char *host_ptr = m->ptr;
            char *src_ptr = static_cast<char *>(t.data_ptr());
            TIMER_STOP(d2h_time, "[D2H][" << _gpu_id << "] Allocation time for " << m->uid << " version " << version, size);
            TIMER_START(memcpy_time);

            checkCuda(cudaMemcpyAsync(host_ptr, src_ptr, size, cudaMemcpyDeviceToHost, _cpy_stream));
            checkCuda(cudaStreamSynchronize(_cpy_stream));

            /*
            size_t rem = size;
            size_t D2H_CHUNK_SIZE = (32 << 20);
            
            while(rem > 0) {
                size_t chunkSize = D2H_CHUNK_SIZE < rem ? D2H_CHUNK_SIZE : rem;
                size_t curr = size - rem;
                TIMER_START(ctime);
                checkCuda(cudaMemcpyAsync(host_ptr+curr, src_ptr+curr, chunkSize, cudaMemcpyDeviceToHost, _cpy_stream));
                checkCuda(cudaStreamSynchronize(_cpy_stream));
                TIMER_STOP(ctime, "[D2H][" << _gpu_id << "] D2H Part Memcpy time for " << m->uid << " version " << version << " rem " << rem << " out of " << size, size);
                rem -= chunkSize;
            }
            
            */
            TIMER_STOP(memcpy_time, "[D2H][" << _gpu_id << "] D2H Memcpy time for " << m->uid << " version " << version, size);

            // TIMER_START(blob_time);
            // torch::Tensor cpu_tensor = torch::from_blob(ptr, t.sizes(), t.dtype());
            // if ((void *)ptr != (void *)(cpu_tensor.data_ptr())) {
            //     FATAL("In d2h trf offsets don't match for " << m->uid << " version " << version);
            // }
            // TIMER_STOP(blob_time, "[D2H] Time to convert from blob to tensor for " << m->uid << " version " << version, size);
            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            // _pending_h2f.push_back(std::make_tuple(version, m->uid, path, cpu_tensor, size, file_offset, get_current_ts()));
            _pending_h2f.push_back(std::make_tuple(version, m->uid, path, host_ptr, size, file_offset, get_current_ts()));
            _lock_h2f.unlock();
            _cv_h2f.notify_all();

            _lock_d2h.lock();
            _pending_d2h.pop_front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            TIMER_STOP(d2h_time, "[D2H][" << _gpu_id << "] Total time for GPU to process " << m->uid << " version " << version, size);
            DBG("[D2H][" << _gpu_id << "] transfer of tensor " << uid  << " version " << version << " delta " << get_current_ts()-enqueued_time << " enqueued at " << enqueued_time << " completed at " << get_current_ts());
            
        } catch (std::exception &e) {
            FATAL("Exception caught in d2h trf." << e.what());
            // DBG("==== Returning from d2h_trf");
        } catch (...) {
            FATAL("Unknown exception caught in d2h trf.");
            // DBG("==== Returning from d2h_trf");
        }
    }
}

void veloc_ckpt_t::_write_file(char * ptr, std::string path, size_t startIdx, size_t endIdx, size_t file_offset, std::uint64_t uid, int version, int threadID) {
    std::ofstream f;            
    f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
    f.seekp(file_offset+startIdx);
    // TIMER_START(f_write_time);
    // f.write(static_cast<char *>(t.data_ptr()), (t.numel()*t.element_size()));
    f.write(ptr+startIdx, endIdx-startIdx);
    f.close();
    // TIMER_STOP(f_write_time, "[H2F] Time to write file for " << uid << " version " << version << " using thread " << threadID, endIdx-startIdx);
}

void veloc_ckpt_t::_h2f_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
        try {
            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            TIMER_START(h2f_wait);
            while(_pending_h2f.empty() && is_active)
                _cv_h2f.wait(_lock_h2f);
            
            if (!is_active) {
                // _pending_h2f.clear();
                _lock_h2f.unlock();
                _cv_h2f.notify_all();
                // DBG("==== Returning from h2f_trf");
                return;
            }
            TIMER_START(h2f_time);
            auto e = _pending_h2f.front();
            _lock_h2f.unlock();
            _cv_h2f.notify_all();
            
            int version = std::get<0>(e);
            uint64_t uid = std::get<1>(e);
            std::string path = std::get<2>(e);
            // torch::Tensor t = std::get<3>(e);
            char* ptr = std::get<3>(e);
            size_t size = std::get<4>(e);
            size_t file_offset = std::get<5>(e);
            uint64_t enqueued_time = std::get<6>(e);
            if (size > (256<<20))
                TIMER_STOP(h2f_wait, "[" << _gpu_id << "] Waiting in h2f transfer ", 10);
            DBG("[H2F][" << _gpu_id << "] flush for tensor uid " << uid  << " version " << version << " delta " << get_current_ts()-enqueued_time << " enqueued at " << enqueued_time << " started at " << get_current_ts());
            // char *ptr = static_cast<char *>(t.data_ptr());

            // -------- Multi-thread working right
            // int num_writer_threads = writer_threads;
            // size_t chunkSize = ceil(size / num_writer_threads);
            // if (chunkSize < MIN_CHUNK_SIZE) {
            //     chunkSize = MIN_CHUNK_SIZE;
            //     num_writer_threads = ceil(size/chunkSize);
            // }
            // std::vector<std::thread> write_threads(num_writer_threads);
            // for(int threadID=0; threadID<num_writer_threads; threadID++) {
            //     size_t startIdx = threadID * chunkSize;
            //     size_t endIdx = (threadID == num_writer_threads - 1) ? size : startIdx + chunkSize;
            //     write_threads[threadID] = std::thread([&] { _write_file(ptr, path, startIdx, endIdx, file_offset, uid, version, threadID); });
            // }
            // for(int threadID=0; threadID<num_writer_threads; threadID++) { 
            //     write_threads[threadID].join();
            // }
            // -------- Multi-thread working right
            
            // -------- Single thread working right
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
            f.seekp(file_offset);
            // f.write(static_cast<char *>(t.data_ptr()), (t.numel()*t.element_size()));
            f.write(ptr, size);
            f.close();
            // -------- Single thread working right

            mem->deallocate(uid, size);
            _lock_h2f.lock();
            _pending_h2f.pop_front();
            _lock_h2f.unlock();
            _cv_h2f.notify_all();
            TIMER_STOP(h2f_time, "[H2F][" << _gpu_id << "] Total time in h2f to save tensor " << uid << " version " << version << " of size " << size, size);
            DBG("[H2F][" << _gpu_id << "] flush for tensor uid " << uid  << " version " << version << " delta " << get_current_ts()-enqueued_time << " enqueued at " << enqueued_time << " completed at " << get_current_ts());
        }  catch (std::exception &e) {
            FATAL("Exception caught in h2f trf." << e.what());
            // DBG("==== Returning from h2f_trf");
        } catch (...) {
            FATAL("Unknown exception caught in h2f trf.");
            // DBG("==== Returning from h2f_trf");
        }
    }
}

void veloc_ckpt_t::_p2f_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
        try {
            std::unique_lock<std::mutex> _lock_p2f(_mutex_p2f);
            while(_pending_p2f.empty() && is_active)
                _cv_p2f.wait(_lock_p2f);
            if (!is_active) {
                _lock_p2f.unlock();
                _cv_p2f.notify_all();
                return;
            }
            TIMER_START(p2f_time);
            auto e = _pending_p2f.front();
            _lock_p2f.unlock();
            _cv_p2f.notify_all();
            
            int version = std::get<0>(e);
            uint64_t uid = std::get<1>(e);
            std::string path = std::get<2>(e);
            char *t = std::get<3>(e);
            size_t size = std::get<4>(e);
            size_t file_offset = std::get<5>(e);
                        
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
            f.seekp(file_offset);
            TIMER_START(f_write_time);
            f.write((const char *)t, size);
            f.close();
            free(t);
            TIMER_STOP(f_write_time, "[P2F] Time to write from pickle to file for " << version << " path " << path, size);

            _lock_p2f.lock();
            _pending_p2f.pop_front();
            _lock_p2f.unlock();
            _cv_p2f.notify_all();
            TIMER_STOP(p2f_time, "[P2F] Total time in p2f to save pickle " <<  version << " of size " << size, size);
        }  catch (std::exception &e) {
            FATAL("Exception caught in p2f trf." << e.what());
        } catch (...) {
            FATAL("Unknown exception caught in p2f trf.");
        }
    }
}


void veloc_ckpt_t::ckpt_header_size(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t value, std::string path) {
    try {
        TIMER_START(header_time);
        std::ofstream f;
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
        f.seekp(start_offset);
        f.write((const char*)&value, end_offset-start_offset);
        f.close();
        TIMER_STOP(header_time, "Time to write header of size " << end_offset-start_offset, end_offset-start_offset);
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt_header_size." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in ckpt_header_size." << path);
    }
}

void veloc_ckpt_t::ckpt_pickle(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, py::bytes value, std::string path) {
    try {
        TIMER_START(pik_time);
        const char* ptr = static_cast<const char *>(PyBytes_AsString(value.ptr()));
        size_t size = PyBytes_Size(value.ptr());
        char *cpp_str = (char *)malloc(size);
        std::memcpy(cpp_str, ptr, size);
        assert((size == end_offset-start_offset) && "Size of pickled object is not correct");
        std::unique_lock<std::mutex> _lock_p2f(_mutex_p2f);
        _pending_p2f.push_back(std::make_tuple(version, 0, path, cpp_str, size, start_offset));
        _lock_p2f.unlock();
        _cv_p2f.notify_all();
        TIMER_STOP(pik_time, "Pickle time path " << path << " size " << size << " end offset " << end_offset 
            << " start offset " << start_offset << " ptr " << (void *)cpp_str, size);
        return; 
        // char* ptr =PyBytes_AsString(value.ptr());
        // size_t size = PyBytes_Size(value.ptr());
        // assert((size == end_offset-start_offset) && "Size of pickled object is not correct");
        // std::ofstream f;
        // f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        // f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
        // f.seekp(start_offset);
        // f.write((const char*)ptr, size);
        // f.close();

    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt_pickle." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in ckpt_pickle." << path);
    }
}

void veloc_ckpt_t::ckpt_obj(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t ptr_id, const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path) {
    FATAL("In ckpt obj for " << version <<  " size " << size << " device " << device_id << " path " << path)
    try {
        char* ptr = reinterpret_cast<char*>(ptr_id);
        cudaPointerAttributes attr;
        checkCuda(cudaPointerGetAttributes(&attr, (const void *)ptr));
        if (attr.type == cudaMemoryTypeDevice)
            assert((attr.type == cudaMemoryTypeDevice && device_id > -1 && device_id == _gpu_id) && "Device pointer problem in ckpt_obj");
        if (attr.type == cudaMemoryTypeDevice && device_id > -1) {
            throw std::runtime_error("Do not know how to checkpoint device objects");
            return;
        }
        throw std::runtime_error("Do not know how to checkpoint device objects");
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt_pickle." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in ckpt." << path);
    }
}

void veloc_ckpt_t::ckpt_tensor(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const torch::Tensor &t, 
        const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path) {
    try {
        if (t.device().is_cuda()) 
            assert((t.device().index() == _gpu_id) && "Tensor not on the same GPU as ckpt engine");
        uint64_t uid = local_uid++;
        if (t.device().is_cuda()) {
            DBG("[" << _gpu_id << "] Enqueuing GPU tensor " << uid << " version  " << version << " size " << size);
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            _pending_d2h.push_back(std::make_tuple(version, uid, path, t, size, file_offset, get_current_ts()));
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            return;
        } 
        DBG("[" << _gpu_id << "] Enqueuing host tensor " << uid << " version  " << version << " size " << size);
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        _pending_h2f.push_back(std::make_tuple(version, uid, path, static_cast<char *>(t.data_ptr()), size, file_offset, get_current_ts()));
        _lock_h2f.unlock();
        _cv_h2f.notify_all();
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt_tensor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in ckpt_tensor." << path);
    }
}

void veloc_ckpt_t::wait(int version) {
    try {
        TIMER_START(wait_timer);
        std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
        while(!(_pending_d2h.empty())) {
            DBG("[" << _gpu_id << "] Waiting in d2h for " << _pending_d2h.size());
            for(auto e: _pending_d2h) {
                DBG("[" << _gpu_id << "]" << std::get<0>(e) << " UID " << std::get<1>(e) << " size " << std::get<4>(e));
            }
            _cv_d2h.wait(_lock_d2h);
        }
        _lock_d2h.unlock();
        _cv_d2h.notify_all();
        TIMER_STOP(wait_timer, "[" << _gpu_id << "] Wait D2H complete ", 1);
    }  catch (std::exception &e) {
        FATAL("Exception caught in wait D2H." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in wait D2H.");
    }
}

void veloc_ckpt_t::shutdown() {
    try {
        wait();
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        // Wait for D2H transfers
        while((!_pending_h2f.empty())) {
            DBG("[" << _gpu_id << "] Waiting in h2f for " << _pending_h2f.size());
            for(auto e: _pending_h2f) {
                DBG("[" << _gpu_id << "]" << std::get<0>(e) << " UID " << std::get<1>(e) << " size " << std::get<4>(e));
            }
            _cv_h2f.wait(_lock_h2f);
        }
        _lock_h2f.unlock();
        _cv_h2f.notify_all();

        std::unique_lock<std::mutex> _lock_p2f(_mutex_p2f);
        // Wait for D2H transfers
        while((!_pending_p2f.empty())) {
            DBG("[" << _gpu_id << "] Waiting in p2f for " << _pending_p2f.size());
            for(auto e: _pending_p2f) {
                DBG("[" << _gpu_id << "] " << std::get<0>(e) << " UID " << std::get<1>(e) << " size " << std::get<4>(e));
            }
            _cv_p2f.wait(_lock_p2f);
        }
        _lock_p2f.unlock();
        _cv_p2f.notify_all();
        
        is_active = false;
        _cv_h2f.notify_all();
        _cv_d2h.notify_all();
        _cv_p2f.notify_all();
        _thread_d2h.join();
        _thread_h2f.join();
        _thread_p2f.join();
        // delete mem;
        DBG("==== Returning from shutdown");
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in shutdown." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in shutdown.");
    }
}
