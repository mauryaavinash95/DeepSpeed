#include <deepspeed_py_veloc.h>

void veloc_ckpt_t::_d2h_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
        try {
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            // _lock_d2h.lock();
            while(_pending_d2h.empty() && is_active)
                _cv_d2h.wait(_lock_d2h);
            // std::cout << "Got out of wait in _d2h_trf thread" << std::endl;
            if (!is_active)
                return;
            auto e = _pending_d2h.front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();

            int version = std::get<0>(e);
            std::string path = std::get<1>(e);
            // const void *const ptr = std::get<1>(e);
            torch::Tensor t = std::get<2>(e);
            size_t size = std::get<3>(e);
            size_t file_offset = std::get<4>(e);
            torch::Tensor cpu_tensor = t.to(torch::kCPU);
            // checkCuda(cudaMemcpyAsync(_start_ptr, ptr, size, cudaMemcpyDeviceToHost, _cpy_stream));
            // std::cout << "Moved to the host" << path << std::endl;

            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            // _lock_h2f.lock();
            _pending_h2f.push_back(std::make_tuple(version, path, cpu_tensor, size, file_offset));
            _lock_h2f.unlock();
            _cv_h2f.notify_all();

            _lock_d2h.lock();
            _pending_d2h.pop_front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            // std::cout << "Popped from the d2h_queue" << std::endl;
        } catch (std::exception &e) {
            std::cerr << "Exception caught in d2h trf." << e.what() << std::endl;
            std::abort();
        } catch (...) {
            std::cerr << "Unknown exception caught in d2h trf." << std::endl;
            throw std::runtime_error("Unknown exception");
            std::abort();
        }
    }
}

void veloc_ckpt_t::_h2f_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
            try {
            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            while(_pending_h2f.empty() && is_active)
                _cv_h2f.wait(_lock_h2f);
            if (!is_active)
                return;
            auto e = _pending_h2f.front();
            _lock_h2f.unlock();
            _cv_h2f.notify_all();

            int version = std::get<0>(e);
            std::string path = std::get<1>(e);
            torch::Tensor t = std::get<2>(e);
            size_t size = std::get<3>(e);
            size_t file_offset = std::get<4>(e);
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
            f.seekp(file_offset);
            auto pickled = torch::pickle_save(t);
            f.write((char*)pickled.data(), size);
            f.close();
            _lock_h2f.lock();
            _pending_h2f.pop_front();
            _lock_h2f.unlock();
            _cv_h2f.notify_all();
        }  catch (std::exception &e) {
            std::cerr << "Exception caught in h2f trf." << e.what() << std::endl;
            std::abort();
        } catch (...) {
            std::cerr << "Unknown exception caught in h2f trf." << std::endl;
            throw std::runtime_error("Unknown exception");
            std::abort();
        }
    }
}

void veloc_ckpt_t::ckpt_header_size(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t value, std::string path) {
    try {
        // std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        // _pending_h2f.push_back(std::make_tuple(path, (char *)&value, end_offset-start_offset, start_offset));
        // _lock_h2f.unlock();
        // _cv_h2f.notify_all();

        std::ofstream f;
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
        f.seekp(start_offset);
        f.write((const char*)&value, end_offset-start_offset);
        f.close();
    } catch (std::exception &e) {
        std::cerr << "Exception caught in ckpt_header_size." << e.what() << std::endl;
        std::abort();
    } catch (...) {
        std::cerr << "Unknown exception caught in ckpt_header_size." << path << std::endl;
        throw std::runtime_error("Unknown exception");
        std::abort();
    }
}

void veloc_ckpt_t::ckpt_pickle(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, py::bytes value, std::string path) {
    try {
        char* ptr = PyBytes_AsString(value.ptr());
        size_t size = PyBytes_Size(value.ptr());
        assert((size == end_offset-start_offset) && "Size of pickled object is not correct");
        // std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        // _pending_h2f.push_back(std::make_tuple(path, ptr, size, start_offset));
        // _lock_h2f.unlock();
        // _cv_h2f.notify_all();

        std::ofstream f;
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
        f.seekp(start_offset);
        f.write((const char*)ptr, size);
        f.close();
    } catch (std::exception &e) {
        std::cerr << "Exception caught in ckpt_pickle." << e.what() << std::endl;
        std::abort();
    } catch (...) {
        std::cerr << "Unknown exception caught in ckpt_pickle." << path << std::endl;
        throw std::runtime_error("Unknown exception");
        std::abort();
    }
}

void veloc_ckpt_t::ckpt_obj(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t ptr_id, const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path) {
    try {
        char* ptr = reinterpret_cast<char*>(ptr_id);
        cudaPointerAttributes attr;
        checkCuda(cudaPointerGetAttributes(&attr, (const void *)ptr));
        if (attr.type == cudaMemoryTypeDevice)
            assert((attr.type == cudaMemoryTypeDevice && device_id > -1 && device_id == _gpu_id) && "Device pointer problem in ckpt_obj");
        if (attr.type == cudaMemoryTypeDevice && device_id > -1) {
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            while (!_pending_d2h.empty())
                _cv_d2h.wait(_lock_d2h);
            // _pending_d2h.push_back(std::make_tuple(path, ptr, size, file_offset));
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            std::cout << "Saving started for GPU data structure in CPP for " << path << std::endl;
            throw std::runtime_error("Do not know how to checkpoint device objects");
            return;
        } 
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        // _pending_h2f.push_back(std::make_tuple(path, ptr, size, file_offset));
        _lock_h2f.unlock();
        _cv_h2f.notify_all();

        // _lock_h2f.lock();
        // while(!_pending_h2f.empty())
        //     _cv_h2f.wait(_lock_h2f);

        std::cout << "Saving started for host data structure in CPP for " << path << " of size " << size << std::endl;
        return;
    } catch (std::exception &e) {
        std::cerr << "Exception caught in ckpt_pickle." << e.what() << std::endl;
        std::abort();
    } catch (...) {
        std::cerr << "Unknown exception caught in ckpt." << path << std::endl;
        throw std::runtime_error("Unknown exception");
        std::abort();
    }
}

void veloc_ckpt_t::ckpt_tensor(int version, const std::uint64_t start_offset, const std::uint64_t end_offset, const torch::Tensor &t, 
        const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path) {
    try {
        if (t.device().is_cuda()) 
            assert((t.device().index() == _gpu_id) && "Tensor not on the same GPU as ckpt engine");
        if (t.device().is_cuda()) {
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            while (!_pending_d2h.empty())
                _cv_d2h.wait(_lock_d2h);
            _pending_d2h.push_back(std::make_tuple(version, path, t, size, file_offset));
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            // std::cout << "Saving started for GPU data structure in CPP for " << path << std::endl;
            return;
        } 
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        _pending_h2f.push_back(std::make_tuple(version, path, t, size, file_offset));
        _lock_h2f.unlock();
        _cv_h2f.notify_all();
        // _lock_h2f.lock();
        // while(!_pending_h2f.empty())
        //     _cv_h2f.wait(_lock_h2f);
        // std::cout << "Saving started for host data structure in CPP for " << path << " of size " << size << std::endl;
        return;
    } catch (std::exception &e) {
        std::cerr << "Exception caught in ckpt_pickle." << e.what() << std::endl;
        std::abort();
    } catch (...) {
        std::cerr << "Unknown exception caught in ckpt." << path << std::endl;
        throw std::runtime_error("Unknown exception");
        std::abort();
    }
}

void veloc_ckpt_t::wait(int version) {
    // _lock_d2h.lock();
    std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
    while (!_pending_d2h.empty())
        _cv_d2h.wait(_lock_d2h);
    _lock_d2h.unlock();
    _cv_d2h.notify_all();

    std::unique_lock<std::mutex> _lock_h2f(_mutex_d2h);
    while (!_pending_h2f.empty())
        _cv_h2f.wait(_lock_h2f);
    _lock_h2f.unlock();
    _cv_h2f.notify_all();
    std::cout << "Wait complete" << std::endl;
}

void veloc_ckpt_t::shutdown() {
    wait();
    is_active = false;
    _cv_h2f.notify_all();
    _cv_d2h.notify_all();
    _thread_d2h.join();
    _thread_h2f.join();
    return;
}
