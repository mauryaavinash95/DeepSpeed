#include <deepspeed_py_veloc.h>

c10::IValue veloc_ckpt_t::convert_to_ivalue(const py::handle& input) {
    try {
        std::cout << "Got input type as " << py::str(input.get_type()) << std::endl;
        std::cout << "Value is set as " << input << std::endl;
        if (py::isinstance<py::none>(input)) 
            return c10::IValue();
        if (py::isinstance<py::str>(input))
            return c10::IValue(py::cast<py::str>(input));
        if (py::isinstance<py::float_>(input))
            return c10::IValue(py::cast<float>(input));
        if (py::isinstance<py::int_>(input))
            return c10::IValue(py::cast<int64_t>(input));
        if (py::isinstance(input, py::module::import("torch").attr("Tensor"))) 
            return c10::IValue(py::cast<at::Tensor>(input));
        if (py::isinstance<py::array>(input)) {
            py::array arr = py::cast<py::array>(input);
            auto numpy_array_info = arr.request();
            at::Tensor tensor = torch::from_blob(numpy_array_info.ptr, {numpy_array_info.size});
            return c10::IValue(tensor);
        }
        if (py::isinstance(input, py::module::import("torch").attr("dtype"))) {
            return c10::IValue(py::cast<py::str>(input)); // HACK
        }
        py::object argparse = py::module::import("argparse");
        if (py::isinstance(input, argparse.attr("Namespace"))) {
            c10::impl::GenericDict namespace_dict(c10::StringType::get(), c10::AnyType::get());     
            py::list attribute_names = py::list(input.attr("__dict__").attr("keys")());
            for (py::handle name : attribute_names) {
                py::object value = input.attr("__dict__")[name];
                namespace_dict.insert(py::str(name), convert_to_ivalue(value));
            }
            return c10::IValue(namespace_dict);
        }
        if (py::isinstance<py::dict>(input)) {
            const py::dict& py_dict = py::cast<py::dict>(input);
            c10::impl::GenericDict ivalue_dict(c10::StringType::get(), c10::AnyType::get());
            for (const auto& item : py_dict) {
                py::str key = py::str(item.first);
                std::cout << "Got dict key as " << key << " of type " << py::str(item.second.get_type()) << std::endl;
                ivalue_dict.insert(key, convert_to_ivalue(item.second));
            }
            return c10::IValue(ivalue_dict);
        } else if (py::isinstance<py::list>(input) || py::isinstance<py::tuple>(input) || py::isinstance<py::set>(input)) {
            auto py_list = py::cast<py::list>(input);
            c10::impl::GenericList ivalue_list_generic(c10::AnyType::get());
            for (const auto &ele : py_list) {
                ivalue_list_generic.push_back(convert_to_ivalue(ele));
            }
            return c10::IValue(ivalue_list_generic);
        } else {
            // TODO: <enum 'ModelType'> for input .ModelType.encoder_or_decoder.
            // std::cout << "Type is now " << py::str(input.get_type()) << " for input ." << input << "."<< std::endl;
            try {
                return c10::IValue(py::cast<py::str>(input));
            } catch(std::exception& e) {
                std::cerr << "Cannot convert unknown type to string:" << input << ". error " << e.what() << " type was " 
                    << py::str(input.get_type()) << std::endl;
                throw std::runtime_error("Unexptected conversion");
            }
        }
    } catch(std::exception& e) {
        std::cout << "Input was |" << input << "|" << std::endl; 
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        std::abort();
    } catch (...) {
        std::cerr << "Unknown exception caught." << input << std::endl;
        throw std::runtime_error("Unknown exception");
        std::abort();
    }
}


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

            std::string path = std::get<0>(e);
            const void *const ptr = std::get<1>(e);
            size_t size = std::get<2>(e);
            size_t file_offset = std::get<3>(e);
            checkCuda(cudaMemcpyAsync(_start_ptr, ptr, size, cudaMemcpyDeviceToHost, _cpy_stream));
            std::cout << "Moved to the host" << path << std::endl;

            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            // _lock_h2f.lock();
            _pending_h2f.push_back(std::make_tuple(path, _start_ptr, size, file_offset));
            _lock_h2f.unlock();
            _cv_h2f.notify_all();

            _lock_d2h.lock();
            _pending_d2h.pop_front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            // std::cout << "Popped from the d2h_queue" << std::endl;
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

            std::string path = std::get<0>(e);
            const void* const ptr = std::get<1>(e);
            size_t size = std::get<2>(e);
            size_t file_offset = std::get<3>(e);
            std::ofstream f;
            try {
                f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
                f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
                f.seekp(file_offset);
            } catch (std::exception &e) {
                std::cerr << "Problem occurs in opening or seeking the file " << std::endl;
                std::abort();
            }
            try {
                f.write((const char*)ptr, size);
            } catch (std::exception &e) {
                std::cerr << "Problem occurs in writing to the file " << " of size " << size << " at offset " << file_offset << " in file " 
                    << path << " pointer is " << (void *)ptr << std::endl;
                std::abort();
            }
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

// void veloc_ckpt_t::ckpt(const long ptr_id, size_t size, std::string path) {
//     try {
//         std::cout << "Got in ckpt fn for " << path << std::endl;
//         std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
//         // std::cout << "Got lock for " << path << std::endl;
//         const void* const ptr = reinterpret_cast<const void* const>(ptr_id);
//         while (!_pending_d2h.empty())
//             _cv_d2h.wait(_lock_d2h);
//         _pending_d2h.push_back(std::make_tuple(path, ptr, size));
//         // std::cout << "Pushed in queue for " << path << std::endl;
//         _lock_d2h.unlock();
//         _cv_d2h.notify_all();
//         std::cout << "Saving started in CPP for " << path << std::endl;
//         return;
//     } catch (...) {
//         std::cerr << "Unknown exception caught in ckpt." << path << std::endl;
//         throw std::runtime_error("Unknown exception");
//         std::abort();
//     }

// }

void veloc_ckpt_t::ckpt_header_size(const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t value, std::string path) {
    try {
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

void veloc_ckpt_t::ckpt_pickle(const std::uint64_t start_offset, const std::uint64_t end_offset, py::bytes value, std::string path) {
    try {
        const char* data = PyBytes_AsString(value.ptr());
        size_t size = PyBytes_Size(value.ptr());
        assert((size == end_offset-start_offset) && "Size of pickled object is not correct");
        std::ofstream f;
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        f.open(path,  std::ofstream::out | std::ofstream::binary | std::ofstream::app);
        f.seekp(start_offset);
        f.write((const char*)data, size);
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

void veloc_ckpt_t::ckpt_obj(const std::uint64_t start_offset, const std::uint64_t end_offset, const std::uint64_t ptr_id, const std::uint64_t size, const int device_id, const std::uint64_t file_offset, std::string path) {
    try {
        const void* const ptr = reinterpret_cast<const void* const>(ptr_id);
        cudaPointerAttributes attr;
        checkCuda(cudaPointerGetAttributes(&attr, (const void *)ptr));
        if (attr.type == cudaMemoryTypeDevice)
            assert((attr.type == cudaMemoryTypeDevice && device_id > -1 && device_id == _gpu_id) && "Device pointer problem in ckpt_obj");
        if (attr.type == cudaMemoryTypeDevice && device_id > -1) {
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            while (!_pending_d2h.empty())
                _cv_d2h.wait(_lock_d2h);
            _pending_d2h.push_back(std::make_tuple(path, ptr, size, file_offset));
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            std::cout << "Saving started for GPU data structure in CPP for " << path << std::endl;
            return;
        } 
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        _pending_h2f.push_back(std::make_tuple(path, ptr, size, file_offset));
        _lock_h2f.unlock();
        _cv_h2f.notify_all();
        std::cout << "Saving started for host data structure in CPP for " << path << std::endl;
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

void veloc_ckpt_t::wait() {
    // _lock_d2h.lock();
    std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
    while (!_pending_d2h.empty())
        _cv_d2h.wait(_lock_d2h);
    _lock_d2h.unlock();
    _cv_d2h.notify_all();
    std::cout << "Wait complete" << std::endl;
    
}
