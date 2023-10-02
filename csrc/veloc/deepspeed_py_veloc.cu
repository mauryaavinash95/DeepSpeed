#include <deepspeed_py_veloc.h>

// void veloc_ckpt_t::ckpt(py::dict &m, std::string path) {
//     c10::impl::GenericDict generic_dict(c10::StringType::get(), m.begin()->second.type());
//     for (const auto& entry : m) {
//         generic_dict.insert(entry.first, entry.second);
//     }
//     torch::serialize::OutputArchive output_archive;
//     output_archive.write("data", generic_dict);
//     output_archive.save_to(path);
//     std::cout << "Saving complete from CPP size for " << path << std::endl;
//     return;
// }

// Helper function to convert Python objects to c10::IValue
// at::IValue convert_to_ivalue(const py::handle &obj) {
//     if (py::isinstance<py::float_>(obj)) {
//         return at::IValue(py::cast<float>(obj));
//     } else if (py::isinstance<py::int_>(obj)) {
//         return at::IValue(py::cast<int>(obj));
//     } else if (py::isinstance<py::list>(obj)) {
//         py::list py_list = py::cast<py::list>(obj);
//         std::vector<at::IValue> ivalue_list;
//         for (const auto &item : py_list) {
//             ivalue_list.push_back(convert_to_ivalue(item));
//         }
//         return at::IValue(ivalue_list);
//     } else if (py::isinstance<py::dict>(obj)) {
//         py::dict py_dict = py::cast<py::dict>(obj);
//         // c10::Dict<at::IValue, at::IValue> ivalue_dict;
//         c10::impl::GenericDict ivalue_dict(c10::StringType::get(), c10::IValue::type());
//         for (const auto &item : py_dict) {
//             ivalue_dict.insert(convert_to_ivalue(item.first), convert_to_ivalue(item.second));
//         }
//         return at::IValue(ivalue_dict);
//     }
//     // Handle other types as needed
//     throw std::runtime_error("Unsupported Python type");
// }

template <typename T>
c10::IValue convert_numpy_to_tensor(const py::handle& input) {
    if (py::isinstance<py::array_t<T>>(input)) {
        py::array_t<T> numpy_array = py::cast<py::array_t<T>>(input);
        auto numpy_array_info = numpy_array.request(); // Get array info
        // at::Tensor tensor = torch::from_blob(numpy_array_info.ptr, {numpy_array_info.size}, at::CppTypeToScalarType<T>::to());
        at::Tensor tensor = torch::from_blob(numpy_array_info.ptr, {numpy_array_info.size});
        return c10::IValue(tensor);
    }
    throw std::runtime_error("Input is not a NumPy array of the specified type.");
}

c10::IValue veloc_ckpt_t::convert_to_ivalue(const py::handle& input) {
    try {
        if (py::isinstance<py::none>(input)) 
            return c10::IValue();
        if (py::isinstance<py::str>(input))
            return c10::IValue(py::cast<py::str>(input));
        if (py::isinstance<py::float_>(input))
            return c10::IValue(py::cast<float>(input));
        if (py::isinstance<py::int_>(input))
            return c10::IValue(py::cast<int64_t>(input));
        // if (py::isinstance<torch::Tensor>(input) || py::isinstance<at::Tensor>(input))
        //     return c10::IValue(py::cast<at::Tensor>(input));
        // if (py::isinstance<py::object>(input) && py::hasattr(input, "dtype")) {
        //     // need this because torch.bfloat16 dtypes are not recognized by torch::Tensor
        //     at::Tensor tensor = py::cast<at::Tensor>(input); 
        //     return c10::IValue(tensor);
        // }
        if (py::isinstance(input, py::module::import("torch").attr("Tensor"))) 
            return c10::IValue(py::cast<at::Tensor>(input));
        // if (py::isinstance<py::array_t<double>>(input)) 
        //     return convert_numpy_to_tensor<double>(input);
        // if (py::isinstance<py::array_t<float>>(input)) 
        //     return convert_numpy_to_tensor<float>(input);
        if (py::isinstance<py::array>(input)) {
            py::array arr = py::cast<py::array>(input);
            auto numpy_array_info = arr.request();
            at::Tensor tensor = torch::from_blob(numpy_array_info.ptr, {numpy_array_info.size});
            return c10::IValue(tensor);
            // return convert_numpy_to_tensor<uint64_t>(input);
        }
        // if (py::isinstance<py::array_t<uint64_t>>(input)) 
        //     return convert_numpy_to_tensor<uint64_t>(input);
        // if (py::isinstance<py::array_t<int64_t>>(input)) 
        //     return convert_numpy_to_tensor<int64_t>(input);
        // if (py::isinstance<py::array_t<int>>(input)) 
        //     return convert_numpy_to_tensor<int>(input);
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
                // const std::string key = py::cast<std::string>(item.first);
                py::str key = py::str(item.first);
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
            std::cout << "Type is now " << py::str(input.get_type()) << " for input ." << input << "."<< std::endl;
            try {
                return c10::IValue(py::cast<py::str>(input));
            } catch(std::exception& e) {
                std::cerr << "Cannot convert unknown type to string:" << input << ". error " << e.what() << std::endl;
                throw std::runtime_error("Unexptected conversion");
            }
            // throw std::runtime_error("Unsupported Python type");
            // std::abort();
        }
    } catch(std::exception& e) {
        std::cout << "Input was |" << input << "|" << std::endl; 
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        // throw std::runtime_error("Standard exception");
        std::abort();
    } catch (...) {
        std::cerr << "Unknown exception caught." << input << std::endl;
        throw std::runtime_error("Unknown exception");
        std::abort();
    }
}

void veloc_ckpt_t::ckpt(py::dict &m, std::string path) {
    c10::IValue ivalue_dict = convert_to_ivalue(m);
    torch::serialize::OutputArchive output_archive;
    output_archive.write("data", ivalue_dict);
    output_archive.save_to(path);
    std::cout << "Saving complete from CPP size for " << path << std::endl;
    return;
}

void veloc_ckpt_t::wait(size_t tensor_id) {
    std::cout << "Not implemented yet" << std::endl;
}
