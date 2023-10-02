#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "deepspeed_py_veloc.h"

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<veloc_ckpt_t>(m, "veloc_ckpt_handle")
        // Host cache, persistent storage path
        .def(py::init<const size_t>())
        .def("ckpt", &veloc_ckpt_t::ckpt)
        .def("wait", &veloc_ckpt_t::wait);
}
