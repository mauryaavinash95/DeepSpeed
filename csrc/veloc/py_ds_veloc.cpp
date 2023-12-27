#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "deepspeed_py_veloc.h"

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<veloc_ckpt_t>(m, "veloc_ckpt_handle")
        .def(py::init<const size_t, int, int>())
        .def("ckpt_obj", &veloc_ckpt_t::ckpt_obj, py::call_guard<py::gil_scoped_release>())
        .def("ckpt_tensor", &veloc_ckpt_t::ckpt_tensor, py::call_guard<py::gil_scoped_release>())
        .def("wait", &veloc_ckpt_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &veloc_ckpt_t::shutdown);
}
