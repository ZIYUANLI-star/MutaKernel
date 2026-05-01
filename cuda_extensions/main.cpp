#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", torch::wrap_pybind_function(forward), "forward");
}