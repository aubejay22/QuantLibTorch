// Build provenance only; no numerical operations or scalar-adapter changes.
#include <torch/library.h>
#include <quantlibtorch_build_info.hpp>
#include <string>

namespace {
std::string build_info() { return QLT_BUILD_INFO; }
}

TORCH_LIBRARY_FRAGMENT(quantlibtorch, m) {
    m.def("build_info() -> str", &build_info);
}
