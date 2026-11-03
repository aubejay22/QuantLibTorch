#pragma once

#include <c10/util/Exception.h>
#include <cstdlib>
#include <cstring>

namespace quantlibtorch {
// Process-wide, immutable diagnostic switches. Set before loading the native
// library. They permit controlled ablations without changing financial inputs,
// the scalar ABI, or the upstream source. Defaults enable both optimizations.
inline bool performance_flag(const char* name) {
    const char* value = std::getenv(name);
    TORCH_CHECK(!value || std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
                name, " must be 0 or 1 (set before importing quantlibtorch)");
    return !value || std::strcmp(value, "1") == 0;
}
inline bool passive_constants_enabled() {
    static const bool enabled = performance_flag("QLT_PASSIVE_CONSTANTS");
    return enabled;
}
inline bool sample_cache_enabled() {
    static const bool enabled = performance_flag("QLT_SAMPLE_CACHE");
    return enabled;
}
inline bool native_adjoint_enabled() {
    static const bool enabled = performance_flag("QLT_NATIVE_ADJOINT");
    return enabled;
}
inline bool packed_arrays_enabled() {
    static const bool enabled = performance_flag("QLT_PACKED_ARRAYS");
    return enabled;
}
inline bool lazy_arrays_enabled() {
    static const bool enabled = performance_flag("QLT_LAZY_ARRAYS");
    return enabled;
}
inline bool vector_sampling_enabled() {
    static const bool enabled = performance_flag("QLT_VECTOR_SAMPLING");
    return enabled;
}
inline bool array_kernels_enabled() {
    static const bool enabled = performance_flag("QLT_ARRAY_KERNELS");
    return enabled;
}
} // namespace quantlibtorch
