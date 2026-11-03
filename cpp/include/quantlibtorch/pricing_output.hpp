#pragma once

#include <quantlibtorch/performance.hpp>

namespace quantlibtorch {
inline bool price_only_enabled() {
    static const bool enabled = performance_flag("QLT_PRICE_ONLY");
    return enabled;
}
bool price_only();
// The Python functional bridge exposes NPV only. Other native engine users
// retain all upstream results; the scope always restores on return/exception.
class PriceOnlyMode {
  public:
    explicit PriceOnlyMode(bool enabled);
    ~PriceOnlyMode();
    PriceOnlyMode(const PriceOnlyMode&) = delete;
    PriceOnlyMode& operator=(const PriceOnlyMode&) = delete;
  private:
    bool previous_;
};
}
