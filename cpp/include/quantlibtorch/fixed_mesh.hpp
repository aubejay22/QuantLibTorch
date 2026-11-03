#pragma once
#include <ql/methods/finitedifferences/meshers/fdmmesher.hpp>

namespace quantlibtorch {
// An explicit fixed numerical mesh; null preserves the upstream automatic mesh.
QuantLib::ext::shared_ptr<QuantLib::FdmMesher> pricing_mesh_override();
class FixedMeshScope {
  public:
    explicit FixedMeshScope(QuantLib::ext::shared_ptr<QuantLib::FdmMesher> mesh);
    ~FixedMeshScope();
  private:
    QuantLib::ext::shared_ptr<QuantLib::FdmMesher> previous_;
};
}
