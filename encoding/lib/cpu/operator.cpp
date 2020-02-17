#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregate_forward", &Aggregate_Forward_CPU, "Aggregate forward (CPU)");
  m.def("aggregate_backward", &Aggregate_Backward_CPU, "Aggregate backward (CPU)");
}
