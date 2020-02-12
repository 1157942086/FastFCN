#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregate_forward", &Aggregate_Forward_CPU, "Aggregate forward (CPU)");
  m.def("aggregate_backward", &Aggregate_Backward_CPU, "Aggregate backward (CPU)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CPU, "ScaledL2 forward (CPU)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CPU, "ScaledL2 backward (CPU)");
}
