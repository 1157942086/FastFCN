#include <ATen/ATen.h>
#include <vector>

at::Tensor Aggregate_Forward_CPU(
    const at::Tensor A,
    const at::Tensor X,
    const at::Tensor C) {
  auto E = (A.unsqueeze(3) * (X.unsqueeze(2).expand({X.size(0), X.size(1),
    C.size(0), C.size(1)}) - C.unsqueeze(0).unsqueeze(0))).sum(1);
  return E;
}

std::vector<at::Tensor> Aggregate_Backward_CPU(
    const at::Tensor GE,
    const at::Tensor A,
    const at::Tensor X,
    const at::Tensor C) {
  auto gradA = (GE.unsqueeze(1) * (X.unsqueeze(2).expand({X.size(0), X.size(1),
    C.size(0), C.size(1)}) - C.unsqueeze(0).unsqueeze(0))).sum(3);
  auto gradX = at::bmm(A, GE);
  auto gradC = (-GE * A.sum(1).unsqueeze(2)).sum(0);
  return {gradA, gradX, gradC};
}
