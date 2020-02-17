#include <torch/extension.h>
#include <vector>

at::Tensor Aggregate_Forward_CUDA(
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_);

std::vector<at::Tensor> Aggregate_Backward_CUDA(
  const at::Tensor GE_,
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_);
