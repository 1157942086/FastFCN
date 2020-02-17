#include <torch/torch.h>
#include <vector>

at::Tensor Aggregate_Forward_CPU(
    const at::Tensor A,
    const at::Tensor X,
    const at::Tensor C);

std::vector<at::Tensor> Aggregate_Backward_CPU(
    const at::Tensor GE,
    const at::Tensor A,
    const at::Tensor X,
    const at::Tensor C);
