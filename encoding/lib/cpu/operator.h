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

at::Tensor ScaledL2_Forward_CPU(
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor S_);

std::vector<at::Tensor> ScaledL2_Backward_CPU(
    const at::Tensor GSL_,
    const at::Tensor X_,
    const at::Tensor C_,
    const at::Tensor S_,
    const at::Tensor SL_);
