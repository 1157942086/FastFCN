#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

#include "common.h"
#include "device_tensor.h"

namespace {

template<typename DType, typename Acctype>
struct AggOp {
  __device__ AggOp(DeviceTensor<DType, 3> a,
                   DeviceTensor<DType, 3> x,
                   DeviceTensor<DType, 2> c) : A(a), X(x), C(c) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(A[b][i][k] * (X[b][i][d] - C[k][d]));
  }
  DeviceTensor<DType, 3> A;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
};

template<typename DType, typename Acctype>
struct AggBackOp {
  __device__ AggBackOp(DeviceTensor<DType, 3> g,
                       DeviceTensor<DType, 3> x,
                       DeviceTensor<DType, 2> c) : G(g), X(x), C(c) {}
  __device__ __forceinline__ Acctype operator()(int b, int i, int k, int d) {
    return ScalarConvert<DType, Acctype>::to(G[b][k][d] * (X[b][i][d] - C[k][d]));
  }
  DeviceTensor<DType, 3> G;
  DeviceTensor<DType, 3> X;
  DeviceTensor<DType, 2> C;
};

template<typename DType, typename Acctype>
__global__ void Aggregate_Forward_kernel (
    DeviceTensor<DType, 3> E,
    DeviceTensor<DType, 3> A,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C) {
  /* declarations of the variables */
  int b, k, d, N;
  /* Get the index and channels */ 
  b = blockIdx.z;
  d = blockIdx.x;
  k = blockIdx.y;
  N = X.getSize(1);
  /* main operation */
  AggOp<DType, Acctype> g(A, X, C);
  E[b][k][d] = reduceN<Acctype>(g, b, k, d, N);
}

template<typename DType, typename Acctype>
__global__ void Aggregate_Backward_kernel (
    DeviceTensor<DType, 3> GA,
    DeviceTensor<DType, 3> GE,
    DeviceTensor<DType, 3> A,
    DeviceTensor<DType, 3> X,
    DeviceTensor<DType, 2> C) {
  /* declarations of the variables */
  int b, k, i, D;
  /* Get the index and channels */ 
  b = blockIdx.z;
  i = blockIdx.y;
  k = blockIdx.x;
  D = GE.getSize(2);
  /* main operation */
  AggBackOp<DType, Acctype> g(GE, X, C);
  GA[b][i][k] = reduceD<Acctype>(g, b, i, k, D);
}

}// namespace

at::Tensor Aggregate_Forward_CUDA(
    const at::Tensor A_,
    const at::Tensor X_,
    const at::Tensor C_) {
  /* Device tensors */
  auto E_ = torch::zeros({A_.size(0), C_.size(0), C_.size(1)}, A_.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // B, K, D
  dim3 blocks(C_.size(1), C_.size(0), X_.size(0));
  dim3 threads(getNumThreads(X_.size(1)));

  AT_DISPATCH_FLOATING_TYPES(A_.type(), "Aggregate_Forward_CUDA", ([&] {
    DeviceTensor<scalar_t, 3> E = devicetensor<scalar_t, 3>(E_);
    DeviceTensor<scalar_t, 3> A = devicetensor<scalar_t, 3>(A_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    /* kernel function */
    Aggregate_Forward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>>(E, A, X, C);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return E_;
}

std::vector<at::Tensor> Aggregate_Backward_CUDA(
    const at::Tensor GE_,
    const at::Tensor A_,
    const at::Tensor X_,
    const at::Tensor C_) {
  auto gradA_ = at::zeros_like(A_);
  auto gradX_ = at::bmm(A_, GE_);
  auto gradC_ = (-GE_ * A_.sum(1).unsqueeze(2)).sum(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // B, K, D
  dim3 blocks(C_.size(0), X_.size(1), X_.size(0));
  dim3 threads(getNumThreads(C_.size(1)));
  AT_DISPATCH_FLOATING_TYPES(A_.type(), "Aggregate_Backward_CUDA", ([&] {
    /* Device tensors */
    DeviceTensor<scalar_t, 3> GA = devicetensor<scalar_t, 3>(gradA_);
    DeviceTensor<scalar_t, 3> GE = devicetensor<scalar_t, 3>(GE_);
    DeviceTensor<scalar_t, 3> A = devicetensor<scalar_t, 3>(A_);
    DeviceTensor<scalar_t, 3> X = devicetensor<scalar_t, 3>(X_);
    DeviceTensor<scalar_t, 2> C = devicetensor<scalar_t, 2>(C_);
    Aggregate_Backward_kernel<scalar_t, scalar_t>
      <<<blocks, threads, 0, stream>>> (GA, GE, A, X, C);
  }));
  AT_ASSERT(cudaGetLastError() == cudaSuccess);
  return {gradA_, gradX_, gradC_};
}
