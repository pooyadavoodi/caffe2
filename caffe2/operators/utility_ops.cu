#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "utility_ops.h"

namespace caffe2 {

__global__ void
ElwiseMaxKernel(const float* X, const float* Y, float* maxout, const int N) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    maxout[i] = max(X[i], Y[i]);
  }
}

template <>
bool MaxOp<float, CUDAContext>::Compute() {
  float* output_data = Output(0)->mutable_data<float>();
  const int N = Input(0).size();

  // Run pairwise-maxes
  for (int i = 1; i < InputSize(); ++i) {
    ElwiseMaxKernel<<<
        CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        (i == 0 ? Input(0).data<float>() : Output(0)->data<float>()),
        Input(i).data<float>(),
        output_data,
        N);
  }

  return true;
}

REGISTER_CUDA_OPERATOR(Max, MaxOp<float, CUDAContext>);

template<typename T_INDEX>
__global__ void
GatherKernel(const float* X, float* Y, const T_INDEX* indices, const int N, const int block_size) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = indices[i];
    const float* src_offset = X + idx * block_size;
    float* dst_offset = Y + i   * block_size;
    for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
      dst_offset[j] = src_offset[j];
    }
  }
}

template <>
bool GatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t,int64_t>>::call(
      this, OperatorBase::Input<TensorCUDA>(INDICES));
}

template <>
template <typename Index>
bool GatherOp<CUDAContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
  auto shape = indices.dims();
  shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
  output->Resize(shape);

  int block_size = data.size() / data.dim(0);
  auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
  CAFFE_ENFORCE(
      block_bytesize == data.nbytes() / data.dim(0),
      "block_bytesize should be consistent with data dim");
  int N = indices.size();

  auto src_base = static_cast<const float*>(data.raw_data());
  const Index* idxs = indices.template data<Index>();
  auto out = static_cast<float*>(output->raw_mutable_data(data.meta()));

  GatherKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
        src_base, out, idxs, N, block_size
      );
  return true;
}

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);

/**
 * @brief Update slices of Y in-place with a batch of weighted X's.
 * Y[idx] = alpha[b] * X[b][i] + Y[idx]
 * i=0,...,N-1
 * b=0,...,B-1
 * idx=Indices[i]
 */
template<typename T_INDEX>
__global__ void 
AxpySliceKernel(
             const TIndex N,
             const TIndex B,
             const TIndex slice_size,
             const float** alpha,
             const float** X,
             const T_INDEX* Indices, 
             float* Y,
             const TIndex M) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = Indices[i];
    float* y_offset = Y + (idx * slice_size);
    for (int b = 0; b < B; b++) {
      const float* x_offset = X[b] + (i * slice_size);
      for (int j = threadIdx.x; j < slice_size; j += blockDim.x) {
        atomicAdd(&y_offset[j], (*alpha[b]) * x_offset[j]);
      }
    }
  }
}

template <>
bool ScatterWeightedSumOp<float,CUDAContext>::RunOnDevice() {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
}

template <>
template <typename Index>
bool ScatterWeightedSumOp<float,CUDAContext>::DoRunWithType() {
  DCHECK_EQ(InputSize() % 2, 1);
  auto& X0 = Input(0);
  auto& weight0 = Input(1);
  auto& indices = Input(2);
  auto* output = Output(0);

  CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");
  DCHECK_GT(X0.size(), 0);
  DCHECK_GT(X0.ndim(), 0) << "X0 has to be at least the vector";
  DCHECK_EQ(weight0.size(), 1);

  TIndex M = X0.size();
  TIndex N = X0.dim(0);
  TIndex K = indices.size();
  TIndex block_size = M / N;

  T* data = output->template mutable_data<T>();
  const Index* Indices = indices.template data<Index>();

  float w0 = *weight0.template data<float>();
  // It's most likely a constant so exact comparison is fine
  if (w0 != 1.0) {
    return false; //Not support for now
  }

  const TIndex B = (InputSize()-3)/2;
  VLOG(0) << "B: " << B;
  VLOG(0) << "block_size: " << block_size;

  const float** x_data_host;
  const float** x_data_device;
  const float** weights_host;
  const float** weights_device;

  x_data_host = (const float**) malloc(B * sizeof(const float*));
  x_data_device = (const float**) context_.New(B * sizeof(const float*));
  weights_host = (const float**) malloc(B * sizeof(const float*));
  weights_device = (const float**) context_.New(B * sizeof(const float*));

  for (int inp = 3; inp < InputSize(); inp += 2) {
    x_data_host [(inp-3)/2] = static_cast<const float*>(Input(inp).raw_data());
    weights_host[(inp-3)/2] = static_cast<const float*>(Input(inp+1).raw_data());
  }
  context_.Copy<const float*,CPUContext,CUDAContext>(B, weights_host, weights_device);
  context_.Copy<const float*,CPUContext,CUDAContext>(B, x_data_host, x_data_device);

  AxpySliceKernel<<<
    std::min<TIndex>(K, CAFFE_MAXIMUM_NUM_BLOCKS),
    CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>
    (
      K, B, block_size, weights_device, x_data_device, Indices, data, M
    );

  free(x_data_host);
  context_.Delete(x_data_device);
  free(weights_host);
  context_.Delete(weights_device);

  return true;
}

REGISTER_CUDA_OPERATOR(ScatterWeightedSum, ScatterWeightedSumOp<float,CUDAContext>);

}  // namespace caffe2
