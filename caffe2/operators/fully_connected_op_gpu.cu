#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

namespace {

// Image is an MxN matrix in column-major
// Or: bias_channels x image_size
template <typename T_B, typename M, typename T_Y>
__global__
void FCBiasAddKernel(const T_B* bias,
                   	 const int image_rows,
                     const int image_cols,
                     T_Y *image) {
  // block per row (row-major)
  const int idx = blockIdx.x;

  // transpose image, so row starts at idx*image_rows
  T_Y* image_row = &image[idx*image_cols];

  // transpose - instead of over cols, over rows
  for (int column=threadIdx.x;
           column < image_cols;
           column += blockDim.x) {

    const M bias_val = convert::To<T_B, M>(bias[column]);
    M current_val = convert::To<T_Y, M>(image_row[column]);
    current_val += bias_val;
    image_row[column] = convert::To<M,T_Y>(current_val);
  }
}

};

template <>
template <typename T_B,
          typename MATH,
          typename T_Y>
void FullyConnectedOp<CUDAContext>::AddBiasWithType() {
  auto& X = Input(0);
  auto& W = Input(1);
  auto& b = Input(2);
  auto* Y = Output(0);

  auto canonical_axis = X.canonical_axis_index(axis_);
  auto M = X.size_to_dim(canonical_axis);
  auto N = W.dim32(0);

  const auto grid_size = M;

  FCBiasAddKernel<T_B, MATH, T_Y>
        <<<grid_size,
           CAFFE_CUDA_NUM_THREADS,
           0, context_.cuda_stream()>>>(
                b.template data<T_B>(),
                M,
                N,
                Y->template mutable_data<T_Y>());
}

template <>
bool FullyConnectedOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, // X
        float16, // W
        float16, // B
        float16, // Y
        float>(); // Math
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 inputs "
               << "are supported by FullyConnectedOp, "
               << "but input " << debug_def().input(0) << " has ["
               << Input(0).meta().name() << "] ";
  }
  return false;
}

template <>
bool FullyConnectedGradientOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, //  X
        float16, //  W
        float16, // dY
        float16, //  B
        float16, // dX
        float16, // dW
        float16, // dB
        float>(); // Math
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 inputs "
               << "are supported by FullyConnectedGradientOp, "
               << "but input " << debug_def().input(0) << " has ["
               << Input(0).meta().name() << "] ";
  }
  return false;
}

#if CUDA_VERSION >= 9000


template <>
template <typename T_B,
          typename MATH,
          typename T_Y>
void FullyConnectedOp<CUDAContext, TensorCoreEngine>::AddBiasWithType() {
  auto& X = Input(0);
  auto& W = Input(1);
  auto& b = Input(2);
  auto* Y = Output(0);

  auto canonical_axis = X.canonical_axis_index(axis_);
  auto M = X.size_to_dim(canonical_axis);
  auto N = W.dim32(0);

  const auto grid_size = M;

  FCBiasAddKernel<T_B, MATH, T_Y>
        <<<grid_size,
           CAFFE_CUDA_NUM_THREADS,
           0, context_.cuda_stream()>>>(
                b.template data<T_B>(),
                M,
                N,
                Y->template mutable_data<T_Y>());
}

// Require these to be defined otherwise TensorCore FC ops will end
// up calling the default FC implementation which doesn't have
// fp16 support...
template <>
bool FullyConnectedOp<CUDAContext, TensorCoreEngine>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, // X
        float16, // W
        float16, // B
        float16, // Y
        float>(); // Math
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 inputs "
               << "are supported by FullyConnectedOp with TensorCoreEngine, "
               << "but input " << debug_def().input(0) << " has ["
               << Input(0).meta().name() << "] ";
  }
  return false;
}

template <>
bool FullyConnectedGradientOp<CUDAContext, TensorCoreEngine>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<
        float16, //  X
        float16, //  W
        float16, // dY
        float16, //  B
        float16, // dX
        float16, // dW
        float16, // dB
        float>(); // Math
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 inputs "
               << "are supported by FullyConnectedGradientOp "
               << "with TensorCoreEngine, "
               << "but input " << debug_def().input(0) << " has ["
               << Input(0).meta().name() << "] ";
  }
  return false;
}

#endif

REGISTER_CUDA_OPERATOR(FC, FullyConnectedOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(FCGradient, FullyConnectedGradientOp<CUDAContext>);

#if CUDA_VERSION >= 9000
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    FC,
    TENSORCORE,
    FullyConnectedOp<CUDAContext, TensorCoreEngine>);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    FCGradient,
    TENSORCORE,
    FullyConnectedGradientOp<CUDAContext, TensorCoreEngine>);
#endif

}  // namespace caffe2
