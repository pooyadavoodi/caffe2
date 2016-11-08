#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/accuracy_op.h"
#include "caffe2/utils/math.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>

namespace caffe2 {

template <>
bool AccuracyOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  Y->Resize(vector<TIndex>());
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CUDAContext>(1, 0, Ydata, &context_);
  const int top_k = top_k_;

  thrust::device_ptr<const float> Xdata(X.data<float>());
  thrust::device_vector<float> Xdata_vector(N*D);
  CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(Xdata_vector.data()), thrust::raw_pointer_cast(Xdata), N*D*sizeof(float), cudaMemcpyDeviceToDevice, context_.cuda_stream()));

  thrust::device_ptr<const int> labelData(label.data<int>());
  thrust::device_vector<int> labelData_vector(N);
  CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(labelData_vector.data()), thrust::raw_pointer_cast(labelData), N*sizeof(int), cudaMemcpyDeviceToDevice, context_.cuda_stream()));

  thrust::device_vector<int> labelSeq(D);
  thrust::sequence(thrust::cuda::par.on(context_.cuda_stream()), labelSeq.begin(), labelSeq.end());

  thrust::device_vector<int> sortedLabelSeq(D);

  int correct = 0;

  for (int i = 0; i < N; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(sortedLabelSeq.data()), thrust::raw_pointer_cast(labelSeq.data()), D*sizeof(int), cudaMemcpyDeviceToDevice, context_.cuda_stream()));
    thrust::sort_by_key(thrust::cuda::par.on(context_.cuda_stream()), Xdata_vector.begin() + i*D, Xdata_vector.begin() + (i+1)*D, sortedLabelSeq.begin(), thrust::greater<float>());
    auto foundIter = thrust::find(thrust::cuda::par.on(context_.cuda_stream()), sortedLabelSeq.begin(), sortedLabelSeq.begin() + top_k, *(labelData_vector.begin() + i));
    if (foundIter != (sortedLabelSeq.begin()+top_k)) {
      correct++;
    }
  }
  DCHECK_LE(correct, N);
  const float accuracy = static_cast<float>(correct) / N;
  cudaMemcpyAsync(Y->mutable_data<float>(), &accuracy, sizeof(float), cudaMemcpyHostToDevice, context_.cuda_stream());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Accuracy, AccuracyOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
