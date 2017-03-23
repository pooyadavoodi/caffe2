#include "caffe2/operators/distance_op.h"

namespace caffe2 {

template<>
bool SquaredL2DistanceOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = X.size() / N;
  distance->Resize(N);
  float* distance_data = distance->mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  for (int i = 0; i < N; ++i) {
    float Xscale, Yscale, cross;
    math::Dot<float, CPUContext>(
        D, X_data + i * D, X_data + i * D, &Xscale, &context_);
    math::Dot<float, CPUContext>(
        D, Y_data + i * D, Y_data + i * D, &Yscale, &context_);
    math::Dot<float, CPUContext>(
        D, X_data + i * D, Y_data + i * D, &cross, &context_);
    distance_data[i] = (Xscale + Yscale) * 0.5 - cross;
  }
  return true;
}

template <>
bool DotProductOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto* result = Output(DOT_OUT);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N, D;
  if (X.size() > 0) {
    N = X.ndim() > 0 ? X.dim32(0) : 1;
    D = X.size() / N;
  } else {
    N = 0;
    D = 0;
  }
  result->Resize(N);
  float* result_data = result->mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offset = i * D;
    math::Dot<float, CPUContext>(
        D, X_data + offset, Y_data + offset, result_data + i, &context_);
  }
  return true;
}

template <>
bool CosineSimilarityOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto* result = Output(COS_OUT);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = X.size() / N;
  result->Resize(N);
  float* result_data = result->mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  float X2, Y2;
  const float kEps = 1e-12f;
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offset = i * D;
    math::Dot<float, CPUContext>(
        D, X_data + offset, X_data + offset, &X2, &context_);
    math::Dot<float, CPUContext>(
        D, Y_data + offset, Y_data + offset, &Y2, &context_);
    math::Dot<float, CPUContext>(
        D, X_data + offset, Y_data + offset, result_data + i, &context_);
    result_data[i] /= std::sqrt(std::max(X2, kEps) * std::max(Y2, kEps));
  }
  return true;
}

template <>
bool DotProductWithPaddingOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& Y = Input(Y_IN);
  auto* result = Output(DOT_OUT);
  CAFFE_ENFORCE_EQ(X.ndim(), Y.ndim());
  CAFFE_ENFORCE_EQ(X.dim32(0), Y.dim32(0));

  int N, D, DX, DY, restD;
  if (X.size() > 0) {
    N = X.ndim() > 0 ? X.dim32(0) : 1;
    DX = X.size() / N;
    DY = Y.size() / N;
  } else {
    N = 0;
    DX = 0;
    DY = 0;
  }

  D = std::min(DX, DY);
  restD = std::max(DX, DY) - D;
  result->Resize(N);
  float* result_data = result->mutable_data<float>();
  const float* X_data = X.data<float>();
  const float* Y_data = Y.data<float>();
  for (int i = 0; i < N; ++i) { // TODO: multithreading
    auto offsetX = i * DX, offsetY = i * DY;
    if (replicate_) {
      // L_ for longer vector and S_ for shorter vector
      const float *L_data, *S_data;
      int DL, DS;
      if (DX > DY) {
        L_data = X_data + offsetX;
        S_data = Y_data + offsetY;
        DL = DX;
        DS = DY;
      } else {
        L_data = Y_data + offsetY;
        S_data = X_data + offsetX;
        DL = DY;
        DS = DX;
      }
      float sum = 0.0;
      float tmp = 0.0;
      for (int j = 0; j < DL / DS; j++) {
        math::Dot<float, CPUContext>(
            DS, L_data + j * DS, S_data, &tmp, &context_);
        sum += tmp;
      }
      *(result_data + i) = sum;
    } else {
      math::Dot<float, CPUContext>(
          D, X_data + offsetX, Y_data + offsetY, result_data + i, &context_);
    }

    if (!replicate_ && DX != DY) {
      const float* rest_data;
      float rest_sum = 0;
      if (DX > DY) {
        rest_data = X_data + offsetX + D;
      } else {
        rest_data = Y_data + offsetY + D;
      }
      math::Sum<float, CPUContext>(restD, rest_data, &rest_sum, &context_);
      result_data[i] += rest_sum * pad_value_;
    }
  }
  return true;
}

namespace {
// L2
REGISTER_CPU_OPERATOR(SquaredL2Distance,
                      SquaredL2DistanceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SquaredL2DistanceGradient,
                      SquaredL2DistanceGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SquaredL2Distance)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
  Given two input float tensors X, Y, and produces one output float tensor
  of the L2 difference between X and Y that is computed as ||(X - Y)^2 / 2||.
  )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

OPERATOR_SCHEMA(SquaredL2DistanceGradient).NumInputs(3).NumOutputs(2);

class GetSquaredL2DistanceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SquaredL2DistanceGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(SquaredL2Distance, GetSquaredL2DistanceGradient);

// Dot Product
REGISTER_CPU_OPERATOR(DotProduct, DotProductOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    DotProductGradient,
    DotProductGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(DotProduct)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
  Given two input float tensors X, Y, and produces one output float tensor
  of the dot product between X and Y.
  )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

OPERATOR_SCHEMA(DotProductGradient).NumInputs(3).NumOutputs(2);

class GetDotProductGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "DotProductGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(DotProduct, GetDotProductGradient);

// Cosine Similarity
REGISTER_CPU_OPERATOR(CosineSimilarity, CosineSimilarityOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    CosineSimilarityGradient,
    CosineSimilarityGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(CosineSimilarity)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
  Given two input float tensors X, Y, and produces one output float tensor
  of the cosine similarity between X and Y.
  )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

OPERATOR_SCHEMA(CosineSimilarityGradient).NumInputs(3).NumOutputs(2);

class GetCosineSimilarityGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CosineSimilarityGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(CosineSimilarity, GetCosineSimilarityGradient);

// Dot Product allows padding
REGISTER_CPU_OPERATOR(
    DotProductWithPadding,
    DotProductWithPaddingOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    DotProductWithPaddingGradient,
    DotProductWithPaddingGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(DotProductWithPadding)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Given two input float tensors X, Y with different shapes and produces one
  output float tensor of the dot product between X and Y. We currently support
  two kinds of strategies to achieve this. Before doing normal dot_product 1)
  pad the smaller tensor (using pad_value) to the same shape as the other one.
  2) replicate the smaller tensor to the same shape as the other one.
  )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor")
    .Arg("pad_value", "the padding value for tensors with smaller dimension")
    .Arg("replicate", "wehther to replicate the smaller tensor or not");

OPERATOR_SCHEMA(DotProductWithPaddingGradient).NumInputs(3).NumOutputs(2);

class GetDotProductWithPaddingGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    float pad_value = 0;
    bool replicate = false;
    if (HasArgument(Def(), "pad_value")) {
      pad_value = GetArgument(Def(), "pad_value").f();
    }
    if (HasArgument(Def(), "replicate")) {
      replicate = GetArgument(Def(), "replicate").i();
    }

    const auto dot_arg =
        vector<Argument>{MakeArgument<float>("pad_value", pad_value),
                         MakeArgument<bool>("replicate", replicate)};

    return SingleGradientDef(
        "DotProductWithPaddingGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)},
        dot_arg);
  }
};
REGISTER_GRADIENT(DotProductWithPadding, GetDotProductWithPaddingGradient);
}  // namespace
}  // namespace caffe2
