#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "tyr_attention.h"

namespace {
using torch::Tensor;
using tyr_ops::AttentionMaskKind;

void close(const Tensor& actual, const Tensor& expected, const std::string& label,
           double tolerance = 1e-9) {
  TORCH_CHECK(torch::isfinite(actual).all().item<bool>(), label, ": nonfinite result");
  TORCH_CHECK(torch::allclose(actual, expected, tolerance, tolerance), label,
      ": max error ", (actual - expected).abs().max().item<double>());
}

Tensor reference(const Tensor& q, const Tensor& k, const Tensor& v,
                 const std::optional<Tensor>& mask, AttentionMaskKind kind,
                 bool causal, std::optional<int64_t> window = std::nullopt,
                 std::optional<double> scale = std::nullopt) {
  const auto repeats = q.size(1) / k.size(1);
  auto scores = torch::matmul(q, k.repeat_interleave(repeats, 1).transpose(-2, -1)) *
      scale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(-1))));
  auto allowed = torch::ones({q.size(0), 1, q.size(2), k.size(2)},
      q.options().dtype(torch::kBool));
  const auto indices = q.options().dtype(torch::kLong);
  auto rows = torch::arange(q.size(2), indices).unsqueeze(1);
  auto cols = torch::arange(k.size(2), indices).unsqueeze(0);
  if (causal || window) allowed = allowed & (cols <= rows);
  if (window) allowed = allowed & ((rows - cols) < *window);
  if (mask) allowed = allowed & (kind == AttentionMaskKind::Padding
      ? mask->unsqueeze(1).unsqueeze(2) != 0 : mask->unsqueeze(1) != 0);
  // Keep all-masked rows well-defined in both forward and backward.
  const auto any = allowed.any(-1, true);
  scores = scores.masked_fill(~allowed, -std::numeric_limits<double>::infinity());
  scores = torch::where(any, scores, torch::zeros_like(scores));
  auto weights = torch::softmax(scores, -1) * any.to(scores.scalar_type());
  return torch::matmul(weights, v.repeat_interleave(repeats, 1));
}

void parity(const torch::Device& device, int64_t q_len, int64_t kv_len,
            bool causal, AttentionMaskKind kind, bool masked,
            std::optional<int64_t> window, const std::string& label) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  auto q = torch::randn({2, 4, q_len, 8}, options).set_requires_grad(true);
  auto k = torch::randn({2, 2, kv_len, 8}, options).set_requires_grad(true);
  auto v = torch::randn({2, 2, kv_len, 8}, options).set_requires_grad(true);
  std::optional<Tensor> mask;
  if (masked) {
    mask = kind == AttentionMaskKind::Padding
        ? torch::randint(0, 2, {2, kv_len}, options.dtype(torch::kLong))
        : torch::randint(0, 2, {2, q_len, kv_len}, options.dtype(torch::kLong));
    mask->select(0, 0).zero_(); // Entirely masked batch, including gradients.
  }
  const std::optional<double> scale = 0.173;
  const auto actual = tyr_ops::bounded_sdpa(q, k, v, mask, kind, 0.0,
      causal, scale, true, window);
  const auto expected = reference(q, k, v, mask, kind, causal, window, scale);
  close(actual, expected, label);
  auto cotangent = torch::randn_like(actual);
  const auto actual_grad = torch::autograd::grad({actual}, {q, k, v}, {cotangent});
  const auto expected_grad = torch::autograd::grad({expected}, {q, k, v}, {cotangent});
  for (size_t i = 0; i < actual_grad.size(); ++i)
    close(actual_grad[i], expected_grad[i], label + " gradient " + std::to_string(i));
}

// Force the otherwise implicit fallback in this isolated test process. The
// production helper must respect and preserve this caller-selected configuration.
struct MathOnly {
  bool flash = at::globalContext().userEnabledFlashSDP();
  bool mem = at::globalContext().userEnabledMemEfficientSDP();
  bool cudnn = at::globalContext().userEnabledCuDNNSDP();
  bool math = at::globalContext().userEnabledMathSDP();
  MathOnly() {
    auto& c = at::globalContext();
    c.setSDPUseFlash(false); c.setSDPUseMemEfficient(false);
    c.setSDPUseCuDNN(false); c.setSDPUseMath(true);
  }
  ~MathOnly() {
    auto& c = at::globalContext();
    c.setSDPUseFlash(flash); c.setSDPUseMemEfficient(mem);
    c.setSDPUseCuDNN(cudnn); c.setSDPUseMath(math);
  }
};

void long_window_and_dropout(const torch::Device& device) {
  torch::NoGradGuard no_grad;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  constexpr int64_t length = 16385;
  auto q = torch::zeros({1, 4, length, 8}, options);
  auto k = torch::zeros({1, 2, length, 8}, options);
  auto positions = torch::arange(length, options);
  auto v = positions.view({1, 1, length, 1}).expand({1, 2, length, 8});
  auto actual = tyr_ops::bounded_sdpa(q, k, v, std::nullopt,
      AttentionMaskKind::Padding, 0.0, false, std::nullopt, true, 8);
  auto expected = (positions + torch::clamp_min(positions - 7, 0)) / 2;
  close(actual, expected.view({1, 1, length, 1}).expand_as(actual),
      "long window moving average", 1e-5);

  q = torch::zeros({1, 4, 17, 8}, options);
  k = torch::zeros({1, 2, 19, 8}, options);
  v = torch::ones_like(k);
  auto zero = tyr_ops::bounded_sdpa(q, k, v, std::nullopt,
      AttentionMaskKind::Padding, 1.0, false, std::nullopt, true);
  close(zero, torch::zeros_like(q), "dropout probability one");
  auto sum = torch::zeros_like(q);
  for (int i = 0; i < 128; ++i)
    sum += tyr_ops::bounded_sdpa(q, k, v, std::nullopt,
        AttentionMaskKind::Padding, 0.25, false, std::nullopt, true);
  close(sum / 128, torch::ones_like(q), "dropout expectation", 0.07);
}

void cuda_dtype_parity(const torch::Device& device) {
  if (!device.is_cuda()) return;
  torch::NoGradGuard no_grad;
  for (const auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
    const auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto q = torch::randn({1, 8, 263, 64}, options);
    auto k = torch::randn({1, 2, 289, 64}, options);
    auto v = torch::randn_like(k);
    // Compare the exact quantized inputs against independently accumulated FP32
    // attention. Float64 tests alone never qualify fused CUDA attention.
    const auto tolerance = dtype == torch::kFloat32 ? 2e-5 :
        (dtype == torch::kFloat16 ? 0.003 : 0.02);
    for (int case_id = 0; case_id < 3; ++case_id) {
      const auto window = case_id == 2 ? std::optional<int64_t>{31} : std::nullopt;
      const auto keys = case_id == 0 ? k.narrow(2, 0, q.size(2)) : k;
      const auto values = case_id == 0 ? v.narrow(2, 0, q.size(2)) : v;
      const auto actual = tyr_ops::bounded_sdpa(q, keys, values, std::nullopt,
          AttentionMaskKind::Padding, 0.0, true, std::nullopt, true, window);
      const auto expected = reference(q.to(torch::kFloat32), keys.to(torch::kFloat32),
          values.to(torch::kFloat32), std::nullopt, AttentionMaskKind::Padding, true, window);
      close(actual.to(torch::kFloat32), expected,
          "CUDA dtype " + std::string(c10::toString(dtype)) +
          (window ? " window" : (case_id == 0 ? " square causal GQA" : " rectangular causal GQA")),
          tolerance);
    }
  }
  std::cout << "CUDA float32/float16/bfloat16 attention parity passed\n";
}
} // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::manual_seed(714);
  const auto device = torch::Device(argc > 1 ? argv[1] : "cpu");
  cuda_dtype_parity(device);
  parity(device, 13, 13, true, AttentionMaskKind::Padding, false, std::nullopt, "native causal GQA");
  {
    MathOnly fallback;
    parity(device, 263, 289, true, AttentionMaskKind::Padding, false, std::nullopt, "chunked rectangular causal");
    parity(device, 269, 257, true, AttentionMaskKind::Padding, true, std::nullopt, "causal padding");
    parity(device, 263, 289, false, AttentionMaskKind::Padding, true, std::nullopt, "padding only");
    parity(device, 263, 289, false, AttentionMaskKind::Edges, true, std::nullopt, "explicit edge mask");
    parity(device, 263, 263, false, AttentionMaskKind::Padding, false, 7, "causal window across chunks");
    parity(device, 19, 19, false, AttentionMaskKind::Padding, false, 0, "zero window");
    parity(device, 1, 19, false, AttentionMaskKind::Padding, false, std::nullopt, "decode attends all cached keys");
    parity(device, 1, 19, true, AttentionMaskKind::Padding, false, std::nullopt, "rectangular upper-left causal alignment");
    long_window_and_dropout(device);
    const auto& c = at::globalContext();
    TORCH_CHECK(!c.userEnabledFlashSDP() && !c.userEnabledMemEfficientSDP() &&
        !c.userEnabledCuDNNSDP() && c.userEnabledMathSDP(), "SDPA changed backend flags");
  }
  std::cout << "Native attention value/gradient, chunking, window, dropout and backend checks passed on "
      << device << '\n';
}
