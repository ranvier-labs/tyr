#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include <ATen/SDPBackend.h>
#include <ATen/ops/_fused_sdp_choice.h>
#include <torch/torch.h>

namespace tyr_ops {

enum class AttentionMaskKind { Padding, Edges };

// These bound temporary fallback scores/masks, rather than imposing a context
// limit. A single query row remains the minimum unit of work.
constexpr int64_t attention_query_chunk = 256;
constexpr int64_t attention_score_elements = 4 * 1024 * 1024;

inline bool has_fused_sdpa(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const std::optional<torch::Tensor>& mask, double dropout_p, bool causal,
    const std::optional<double>& scale, bool gqa) {
  // The selector has CPU/CUDA implementations. Other devices use the bounded
  // path below, letting their normal SDPA dispatcher choose the actual kernel.
  if (!q.is_cpu() && !q.is_cuda()) return false;
  const auto backend = at::_fused_sdp_choice(q, k, v, mask, dropout_p, causal, scale, gqa);
  return backend != static_cast<int64_t>(at::SDPBackend::math) &&
      backend != static_cast<int64_t>(at::SDPBackend::error);
}

inline torch::Tensor sdpa_chunk(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const std::optional<torch::Tensor>& mask, double dropout_p,
    const std::optional<double>& scale, bool enable_gqa) {
  if (q.size(1) == k.size(1) || !enable_gqa ||
      has_fused_sdpa(q, k, v, mask, dropout_p, false, scale, true)) {
    return torch::scaled_dot_product_attention(q, k, v, mask, dropout_p, false, scale, enable_gqa);
  }
  // The math GQA implementation repeats K/V. Process individual query heads
  // against views of their KV head instead, retaining autograd without a KV copy.
  const auto group = q.size(1) / k.size(1);
  std::vector<torch::Tensor> heads;
  heads.reserve(q.size(1));
  for (int64_t h = 0; h < q.size(1); ++h) {
    const auto kvh = h / group;
    heads.push_back(torch::scaled_dot_product_attention(
        q.narrow(1, h, 1), k.narrow(1, kvh, 1), v.narrow(1, kvh, 1),
        mask, dropout_p, false, scale));
  }
  return torch::cat(heads, 1);
}

// The optional mask contains 0 for forbidden edges and nonzero for allowed
// edges: [B,K] padding or [B,Q,K] explicit edges. Causality has SDPA's upper-left
// alignment, including rectangular Q/K lengths. A window is always causal and
// includes the current token (the historical GQAWindow contract).
inline torch::Tensor bounded_sdpa(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const std::optional<torch::Tensor>& mask, AttentionMaskKind mask_kind,
    double dropout_p, bool is_causal, const std::optional<double>& scale,
    bool enable_gqa, std::optional<int64_t> window = std::nullopt) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
      "tyr::sdpa: Q/K/V must be rank-4");
  TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0) &&
      k.size(1) == v.size(1) && k.size(2) == v.size(2),
      "tyr::sdpa: Q/K/V batch or KV shape mismatch");
  TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0,
      "tyr::sdpa: attention requires positive head counts");
  TORCH_CHECK(!enable_gqa || q.size(1) % k.size(1) == 0,
      "tyr::sdpa: query heads must be divisible by KV heads for GQA");
  TORCH_CHECK(!window || *window >= 0, "tyr::sdpa: window must be nonnegative");
  const auto q_len = q.size(2), kv_len = k.size(2);
  const bool causal = is_causal || window.has_value();
  std::optional<torch::Tensor> source_mask;
  if (mask && mask->defined()) {
    const bool padding = mask_kind == AttentionMaskKind::Padding;
    TORCH_CHECK(mask->dim() == (padding ? 2 : 3) && mask->size(0) == q.size(0) &&
        mask->size(-1) == kv_len && (padding || mask->size(1) == q_len),
        "tyr::sdpa: mask must have shape [B,K] (padding) or [B,Q,K] (edges)");
    source_mask = mask->to(q.device());
  }
  // Keep implicit causality and native GQA for fused kernels. Backend selection
  // only reads user preferences; no process-global backend flags are changed.
  if (!source_mask && !window &&
      has_fused_sdpa(q, k, v, std::nullopt, dropout_p, causal, scale, enable_gqa)) {
    return torch::scaled_dot_product_attention(
        q, k, v, std::nullopt, dropout_p, causal, scale, enable_gqa);
  }
  if (q_len == 0 || kv_len == 0 || q.size(0) == 0) {
    return torch::scaled_dot_product_attention(
        q, k, v, std::nullopt, dropout_p, causal, scale, enable_gqa);
  }
  // Account for scores across batch and heads. For window attention only the
  // keys reachable by this query chunk are passed to SDPA.
  const auto max_keys = window && *window > 0
      ? std::min(kv_len, *window + std::min(attention_query_chunk, q_len) - 1)
      : kv_len;
  const auto row_elements = std::max<int64_t>(1, q.size(0) * q.size(1) * max_keys);
  const auto chunk = std::max<int64_t>(1,
      std::min(attention_query_chunk, attention_score_elements / row_elements));
  const auto index_options = torch::TensorOptions().dtype(torch::kLong).device(q.device());
  std::vector<torch::Tensor> pieces;
  for (int64_t start = 0; start < q_len; start += chunk) {
    const auto end = std::min(q_len, start + chunk);
    const auto key_end = causal ? std::min(kv_len, end) : kv_len;
    const auto key_start = window && *window > 0
        ? std::min(key_end, std::max<int64_t>(0, start - *window + 1)) : 0;
    std::optional<torch::Tensor> allowed;
    if (causal) {
      const auto rows = torch::arange(start, end, index_options).unsqueeze(1);
      const auto cols = torch::arange(key_start, key_end, index_options).unsqueeze(0);
      auto edges = cols <= rows;
      if (window) edges = edges & ((rows - cols) < *window);
      allowed = edges.unsqueeze(0).unsqueeze(0);
    }
    if (source_mask) {
      auto edges = source_mask->slice(-1, key_start, key_end);
      edges = mask_kind == AttentionMaskKind::Padding
          ? edges.unsqueeze(1).unsqueeze(2)
          : edges.slice(1, start, end).unsqueeze(1);
      edges = edges != 0;
      allowed = allowed ? (*allowed & edges) : edges;
    }
    pieces.push_back(sdpa_chunk(q.slice(2, start, end),
        k.slice(2, key_start, key_end), v.slice(2, key_start, key_end),
        allowed, dropout_p, scale, enable_gqa));
  }
  return torch::cat(pieces, 2);
}

} // namespace tyr_ops
