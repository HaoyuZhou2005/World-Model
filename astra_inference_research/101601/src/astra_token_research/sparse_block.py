"""REAL selected-query computation for the checked Astra DiTBlockWithMoE.

Selected tokens: self attention Q is sparse; K/V remain *full sequence*.
Selected tokens also use Q-sparse text cross-attention and selected-row FFN.
Unselected tokens: the prior denoising-step cached block residual is reused.

IMPORTANT:
- This is not full attention KV reuse, no cross-chunk cache, and no
  "compute all rows then mask" implementation.
- Full K/V projection, all-token layernorm and upstream MoE remain expensive.
- Numerically matches selected rows of the original block *for the same full
  current input* (up to kernel tolerance), not the hypothetical full diffusion
  trajectory after previous approximate steps.
- Fail closed on unsupported configurations; B=1 only.
"""
from __future__ import annotations

from typing import Any
import torch


def _upstream_helpers(block: Any):
    namespace = getattr(block.self_attn.forward, "__globals__", {})
    rope_apply = namespace.get("rope_apply")
    modulate = namespace.get("modulate")
    if not callable(rope_apply) or not callable(modulate):
        raise RuntimeError(
            "Unexpected Astra attention implementation; no verified rope_apply/modulate "
            "in original self-attention module. Refuse to approximate full attention.")
    return rope_apply, modulate


def selected_block_rows(block: Any, x: torch.Tensor, indices: torch.Tensor,
                        context: Any, cam_emb: Any, t_mod: torch.Tensor,
                        freqs: torch.Tensor, modality_inputs: Any = None,
                        router_weights: Any = None,
                        router_indices: Any = None) -> torch.Tensor:
    """Compute only the indicated output rows, shape [1, M, D].

    Matches upstream DiTBlockWithMoE.forward exactly in mathematical ops for
    those rows, except flash attention may use a different kernel/numeric order.
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("Sparse Astra prototype supports B=1 only")
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise TypeError("indices must be 1D LongTensor")
    if not indices.numel():
        raise ValueError("Never call sparse block kernel with empty selection")
    if int(indices.min()) < 0 or int(indices.max()) >= x.shape[1]:
        raise IndexError("Selected token index outside packed sequence")
    if freqs.ndim != 4 or freqs.shape[0] != 1 or freqs.shape[1] != x.shape[1]:
        raise ValueError("RoPE freq shape must match exact packed full token sequence")
    if context is None:
        raise ValueError("Text context is missing; use original Astra model forward")
    if not hasattr(block, "projector"):
        raise RuntimeError("Astra camera projector absent: architecture not initialized")
    rope_apply, modulate = _upstream_helpers(block)
    sh_msa, sc_msa, gate_msa, sh_mlp, sc_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=1)

    # K/V depend on *all* incoming tokens. Reducing K/V to selected tokens
    # would silently change all selected-query attention outputs.
    full_input = modulate(block.norm1(x), sh_msa, sc_msa)

    if (getattr(block, "use_moe", False) and modality_inputs
            and hasattr(block, "moe") and router_weights is not None):
        combined = None
        active_modality = "unknown"
        for name, tensor in modality_inputs.items():
            active_modality = name
            combined = tensor if combined is None else combined + tensor
        if combined is not None:
            # Full MoE is necessary for K/V of every token.
            moe_output, stats = block.moe(
                combined, router_weights, router_indices, active_modality)
            full_input = full_input + moe_output
            if not hasattr(block, "expert_stats_buffer"):
                block.expert_stats_buffer = []
            block.expert_stats_buffer.append(stats)
    elif cam_emb is not None and hasattr(block, "cam_encoder"):
        encoded = block.cam_encoder(cam_emb.to(block.cam_encoder.weight.dtype))
        full_input = full_input + encoded

    full_input = full_input.to(block.projector.weight.dtype)
    sa = block.self_attn
    full_input = full_input.to(sa.q.weight.dtype)
    picked = full_input.index_select(1, indices)
    q = sa.norm_q(sa.q(picked))
    k = sa.norm_k(sa.k(full_input))
    v = sa.v(full_input)
    q_freq = freqs.index_select(1, indices)
    q = rope_apply(q, q_freq, sa.num_heads)
    k = rope_apply(k, freqs, sa.num_heads)
    # AttentionModule internally uses upstream flash_attention with
    # Q length M and K/V length S; no causal=True is silently introduced.
    attn = sa.attn(q, k, v)
    attn_output = sa.o(attn).to(block.projector.weight.dtype)
    chosen_x = x.index_select(1, indices)
    chosen_x = chosen_x + gate_msa * block.projector(attn_output)
    chosen_x = chosen_x.to(block.norm3.weight.dtype)
    # Cross attention is query-row separable; its keys are text context.
    chosen_x = chosen_x + block.cross_attn(block.norm3(chosen_x), context)
    chosen_mlp = modulate(block.norm2(chosen_x), sh_mlp, sc_mlp)
    chosen_x = chosen_x + gate_mlp * block.ffn(chosen_mlp)
    if chosen_x.shape != (1, len(indices), x.shape[2]):
        raise RuntimeError("Sparse output shape mismatch")
    return chosen_x
