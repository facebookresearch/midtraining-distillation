# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from copy import deepcopy
import gc
import json
import logging
import math
import os
import re
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_global_rank,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from apps.main.transformer import (
    LMTransformerArgs,
    LMTransformer,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    tp_parallelize,
    get_no_recompute_ops,
)
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

from transformers import AutoModelForCausalLM

import wandb

logger = logging.getLogger()


@torch.compile(dynamic=False)
def compute_kl_distillation_loss(
    student_logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    teacher_logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    labels: torch.Tensor,  # (batch_size, seq_len)
    temperature: float = 1.0,
    alpha: float = 0.5,
    chunk_size: int = 128,  # Process this many tokens at a time to save memory
) -> tuple:
    """
    Memory-efficient Classic Knowledge Distillation: minimize KL divergence between student and teacher.

    Loss = α * KL(teacher || student) * τ² + (1 - α) * CE(student, labels)

    This implementation processes tokens in chunks along the sequence dimension
    to avoid materializing full (B, S, V) softmax tensors simultaneously.

    Args:
        student_logits: Student model logits (B, S, V)
        teacher_logits: Teacher model logits (B, S, V)
        labels: Target token IDs for hard label loss
        temperature: Temperature for softening distributions (higher = softer)
        alpha: Weight for distillation loss (1 - alpha = weight for hard label loss)
        chunk_size: Number of tokens to process at once (lower = less memory)

    Returns:
        (total_loss, stats_dict)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    # Hard label cross-entropy loss (memory efficient - doesn't need full vocab materialization)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, seq_len)
    ce_loss_mean = ce_loss.sum() / n_valid

    # Memory-efficient KL computation: process in chunks along sequence dimension
    kl_per_token = torch.zeros(
        batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype
    )

    # For logging - accumulate entropy stats
    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)

        # Get chunk slices
        student_chunk = student_logits[:, chunk_start:chunk_end, :]  # (B, chunk, V)
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]  # (B, chunk, V)
        mask_chunk = mask[:, chunk_start:chunk_end]  # (B, chunk)

        # Compute scaled log-softmax for student (needs gradients)
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)

        # Compute log-softmax for teacher (no gradients needed).
        # Using log_target=True avoids materializing a separate softmax tensor
        # and allows torch.compile to fuse exp(log_softmax) internally.
        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)

        # KL(teacher || student) = sum_v exp(teacher_log_soft) * (teacher_log_soft - student_log_soft)
        kl_chunk = F.kl_div(
            student_log_soft,
            teacher_log_soft,
            reduction="none",
            log_target=True,
        ).sum(
            dim=-1
        )  # (B, chunk)

        kl_per_token[:, chunk_start:chunk_end] = kl_chunk

        # Accumulate entropy statistics for logging (sampled from first chunk only to save compute)
        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    # Reuse already-computed log_softmax for entropy — avoids extra log() call.
                    teacher_ent = -(teacher_log_soft * torch.exp(teacher_log_soft)).sum(
                        dim=-1
                    )
                    student_ent = -(student_log_soft * torch.exp(student_log_soft)).sum(
                        dim=-1
                    )
                    teacher_entropy_sum = teacher_ent[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = chunk_mask.sum().item()

        # Free intermediate tensors
        del student_log_soft, teacher_log_soft, kl_chunk

    # Apply mask and compute mean KL
    kl_loss_mean = (kl_per_token * mask).sum() / n_valid

    # Scale KL by temperature² to maintain gradient magnitude
    kl_loss_scaled = kl_loss_mean * (temperature**2)

    # Combined loss
    total_loss = alpha * kl_loss_scaled + (1 - alpha) * ce_loss_mean

    with torch.no_grad():
        valid_kl = kl_per_token[mask.bool()]
        valid_ce = ce_loss[mask.bool()]

        stats = {
            "kd/kl_loss": kl_loss_scaled.item(),
            "kd/ce_loss": ce_loss_mean.item(),
            "kd/total_loss": total_loss.item(),
            "kd/kl_per_token_mean": valid_kl.mean().item(),
            "kd/kl_per_token_std": valid_kl.std().item(),
            "kd/ce_per_token_mean": valid_ce.mean().item(),
            "kd/teacher_entropy_mean": teacher_entropy_sum / max(n_entropy_samples, 1),
            "kd/student_entropy_mean": student_entropy_sum / max(n_entropy_samples, 1),
            "kd/temperature": temperature,
            "kd/alpha": alpha,
        }

    return total_loss, stats


def compute_reverse_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    chunk_size: int = 128,
) -> tuple:
    """Reverse-KL distillation (MiniLLM-style), same shape as kd-rl7b but with
    the *student* expectation:

        L = (1 - alpha) * CE(z_S, y)  +  alpha * T^2 * KL(p_S^T || p_T^T)

    where p_S^T = softmax(z_S / T) and p_T^T = softmax(z_T / T) are both at
    temperature T. The KL takes its expectation under p_S, not p_T:

        KL(p_S || p_T) = sum_v p_S(v) * (log p_S(v) - log p_T(v))

    Motivation (Gu et al. 2024, "MiniLLM"): forward KL is mode-covering -- it
    punishes the student for placing low mass where the teacher has high
    mass, which forces the student to spend capacity on *every* mode of the
    teacher, including high-entropy regions a smaller student cannot
    faithfully represent. Reverse KL is mode-seeking -- it punishes the
    student for placing high mass where the teacher has low mass, so the
    student can confidently concentrate on the dominant teacher mode(s) and
    ignore the long tail. For an LM with a capacity gap (1B student, 7B
    teacher), reverse KL is the theoretically-motivated divergence: it does
    not ask the student to do something it cannot do.

    Implementation:
      * Gradients flow through both p_S (target of the KL expectation) and
        log p_S (the denominator). Teacher log-softmax is detached.
      * Chunked along the seq dim, same memory profile as
        compute_kl_distillation_loss.
      * Per-chunk RKL uses F.kl_div(teacher_log_soft, student_log_soft,
        log_target=True) which is mathematically identical to the manual
        sum_v p_S * (log p_S - log p_T) (verified: bit-exact value AND
        gradient match the manual implementation). The fused kernel
        avoids explicitly materializing student_soft=exp(log_soft) in the
        autograd graph, reducing per-chunk memory footprint.
      * The RKL forward+backward is wrapped in torch.utils.checkpoint, so
        the per-chunk log-softmax / KL materializations are recomputed
        during backward instead of pinned across all chunks. This lets us
        train with a larger sequence-length / chunk-size budget without
        running into peak-activation OOM.
      * Telemetry (entropy, fkl-for-comparison) is computed under no_grad
        from the recomputed activations on the first chunk only, so it
        does not contribute to backward memory.

    Telemetry (kd_rkl/*):
      * rkl_loss      = alpha * T^2 * E[KL(p_S || p_T)]   (the actual loss term)
      * fkl_loss      = alpha * T^2 * E[KL(p_T || p_S)]   (the forward-KL
                        value at the same T on the same batch -- what
                        kd-rl7b sees. For comparison only; not in the loss.)
      * rkl_per_token_mean, fkl_per_token_mean, rkl/fkl ratio
      * teacher_entropy, student_entropy
      * temperature, alpha
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, seq_len)
    ce_loss_mean = ce_loss.sum() / n_valid

    rkl_per_token = torch.zeros(
        batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype
    )
    fkl_per_token = torch.zeros(
        batch_size, seq_len, device=student_logits.device, dtype=student_logits.dtype
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0
    masked_teacher_mass = 0.0  # teacher probability deleted by the mask (first chunk)

    # Inner KL kernel — wrapped in checkpoint to drop intermediate softmax
    # activations from peak memory. Returns the per-token RKL for this chunk.
    # NOTE: the function captures `temperature` from the enclosing scope; that
    # is safe because temperature is a Python float and not a tensor.
    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        # KL(p_S || p_T) via PyTorch's fused kernel. F.kl_div(input, target,
        # log_target=True, reduction='none') computes target.exp() * (target - input)
        # so passing (teacher_log_soft, student_log_soft) gives
        #   exp(student_log_soft) * (student_log_soft - teacher_log_soft)
        # which is the reverse-KL summand. Verified bit-exact vs manual.
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)

        # Activation-checkpointed RKL: forward materializes student_log_soft
        # inside the checkpoint and discards it; backward recomputes from
        # student_chunk. use_reentrant=False is required because we have
        # detached tensors in the closure (teacher_log_soft).
        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        # Forward KL for telemetry only. No-grad path so it does not affect
        # backward memory. Re-derive student_log_soft under no_grad just for
        # this telemetry compute.
        with torch.no_grad():
            student_log_soft_nograd = F.log_softmax(student_chunk / temperature, dim=-1)
            fkl_chunk = (
                teacher_log_soft.exp() * (teacher_log_soft - student_log_soft_nograd)
            ).sum(dim=-1)
            fkl_per_token[:, chunk_start:chunk_end] = fkl_chunk

            if chunk_start == 0:
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    teacher_ent = -(teacher_log_soft * teacher_log_soft.exp()).sum(
                        dim=-1
                    )
                    student_ent = -(
                        student_log_soft_nograd * student_log_soft_nograd.exp()
                    ).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del teacher_ent, student_ent
            del student_log_soft_nograd

        del teacher_log_soft, rkl_chunk, fkl_chunk

    rkl_loss_mean = (rkl_per_token * mask).sum() / n_valid
    rkl_loss_scaled = rkl_loss_mean * (temperature**2)
    total_loss = alpha * rkl_loss_scaled + (1.0 - alpha) * ce_loss_mean

    with torch.no_grad():
        valid_mask = mask.bool()
        valid_rkl = rkl_per_token[valid_mask]
        valid_fkl = fkl_per_token[valid_mask]
        valid_ce = ce_loss[valid_mask]
        ent_norm = max(n_entropy_samples, 1)
        rkl_mean = valid_rkl.mean().item() if valid_rkl.numel() > 0 else 0.0
        fkl_mean = valid_fkl.mean().item() if valid_fkl.numel() > 0 else 0.0
        # Reverse / forward asymmetry. <1 means RKL is "easier" than FKL on
        # this batch (teacher has long-tail mass the student can ignore);
        # >1 means RKL is "harder" (student has mass on tokens the teacher
        # gives low probability to -- usually means student is wrong here).
        rkl_to_fkl_ratio = (rkl_mean / fkl_mean) if fkl_mean > 1e-12 else 0.0

        stats = {
            "kd_rkl/rkl_loss": rkl_loss_scaled.item(),
            "kd_rkl/fkl_loss": (fkl_mean * (temperature**2)) * alpha,
            "kd_rkl/ce_loss": ce_loss_mean.item(),
            "kd_rkl/total_loss": total_loss.item(),
            "kd_rkl/rkl_per_token_mean": rkl_mean,
            "kd_rkl/rkl_per_token_std": (
                valid_rkl.std().item() if valid_rkl.numel() > 1 else 0.0
            ),
            "kd_rkl/fkl_per_token_mean": fkl_mean,
            "kd_rkl/rkl_to_fkl_ratio": rkl_to_fkl_ratio,
            "kd_rkl/ce_per_token_mean": (
                valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0
            ),
            "kd_rkl/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "kd_rkl/student_entropy_mean": student_entropy_sum / ent_norm,
            "kd_rkl/temperature": float(temperature),
            "kd_rkl/alpha": float(alpha),
            # Both 0 when masking is off, so a run's masking state is auditable from its
            # metrics alone without cross-referencing the config.
            "kd_rkl/masked_teacher_mass": masked_teacher_mass,
        }

    return total_loss, stats


def compute_switch_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    chunk_size: int = 128,
    lambda_ce: float = 1.0,
    gate_quantile: float = 0.20,
) -> tuple:
    """Switch distillation: PARTITION CE and RKL by teacher entropy
    rather than stacking them.

        H_t  = H(p_T^{T})_t
        tau  = quantile(H_t, gate_quantile)
        m_t  = 1[H_t <= tau]
        L_t  = (1 - m_t) * lambda_ce * CE(y_t) + m_t * T^2 * KL(p_S || p_T)_t

    Compared to a stacked formulation (`lambda_ce*CE + m_t*RKL`, i.e. CE on every
    token plus a gated RKL bonus), this drops the
    CE term on the *fired* (low-teacher-entropy) tokens. Test: does CE on
    math/structural tokens help, or just dilute the RKL signal there? An
    earlier ablation supported the dilution hypothesis at the global level;
    the per-token switch isolates it.

    Loss magnitudes follow the stacked variant's averaging convention:
      RKL averaged over fired tokens (so per-token weight is comparable
      across q-values), CE averaged over UNFIRED tokens (so the per-token
      CE weight is comparable to the stacked variant's unfired regime).
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    mask = (labels != -100).float()
    n_valid = mask.sum().clamp(min=1)

    ce_per_token = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, seq_len)

    rkl_per_token = torch.zeros(
        batch_size,
        seq_len,
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    teacher_ent_per_token = torch.zeros(
        batch_size,
        seq_len,
        device=student_logits.device,
        dtype=torch.float32,
    )

    teacher_entropy_sum = 0.0
    student_entropy_sum = 0.0
    n_entropy_samples = 0

    def _rkl_chunk_fn(student_chunk, teacher_log_soft_detached):
        student_log_soft = F.log_softmax(student_chunk / temperature, dim=-1)
        return F.kl_div(
            teacher_log_soft_detached,
            student_log_soft,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        student_chunk = student_logits[:, chunk_start:chunk_end, :]
        teacher_chunk = teacher_logits[:, chunk_start:chunk_end, :]
        mask_chunk = mask[:, chunk_start:chunk_end]

        with torch.no_grad():
            teacher_log_soft = F.log_softmax(teacher_chunk / temperature, dim=-1)
            teacher_ent_chunk = -(teacher_log_soft.exp() * teacher_log_soft).sum(dim=-1)
            teacher_ent_per_token[:, chunk_start:chunk_end] = teacher_ent_chunk.float()

        rkl_chunk = torch.utils.checkpoint.checkpoint(
            _rkl_chunk_fn,
            student_chunk,
            teacher_log_soft,
            use_reentrant=False,
        )
        rkl_per_token[:, chunk_start:chunk_end] = rkl_chunk

        if chunk_start == 0:
            with torch.no_grad():
                chunk_mask = mask_chunk.bool()
                if chunk_mask.any():
                    student_log_soft_nograd = F.log_softmax(
                        student_chunk / temperature, dim=-1
                    )
                    student_ent = -(
                        student_log_soft_nograd * student_log_soft_nograd.exp()
                    ).sum(dim=-1)
                    teacher_entropy_sum = teacher_ent_chunk[chunk_mask].sum().item()
                    student_entropy_sum = student_ent[chunk_mask].sum().item()
                    n_entropy_samples = int(chunk_mask.sum().item())
                    del student_log_soft_nograd, student_ent
            del teacher_ent_chunk

        del teacher_log_soft, rkl_chunk

    valid_mask = mask.bool()
    with torch.no_grad():
        valid_ent = teacher_ent_per_token[valid_mask]
        if valid_ent.numel() > 0:
            tau = torch.quantile(valid_ent.float(), gate_quantile).item()
        else:
            tau = 0.0
        gate = (teacher_ent_per_token <= tau).to(rkl_per_token.dtype) * mask
        ce_gate = (1.0 - gate) * mask  # CE fires on unfired tokens only
        n_fired = gate.sum().clamp(min=1)
        n_unfired = ce_gate.sum().clamp(min=1)
        fired_frac = (gate.sum() / n_valid).item()

    gated_rkl_per_token = rkl_per_token * gate
    rkl_loss_mean = gated_rkl_per_token.sum() / n_fired
    rkl_loss_scaled = rkl_loss_mean * (temperature**2)

    # CE averaged over UNFIRED tokens only (per-token weight comparable to
    # the stacked variant's unfired regime).
    ce_loss_mean = (ce_per_token * ce_gate).sum() / n_unfired

    total_loss = lambda_ce * ce_loss_mean + rkl_loss_scaled

    with torch.no_grad():
        ent_norm = max(n_entropy_samples, 1)
        valid_rkl_all = rkl_per_token[valid_mask]
        valid_rkl_fired = rkl_per_token[gate.bool()]
        valid_ce = ce_per_token[valid_mask]
        ce_unfired = ce_per_token[ce_gate.bool()] if ce_gate.sum() > 0 else valid_ce[:0]
        fired_ent = (
            teacher_ent_per_token[gate.bool()] if gate.sum() > 0 else valid_ent[:0]
        )
        unfired_mask = (mask.bool()) & (~gate.bool())
        unfired_ent = (
            teacher_ent_per_token[unfired_mask]
            if unfired_mask.sum() > 0
            else valid_ent[:0]
        )

        stats = {
            "switch_distill/rkl_loss": rkl_loss_scaled.item(),
            "switch_distill/ce_term": (lambda_ce * ce_loss_mean).item(),
            "switch_distill/total_loss": total_loss.item(),
            "switch_distill/rkl_per_token_mean_all": (
                valid_rkl_all.mean().item() if valid_rkl_all.numel() > 0 else 0.0
            ),
            "switch_distill/rkl_per_token_mean_fired": (
                valid_rkl_fired.mean().item() if valid_rkl_fired.numel() > 0 else 0.0
            ),
            "switch_distill/ce_per_token_mean_all": (
                valid_ce.mean().item() if valid_ce.numel() > 0 else 0.0
            ),
            "switch_distill/ce_per_token_mean_unfired": (
                ce_unfired.mean().item() if ce_unfired.numel() > 0 else 0.0
            ),
            "switch_distill/teacher_entropy_mean": teacher_entropy_sum / ent_norm,
            "switch_distill/student_entropy_mean": student_entropy_sum / ent_norm,
            "switch_distill/teacher_entropy_full_mean": (
                valid_ent.mean().item() if valid_ent.numel() > 0 else 0.0
            ),
            "switch_distill/teacher_entropy_fired_mean": (
                fired_ent.mean().item() if fired_ent.numel() > 0 else 0.0
            ),
            "switch_distill/teacher_entropy_unfired_mean": (
                unfired_ent.mean().item() if unfired_ent.numel() > 0 else 0.0
            ),
            "switch_distill/tau": float(tau),
            "switch_distill/fired_frac": fired_frac,
            "switch_distill/gate_quantile": float(gate_quantile),
            "switch_distill/lambda_ce": float(lambda_ce),
            "switch_distill/temperature": float(temperature),
        }

    return total_loss, stats


@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1
    # Periodic gradient-geometry probe across source domains (e.g., dclm/math/flan).
    # Disabled when None.
    grad_probe_freq: Optional[int] = None
    # Minimum sequences per domain in a batch to include that domain in the probe.
    grad_probe_min_seqs: int = 2
    # Probe uses only a subset of params for efficiency; cap total elements here.
    grad_probe_max_param_elems: int = 5_000_000
    # Max number of parameter tensors included in the probe subset.
    grad_probe_max_tensors: int = 16
    # Domain name fragments used to bucket source labels.
    grad_probe_domains: List[str] = field(
        default_factory=lambda: ["dclm", "math", "flan"]
    )

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    # On-the-fly teacher model for token-level delta weighting.
    # When set, loads a frozen HF model and computes teacher logprobs each step
    # instead of reading pre-cached logprobs from data.
    teacher_model_path: Optional[str] = None

    # Multi-expert online inference: list of HF model paths.
    # When set, runs a no-grad forward pass through each model per batch,
    # takes the per-token min NLL across all experts, and uses that as the
    # reference signal (Δ = student_nll - min_expert_nll).
    # Takes precedence over teacher_model_path if both are set.
    teacher_model_paths: Optional[List[str]] = None

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # Online excess-loss reweighting controller (set enabled=True to activate).

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(
            f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}"
        )
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def _parse_source_weight_schedule(
    raw_schedule: Optional[Dict[Any, Dict[str, float]]],
) -> Dict[int, Dict[str, float]]:
    if not raw_schedule:
        return {}

    parsed: Dict[int, Dict[str, float]] = {}
    for step_key, overrides in raw_schedule.items():
        step = int(step_key)
        if step < 0:
            raise ValueError(f"source_weight_schedule step must be >= 0, got {step}")
        if not isinstance(overrides, dict):
            raise ValueError(
                f"source_weight_schedule[{step_key}] must be a mapping of source->weight"
            )
        parsed[step] = {str(k): float(v) for k, v in overrides.items()}
    return dict(sorted(parsed.items(), key=lambda kv: kv[0]))


def _get_loader_sources_dict(data_loader_state: Dict[str, Any]) -> Dict[str, float]:
    # PrefetchState -> PackTokensState -> TokenizerState -> MultiChoiceState
    return data_loader_state["it_state"]["it_state"]["it_state"]["sources"]


def _apply_source_weight_overrides(
    data_loader_state: Dict[str, Any],
    overrides: Dict[str, float],
    step: int,
) -> Dict[str, float]:
    sources = _get_loader_sources_dict(data_loader_state)
    updated = dict(sources)
    updated.update(overrides)

    if any(v < 0 for v in updated.values()):
        raise ValueError(
            f"source_weight_schedule produced negative weights at step {step}: {updated}"
        )
    if sum(updated.values()) <= 0:
        raise ValueError(
            f"source_weight_schedule produced zero total weight at step {step}: {updated}"
        )

    # Mutate in-place so the running choose_source iterator sees new values.
    sources.clear()
    sources.update(updated)
    return updated


def _load_source_weights_file(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Source weights file must be a JSON object: {path}")
    weights = {str(k): float(v) for k, v in payload.items()}
    if any(v < 0 for v in weights.values()):
        raise ValueError(f"Negative weights are not allowed in {path}: {weights}")
    if sum(weights.values()) <= 0:
        raise ValueError(f"Sum of weights must be > 0 in {path}: {weights}")
    return weights


def _select_grad_probe_params(
    model: torch.nn.Module,
    max_elems: int,
    max_tensors: int,
) -> List[tuple[str, torch.nn.Parameter]]:
    """Select a representative parameter subset for inexpensive gradient probes."""
    candidates: List[tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip tiny scalars/vectors that tend to be noisy and uninformative.
        if p.numel() < 1024:
            continue
        candidates.append((name, p))

    if not candidates:
        return []

    # Prefer one middle transformer layer when we can infer layer indices.
    # This keeps probe scope stable and cheap versus sampling across the model.
    layer_candidates: List[tuple[int, str, torch.nn.Parameter]] = []
    layer_pat = re.compile(r"\.(?:layers|blocks|h)\.(\d+)\.")
    for name, p in candidates:
        m = layer_pat.search(name)
        if m is None:
            continue
        layer_candidates.append((int(m.group(1)), name, p))

    if layer_candidates:
        layer_ids = sorted({lid for lid, _, _ in layer_candidates})
        middle_layer = layer_ids[len(layer_ids) // 2]
        middle_params = [
            (name, p) for lid, name, p in layer_candidates if lid == middle_layer
        ]
        # Start with larger tensors from the middle layer for stronger signal.
        middle_params.sort(key=lambda x: x[1].numel(), reverse=True)
        picked_middle: List[tuple[str, torch.nn.Parameter]] = []
        total_middle = 0
        for name, p in middle_params:
            if len(picked_middle) >= max_tensors:
                break
            if total_middle + p.numel() > max_elems and picked_middle:
                break
            picked_middle.append((name, p))
            total_middle += p.numel()
        if picked_middle:
            return picked_middle

    # Prefer broad model coverage by taking evenly-spaced tensors.
    step = max(1, len(candidates) // max(1, max_tensors))
    picked: List[tuple[str, torch.nn.Parameter]] = []
    total = 0
    for i in range(0, len(candidates), step):
        name, p = candidates[i]
        if len(picked) >= max_tensors:
            break
        if total + p.numel() > max_elems and picked:
            break
        picked.append((name, p))
        total += p.numel()

    # Fallback to first candidate if budget is extremely small.
    if not picked:
        picked.append(candidates[0])
    return picked


def _build_domain_index_map(
    source_labels: Optional[List[str]],
    batch_size: int,
    domain_keys: List[str],
) -> Dict[str, List[int]]:
    """Map domain key -> sequence indices using substring matching on source labels."""
    domain_to_indices: Dict[str, List[int]] = {k: [] for k in domain_keys}
    if not source_labels or len(source_labels) != batch_size:
        return domain_to_indices

    for i, raw in enumerate(source_labels):
        label = str(raw).lower()
        for key in domain_keys:
            if key in label:
                domain_to_indices[key].append(i)
                break
    return domain_to_indices


def _grad_probe_stats_for_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    source_labels: Optional[List[str]],
    probe_named_params: List[tuple[str, torch.nn.Parameter]],
    domain_keys: List[str],
    min_seqs: int,
) -> Dict[str, float]:
    """
    Compute pairwise gradient cosine similarities across requested domains.
    Uses CE loss on current batch subsets and a parameter subset for efficiency.
    """
    if not probe_named_params:
        return {"grad_probe/available": 0.0, "grad_probe/reason_no_params": 1.0}

    bsz = int(labels.shape[0])
    domain_to_indices = _build_domain_index_map(source_labels, bsz, domain_keys)
    stats: Dict[str, float] = {
        "grad_probe/available": 1.0,
        "grad_probe/domains_requested": float(len(domain_keys)),
        "grad_probe/domains_usable": float(
            sum(1 for k in domain_keys if len(domain_to_indices.get(k, [])) >= min_seqs)
        ),
    }
    for k in domain_keys:
        stats[f"grad_probe/nseq/{k}"] = float(len(domain_to_indices.get(k, [])))

    params = [p for _, p in probe_named_params]
    grads_by_domain: Dict[str, List[Optional[torch.Tensor]]] = {}
    real_domain_grad: Dict[str, bool] = {}
    fallback_idx = list(range(min(max(1, min_seqs), max(1, bsz))))
    if not fallback_idx:
        fallback_idx = [0]

    # IMPORTANT: run the same number of forward/backward-style calls on every rank
    # to keep distributed collectives aligned.
    for dom in domain_keys:
        idxs = domain_to_indices.get(dom, [])
        has_real = len(idxs) >= min_seqs
        real_domain_grad[dom] = has_real
        use_indices = idxs if has_real else fallback_idx
        stats[f"grad_probe/has_real_batch/{dom}"] = 1.0 if has_real else 0.0
        idx = torch.tensor(idxs, device=input_ids.device, dtype=torch.long)
        if not has_real:
            idx = torch.tensor(use_indices, device=input_ids.device, dtype=torch.long)
        dom_loss = model(input_ids[idx], labels[idx])
        grads = torch.autograd.grad(
            dom_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        cached: List[Optional[torch.Tensor]] = []
        sq = 0.0
        for g in grads:
            if g is None:
                cached.append(None)
                continue
            gc = g.detach().float().cpu()
            cached.append(gc)
            sq += float((gc * gc).sum().item())
        grads_by_domain[dom] = cached
        stats[f"grad_probe/grad_norm/{dom}"] = sq**0.5

    # Pairwise cosine on the probe subset.
    doms = list(domain_keys)
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = doms[i], doms[j]
            dot = 0.0
            n1 = 0.0
            n2 = 0.0
            for ga, gb in zip(grads_by_domain[a], grads_by_domain[b]):
                if ga is None or gb is None:
                    continue
                dot += float((ga * gb).sum().item())
                n1 += float((ga * ga).sum().item())
                n2 += float((gb * gb).sum().item())
            denom = max((n1 * n2) ** 0.5, 1e-12)
            stats[f"grad_probe/cosine/{a}_vs_{b}"] = dot / denom
            stats[f"grad_probe/cosine_valid/{a}_vs_{b}"] = (
                1.0 if (real_domain_grad[a] and real_domain_grad[b]) else 0.0
            )

    return stats


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = (
                dp_rank * world_mesh["dp_shard"].size()
                + world_mesh["dp_shard"].get_local_rank()
            )
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = LMTransformer(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            if "olmo" in args.checkpoint.init_ckpt_path.lower():
                assert (
                    args.model.qk_norm
                ), f"OLMo checkpoint requires qk_norm=true, got {args.model.qk_norm}"
                assert (
                    args.model.post_norm
                ), f"OLMo checkpoint requires post_norm=true, got {args.model.post_norm}"
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(
                args.checkpoint.init_ckpt_path, model, model_key=""
            )  # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters()  # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # Load frozen teacher model(s) for on-the-fly delta weighting.
        # teacher_model_paths (list) takes precedence over teacher_model_path (single).
        def _load_teacher_hf(path: str):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                logger.info(f"Teacher load: using flash_attention_2 for {path}")
            except Exception as e:
                logger.warning(
                    f"Teacher load: flash_attention_2 unavailable for {path} ({type(e).__name__}: {e}); "
                    "falling back to default attention."
                )
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                )
            model.config.use_cache = False
            model = model.cuda().eval()

            # Opt-in: replace the layer-stack Linear modules with rowwise-scaled
            # FP8 (e4m3fn) on H200. Teacher forwards run under torch.inference_mode
            # so no backward path is exercised; rowwise FP8 inference quality
            # difference vs bf16 is well below KD signal floor (typical < 0.1pp on
            # task avg for Llama-style models). Skips lm_head and embed_tokens by
            # filter — only the per-layer attention/FFN projections are converted.
            # Driven by env var so in-flight jobs reading this code path on requeue
            # see identical behavior unless LINGUA_TEACHER_FP8=1 is set.
            if os.environ.get("LINGUA_TEACHER_FP8", "0") == "1":
                try:
                    from lingua.float8 import convert_linears_to_fp8

                    fp8_filter = os.environ.get(
                        "LINGUA_TEACHER_FP8_FILTER", r"layers\.[0-9]+\."
                    )
                    n_lin_before = sum(
                        1 for m in model.modules() if isinstance(m, torch.nn.Linear)
                    )
                    model = convert_linears_to_fp8(model, "rowwise", fp8_filter)
                    n_fp8 = sum(
                        1
                        for m in model.modules()
                        if m.__class__.__name__ == "Fp8Linear"
                    )
                    logger.info(
                        f"Teacher load: FP8 (rowwise) applied to {path}: "
                        f"{n_fp8}/{n_lin_before} Linear modules converted "
                        f"(filter={fp8_filter!r})."
                    )
                except Exception as e:
                    logger.warning(
                        f"Teacher load: FP8 conversion failed for {path} "
                        f"({type(e).__name__}: {e}); falling back to bf16 teacher."
                    )

            # Opt-in: wrap teacher in torch.compile for ~15-30% faster fwd. Off
            # by default so in-flight jobs (which read this code path on requeue)
            # see identical behavior; flip on via env var in the launch script.
            # NOTE: when stacked with FP8, compile happens AFTER FP8 conversion
            # so Inductor sees the Fp8Linear forward path and can fuse around it.
            if os.environ.get("LINGUA_COMPILE_TEACHER", "0") == "1":
                try:
                    model = torch.compile(model, dynamic=False, fullgraph=False)
                    logger.info(f"Teacher load: torch.compile wrapped {path}")
                except Exception as e:
                    logger.warning(
                        f"Teacher load: torch.compile failed for {path} "
                        f"({type(e).__name__}: {e}); using eager teacher."
                    )
            return model

        teacher_model = None
        if args.teacher_model_path:
            logger.info(f"Loading frozen teacher model: {args.teacher_model_path}")
            teacher_model = _load_teacher_hf(args.teacher_model_path)
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            teacher_param_count = sum(p.numel() for p in teacher_model.parameters())
            logger.info(
                f"Teacher model loaded: {teacher_param_count:,} parameters (frozen)"
            )
            logger.info(f"GPU memory after teacher load: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader_state = init_dataloader_state_from_args(
            args.data, dp_rank, dp_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        source_weight_schedule = _parse_source_weight_schedule(
            getattr(args.data, "source_weight_schedule", None)
        )
        applied_source_weight_steps: set[int] = set()
        first_micro_acc_step = 1 % args.grad_acc_steps
        source_weight_refresh_every = int(
            getattr(args.checkpoint.dump, "every", 0) or 0
        )
        source_weights_file = os.path.join(args.dump_dir, "source_weights.json")
        source_weights_file_mtime: Optional[float] = None

        if source_weight_schedule:
            # Catch up to the latest milestone on resume so a restarted job adopts
            # the intended active domain mix immediately.
            resume_target_step = max(
                (s for s in source_weight_schedule if s <= train_state.step),
                default=None,
            )
            if resume_target_step is not None:
                updated = _apply_source_weight_overrides(
                    train_state.data_loader_state,
                    source_weight_schedule[resume_target_step],
                    resume_target_step,
                )
                applied_source_weight_steps.add(resume_target_step)
                if get_is_master():
                    norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                    logger.info(
                        f"Applied source_weight_schedule catch-up at step {resume_target_step}: "
                        f"raw={updated} norm={norm}"
                    )
        if os.path.isfile(source_weights_file):
            try:
                file_weights = _load_source_weights_file(source_weights_file)
                updated = _apply_source_weight_overrides(
                    train_state.data_loader_state,
                    file_weights,
                    train_state.step,
                )
                source_weights_file_mtime = os.path.getmtime(source_weights_file)
                if get_is_master():
                    norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                    logger.info(
                        f"Applied source weights from file at startup ({source_weights_file}): "
                        f"raw={updated} norm={norm}"
                    )
            except Exception as e:
                if get_is_master():
                    logger.warning(
                        f"Failed to apply startup source weights file {source_weights_file}: {e}"
                    )
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        grad_probe_named_params: List[tuple[str, torch.nn.Parameter]] = []
        grad_probe_warned_no_labels = False
        pending_grad_probe_stats: Dict[str, float] = {}
        if args.grad_probe_freq is not None:
            grad_probe_named_params = _select_grad_probe_params(
                model,
                max_elems=max(1, int(args.grad_probe_max_param_elems)),
                max_tensors=max(1, int(args.grad_probe_max_tensors)),
            )
            if get_is_master():
                n_elems = sum(p.numel() for _, p in grad_probe_named_params)
                logger.info(
                    f"[GradProbe] enabled freq={args.grad_probe_freq}, "
                    f"params={len(grad_probe_named_params)} tensors, elems={n_elems}"
                )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )

        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
            )
        )
        # ─────────────────────────────────────────────────────────────────────
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        saved = False
        routing_buffer_cache = {
            "padded": None,
            "labels": None,
            "valid_mask": None,
            "capacity_docs": 0,
            "capacity_len": 0,
        }

        # def ids_for(txt: str):
        #     # Encode without BOS/EOS so we get exactly the token pieces of the string.
        #     return tokenizer.encode(txt, add_bos=False, add_eos=False)

        # IGNORE_IDS = set()
        # if args.data.add_special_tokens:
        #     for s in (SpecialTokens.FACTUAL_TOKEN, SpecialTokens.NONFACTUAL_TOKEN, SpecialTokens.PARTIAL_FACTUAL_TOKEN):
        #         IGNORE_IDS.update(ids_for(s.value))
        #         assert len(ids_for(s.value)) == 1, f"{s.value} splits into {ids_for(s.value)}; register as a single special token!"
        # IGNORE_IDS_T = torch.tensor(list(IGNORE_IDS), dtype=torch.long)

        # attn_impl = "flex_attention" if args.data.mask_cross_doc_loss else "sdpa"
        # logger.info(f"Attn implementation: {attn_impl}")
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps

            # Optional runtime domain/source reweighting (no-op when schedule is empty).
            if (
                source_weight_schedule
                and train_state.acc_step == first_micro_acc_step
                and train_state.step in source_weight_schedule
                and train_state.step not in applied_source_weight_steps
            ):
                updated = _apply_source_weight_overrides(
                    train_state.data_loader_state,
                    source_weight_schedule[train_state.step],
                    train_state.step,
                )
                applied_source_weight_steps.add(train_state.step)
                if get_is_master():
                    norm = {k: v / sum(updated.values()) for k, v in updated.items()}
                    logger.info(
                        f"Applied source_weight_schedule at step {train_state.step}: "
                        f"raw={updated} norm={norm}"
                    )
            # Config-free periodic source reweighting:
            # If dump_dir/source_weights.json exists, refresh it at fixed intervals.
            # This allows updating domain weights mid-run without editing YAML.
            if (
                source_weight_refresh_every > 0
                and train_state.acc_step == first_micro_acc_step
                and train_state.step > 0
                and (train_state.step % source_weight_refresh_every == 0)
                and os.path.isfile(source_weights_file)
            ):
                try:
                    curr_mtime = os.path.getmtime(source_weights_file)
                    if (
                        source_weights_file_mtime is None
                        or curr_mtime != source_weights_file_mtime
                    ):
                        file_weights = _load_source_weights_file(source_weights_file)
                        updated = _apply_source_weight_overrides(
                            train_state.data_loader_state,
                            file_weights,
                            train_state.step,
                        )
                        source_weights_file_mtime = curr_mtime
                        if get_is_master():
                            norm = {
                                k: v / sum(updated.values()) for k, v in updated.items()
                            }
                            logger.info(
                                f"Applied source weights from file at step {train_state.step} "
                                f"({source_weights_file}): raw={updated} norm={norm}"
                            )
                except Exception as e:
                    if get_is_master():
                        logger.warning(
                            f"Failed to refresh source weights from {source_weights_file} "
                            f"at step {train_state.step}: {e}"
                        )

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch, train_state.data_loader_state = next(data_loader)
            # The source-labeled loader was removed with the online-reweighting
            # path; the standard loader does not carry per-document sources.
            _source_labels = None

            # Handle new dict format with tokens and cu_seqlens
            if isinstance(batch, dict):
                batch_tokens = batch["tokens"]
                batch_cu_seqlens = batch.get(
                    "cu_seqlens", None
                )  # List of cu_seqlens per batch item
            else:
                batch_tokens = batch
                batch_cu_seqlens = None

            # Avoid an unconditional copy when the loader already returns a tensor.
            batch_tokens = torch.as_tensor(batch_tokens, dtype=torch.long)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            # Extract input_ids and labels (views 0 and 1)
            input_ids = batch_tokens[:, :, 0].to(device="cuda", non_blocking=True)
            labels = batch_tokens[:, :, 1].to(device="cuda", non_blocking=True)

            # Prepare cu_seqlens for FA2 varlen if cross-doc attention masking is enabled
            cu_seqlens_tensor = None
            max_seqlen = None
            if (
                getattr(args.data, "disable_cross_doc_attn", False)
                and batch_cu_seqlens is not None
            ):
                # Convert per-batch cu_seqlens to a single flattened tensor for FA2 varlen
                # FA2 varlen expects cu_seqlens to mark document boundaries across the flattened batch
                # For batch_size B and seq_len S, we have B*S total tokens
                # Each item in batch_cu_seqlens is a list like [0, doc1_end, doc2_end, ..., S]

                bsz = batch_tokens.shape[0]
                seq_len = batch_tokens.shape[1]

                # Build global cu_seqlens: offset each batch item's boundaries by batch_idx * seq_len
                global_cu_seqlens = [0]
                max_doc_len = 0
                for batch_idx, cu_seqs in enumerate(batch_cu_seqlens):
                    offset = batch_idx * seq_len
                    # Add all boundaries except the first (0) since we already have 0 or previous end
                    for i, pos in enumerate(cu_seqs):
                        if i == 0:
                            continue  # Skip the leading 0
                        global_pos = offset + pos
                        global_cu_seqlens.append(global_pos)

                        # Track max document length
                        prev_pos = cu_seqs[i - 1]
                        doc_len = pos - prev_pos
                        max_doc_len = max(max_doc_len, doc_len)

                cu_seqlens_tensor = torch.tensor(
                    global_cu_seqlens, dtype=torch.int32, device="cuda"
                )
                max_seqlen = max_doc_len

            # All teacher signals come from the on-the-fly teacher model; the
            # precomputed-signal (JSONL field) path was removed with the extra
            # dataloader views, so there is never a cached teacher NLL.
            teacher_logprobs = None

            if teacher_model is not None:
                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_lp = F.log_softmax(teacher_out.logits, dim=-1)

                    if teacher_logprobs is None:
                        teacher_logprobs = -teacher_lp.gather(
                            dim=-1, index=labels.clamp(min=0).unsqueeze(-1)
                        ).squeeze(-1)
                        teacher_logprobs[labels == -100] = 0.0

                    del teacher_out, teacher_lp  # Free full vocab tensors

            # Log dataloader output at start of training to verify masking
            if train_state.step == 0 and train_state.acc_step == 1 and get_is_master():
                total_labels = labels.numel()
                masked_labels = (labels == -100).sum().item()
                valid_labels = total_labels - masked_labels
                mask_pct = (
                    100.0 * masked_labels / total_labels if total_labels > 0 else 0
                )

                logger.info("=" * 60)
                logger.info("DATALOADER DEBUG - First batch sample")
                logger.info(f"  Batch shape: {labels.shape} (batch_size, seq_len)")
                logger.info(f"  Total labels: {total_labels}")
                logger.info(
                    f"  Masked labels (-100): {masked_labels} ({mask_pct:.1f}%)"
                )
                logger.info(
                    f"  Valid labels (loss computed): {valid_labels} ({100-mask_pct:.1f}%)"
                )
                logger.info("=" * 60)

            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            bsz, seqlen = labels.shape
            grad_probe_stats: Dict[str, float] = {}

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                    )

                    # if doc_ids_t is not None:
                    #     probe_doc = doc_ids_t[:probe_bsz, :probe_seq]
                    #     probe_attn_impl = "flex_attention"
                    # else:
                    #     probe_doc = None
                    #     probe_attn_impl = "sdpa"

                    # probe_loss = model(
                    #     input_ids[:probe_bsz, :probe_seq],
                    #     labels[:probe_bsz, :probe_seq],
                    #     attn_impl=probe_attn_impl,
                    #     doc_ids=probe_doc,
                    #     mask_cross_doc_loss=args.data.mask_cross_doc_loss,
                    # )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            if (
                args.grad_probe_freq is not None
                and train_state.acc_step == first_micro_acc_step
                and every_n_steps(
                    train_state, args.grad_probe_freq, acc_step=first_micro_acc_step
                )
            ):
                if (
                    _source_labels is None
                    and not grad_probe_warned_no_labels
                    and get_is_master()
                ):
                    logger.warning(
                        "[GradProbe] source labels are unavailable for this dataloader path; "
                        "domain cosine metrics require source-labeled batches."
                    )
                    grad_probe_warned_no_labels = True
                try:
                    assert (
                        next(model.parameters()).grad is None
                    ), "Grad probe requires clean grads (run at optimizer-step boundary)."
                    grad_probe_stats = _grad_probe_stats_for_batch(
                        model=model,
                        input_ids=input_ids,
                        labels=labels,
                        source_labels=_source_labels,
                        probe_named_params=grad_probe_named_params,
                        domain_keys=[str(x).lower() for x in args.grad_probe_domains],
                        min_seqs=max(1, int(args.grad_probe_min_seqs)),
                    )
                    optimizer.zero_grad()
                except Exception as e:
                    grad_probe_stats = {"grad_probe/error": 1.0}
                    if get_is_master():
                        logger.warning(
                            f"[GradProbe] failed at step {train_state.step}: {e}"
                        )
                pending_grad_probe_stats = grad_probe_stats

            # loss = model(input_ids, labels, doc_ids=doc_ids_t, mask_cross_doc_loss=args.data.mask_cross_doc_loss, attn_impl=attn_impl)

            # Loss-form selection. Exactly one of these may be set in the
            # recipe YAML; if none is, training falls through to vanilla NTP
            # (the bare `else` at the end of the dispatch below).
            use_kl_distillation = (
                getattr(args.data, "use_kl_distillation", False)
                and teacher_model is not None
            )
            use_reverse_kl_distillation = (
                getattr(args.data, "use_reverse_kl_distillation", False)
                and teacher_model is not None
            )
            use_switch_distill = (
                getattr(args.data, "use_switch_distill", False)
                and teacher_model is not None
            )
            batch_delta_stats = None
            if use_kl_distillation:
                # Classic Knowledge Distillation: KL divergence between student and teacher
                logits = model(input_ids, target=None)

                # Recompute teacher logits for KL distillation (need full logits, not just NLL)
                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    # Handle vocab size mismatch between student and teacher
                    # Student vocab may be padded to multiple_of (e.g., 100278 → 100352)
                    # Teacher from HuggingFace has actual vocab size (100278)
                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]

                        if V_teacher < V_student:
                            # Pad teacher logits with very negative values (will have ~0 probability after softmax)
                            padding_size = V_student - V_teacher
                            padding = torch.full(
                                (*teacher_logits_for_kd.shape[:-1], padding_size),
                                -1e10,  # Very negative logits → 0 probability
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if (
                                train_state.step == 0
                                and train_state.acc_step == 1
                                and get_is_master()
                            ):
                                logger.info(
                                    f"Padded teacher logits from {V_teacher} to {V_student} (student uses multiple_of padding)"
                                )
                        else:
                            # Truncate teacher logits if teacher is larger (shouldn't happen, but handle it)
                            teacher_logits_for_kd = teacher_logits_for_kd[
                                ..., :V_student
                            ]
                            if (
                                train_state.step == 0
                                and train_state.acc_step == 1
                                and get_is_master()
                            ):
                                logger.info(
                                    f"Truncated teacher logits from {V_teacher} to {V_student}"
                                )

                loss, batch_delta_stats = compute_kl_distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=getattr(args.data, "kl_temperature", 2.0),
                    alpha=getattr(args.data, "kl_alpha", 0.5),
                )

                del teacher_logits_for_kd  # Free memory

                if (
                    train_state.step == 0
                    and train_state.acc_step == 1
                    and get_is_master()
                ):
                    logger.info("=" * 60)
                    logger.info(
                        "CLASSIC KNOWLEDGE DISTILLATION - First batch statistics"
                    )
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(
                        f"  Temperature: {getattr(args.data, 'kl_temperature', 2.0)}"
                    )
                    logger.info(
                        f"  Alpha (KL weight): {getattr(args.data, 'kl_alpha', 0.5)}"
                    )
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_reverse_kl_distillation:
                # Reverse-KL distillation (MiniLLM): minimize KL(p_S || p_T) instead
                # of KL(p_T || p_S). Mode-seeking -- the principled capacity-gap
                # remedy when teacher >> student, because it lets the student
                # concentrate on the dominant teacher mode and ignore the long tail
                # of teacher mass it cannot represent.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (
                                    *teacher_logits_for_kd.shape[:-1],
                                    V_student - V_teacher,
                                ),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if (
                                train_state.step == 0
                                and train_state.acc_step == 1
                                and get_is_master()
                            ):
                                logger.info(
                                    f"[reverse-kl] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[
                                ..., :V_student
                            ]
                            if (
                                train_state.step == 0
                                and train_state.acc_step == 1
                                and get_is_master()
                            ):
                                logger.info(
                                    f"[reverse-kl] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_reverse_kl_distillation_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(
                        getattr(args.data, "reverse_kl_temperature", 2.0)
                    ),
                    alpha=float(getattr(args.data, "reverse_kl_alpha", 0.5)),
                    chunk_size=int(getattr(args.data, "reverse_kl_chunk_size", 128)),
                )

                del teacher_logits_for_kd

                if (
                    train_state.step == 0
                    and train_state.acc_step == 1
                    and get_is_master()
                ):
                    logger.info("=" * 60)
                    logger.info(
                        "REVERSE-KL DISTILLATION (MiniLLM-style) - First batch statistics"
                    )
                    logger.info(f"  Teacher model: {args.teacher_model_path}")
                    logger.info(
                        f"  Loss:         L = (1-alpha)*CE + alpha*T^2*KL(p_S || p_T)"
                    )
                    logger.info(
                        f"  temperature:  {getattr(args.data, 'reverse_kl_temperature', 2.0)}"
                    )
                    logger.info(
                        f"  alpha:        {getattr(args.data, 'reverse_kl_alpha', 0.5)}"
                    )
                    for k, v in batch_delta_stats.items():
                        logger.info(f"  {k}: {v:.4f}")
                    logger.info("=" * 60)
            elif use_switch_distill:
                # Switch distillation. Partitions CE and RKL
                # by the same teacher-entropy gate (no overlap):
                #     m_t = 1[H(p_T)_t <= tau]
                #     L = (1 - m_t) * lambda_ce * CE + m_t * T^2 * KL(p_S||p_T)
                # Tests whether the gold-token CE on math-flavored tokens is
                # helping or just diluting the RKL signal there. The per-token
                # switch isolates the effect.
                logits = model(input_ids, target=None)

                with torch.inference_mode():
                    teacher_out = teacher_model(input_ids, use_cache=False)
                    teacher_logits_for_kd = teacher_out.logits
                    del teacher_out

                    if teacher_logits_for_kd.shape[-1] != logits.shape[-1]:
                        V_teacher = teacher_logits_for_kd.shape[-1]
                        V_student = logits.shape[-1]
                        if V_teacher < V_student:
                            padding = torch.full(
                                (
                                    *teacher_logits_for_kd.shape[:-1],
                                    V_student - V_teacher,
                                ),
                                -1e10,
                                device=teacher_logits_for_kd.device,
                                dtype=teacher_logits_for_kd.dtype,
                            )
                            teacher_logits_for_kd = torch.cat(
                                [teacher_logits_for_kd, padding], dim=-1
                            )
                            if (
                                train_state.step == 0
                                and train_state.acc_step == 1
                                and get_is_master()
                            ):
                                logger.info(
                                    f"[switch-distill] Padded teacher logits {V_teacher} -> {V_student}"
                                )
                        else:
                            teacher_logits_for_kd = teacher_logits_for_kd[
                                ..., :V_student
                            ]
                            if (
                                train_state.step == 0
                                and train_state.acc_step == 1
                                and get_is_master()
                            ):
                                logger.info(
                                    f"[switch-distill] Truncated teacher logits {V_teacher} -> {V_student}"
                                )

                loss, batch_delta_stats = compute_switch_distill_loss(
                    student_logits=logits,
                    teacher_logits=teacher_logits_for_kd,
                    labels=labels,
                    temperature=float(
                        getattr(args.data, "switch_distill_temperature", 2.0)
                    ),
                    chunk_size=int(
                        getattr(args.data, "switch_distill_chunk_size", 128)
                    ),
                    lambda_ce=float(
                        getattr(args.data, "switch_distill_lambda_ce", 1.0)
                    ),
                    gate_quantile=float(
                        getattr(args.data, "switch_distill_quantile", 0.20)
                    ),
                )

                del teacher_logits_for_kd

                if (
                    train_state.step == 0
                    and train_state.acc_step == 1
                    and get_is_master()
                ):
                    logger.info("=" * 60)
                    logger.info(
                        "SWITCH DISTILLATION (CE/RKL partition by teacher entropy) - First batch statistics"
                    )
                    logger.info(f"  Teacher model:    {args.teacher_model_path}")
                    logger.info(
                        f"  Loss:             L = (1-m_t)*lambda_ce*CE + m_t*T^2*KL(p_S||p_T)"
                    )
                    logger.info(
                        f"  temperature:      {getattr(args.data, 'switch_distill_temperature', 2.0)}"
                    )
                    logger.info(
                        f"  lambda_ce:        {getattr(args.data, 'switch_distill_lambda_ce', 1.0)}"
                    )
                    logger.info(
                        f"  gate_quantile:    {getattr(args.data, 'switch_distill_quantile', 0.20)}"
                    )
                    for k, v in batch_delta_stats.items():
                        if isinstance(v, (int, float)):
                            logger.info(f"  {k}: {v:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")
                    logger.info("=" * 60)
            else:
                loss = model(input_ids, labels)
            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            skipped_nonfinite_step = False
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm
                ).item()

                if not math.isfinite(grad_norm):
                    skipped_nonfinite_step = True
                    optimizer.zero_grad()
                    scheduler.step()
                    train_state.step += 1
                    if not hasattr(train_state, "consecutive_nonfinite_steps"):
                        train_state.consecutive_nonfinite_steps = 0
                    train_state.consecutive_nonfinite_steps += 1
                    if get_is_master():
                        logger.warning(
                            f"[NaN-guard] step={train_state.step}: "
                            f"non-finite grad_norm ({grad_norm}); skipped "
                            f"optimizer.step(). "
                            f"consecutive_skipped={train_state.consecutive_nonfinite_steps}"
                        )
                    _max_consec = int(
                        os.environ.get("LINGUA_MAX_CONSECUTIVE_NAN_SKIPS", "10")
                    )
                    if train_state.consecutive_nonfinite_steps >= _max_consec:
                        raise RuntimeError(
                            f"[NaN-guard] {train_state.consecutive_nonfinite_steps} "
                            f"consecutive non-finite-grad steps at step "
                            f"{train_state.step}; aborting to avoid silently "
                            f"burning compute. Override threshold with "
                            f"LINGUA_MAX_CONSECUTIVE_NAN_SKIPS=<N>."
                        )
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    train_state.step += 1
                    if hasattr(train_state, "consecutive_nonfinite_steps"):
                        train_state.consecutive_nonfinite_steps = 0

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                if batch_delta_stats is not None:
                    to_sync.update(
                        {
                            k: v
                            for k, v in batch_delta_stats.items()
                            if not k.startswith("_")
                        }
                    )
                if pending_grad_probe_stats:
                    to_sync.update(pending_grad_probe_stats)
                metrics.update(dist_mean_dict(to_sync))
                pending_grad_probe_stats = {}

                if get_is_master():
                    if batch_delta_stats is not None:
                        for k, v in batch_delta_stats.items():
                            if k.startswith("_"):
                                # Debug-only stats may be vectors (e.g. per-seq arrays).
                                # Log scalar tensors directly and reduce non-scalar tensors
                                # to their mean to keep metrics logging robust.
                                if isinstance(v, torch.Tensor):
                                    if v.numel() == 1:
                                        metrics[k[1:]] = v.item()
                                    else:
                                        metrics[f"{k[1:]}/mean"] = (
                                            v.float().mean().item()
                                        )
                                else:
                                    metrics[k[1:]] = v
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                delta_info = ""
                if batch_delta_stats is not None:
                    if "kd/kl_loss" in batch_delta_stats:
                        # Forward-KL distillation
                        delta_info = (
                            f"  KL: {batch_delta_stats['kd/kl_loss']:.3f}"
                            f"  CE: {batch_delta_stats['kd/ce_loss']:.3f}"
                        )
                    elif "switch_distill/total_loss" in batch_delta_stats:
                        # Switch distillation: CE/RKL partition by teacher entropy
                        delta_info = (
                            f"  RKL: {batch_delta_stats['switch_distill/rkl_loss']:.3f}"
                            f"  CE: {batch_delta_stats['switch_distill/ce_term']:.3f}"
                            f"  tau: {batch_delta_stats['switch_distill/tau']:.3f}"
                            f"  fired: {batch_delta_stats['switch_distill/fired_frac']*100:.1f}%"
                        )
                    else:
                        # No specific loss type matched - no delta info to display
                        delta_info = ""
                logger.info(
                    f"step: {train_state.step}"
                    f"  total_tokens: {total_tokens}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"{delta_info}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            is_dump_step = every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            )
            is_eval_step = every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            )
            if is_dump_step or is_eval_step:
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if args.eval is not None and every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                from apps.main.eval_olmes import (
                    launch_olmes_eval as _launch_eval,
                    EVAL_FOLDER_NAME,
                    OlmesEvalArgs as _EvalArgs,
                )

                _script = "apps.main.eval_olmes"

                eval_dict = dict(args.eval)
                eval_dict.pop("harness", None)
                eval_args = dataclass_from_dict(_EvalArgs, eval_dict)

                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    _launch_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                        eval_args.wandb.id = wandb.run.id
                        eval_args.wandb.entity = wandb.run.entity
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching OLMES evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script=_script,
                                copy_code=False,
                                nodes=max(1, args.async_eval_gpus // 8),
                                ngpu=args.async_eval_gpus,
                                time=480,
                                mem="200GB",
                                account=os.environ.get("SLURM_ACCOUNT", ""),
                                qos=os.environ.get("SLURM_QOS", ""),
                                override=False,
                                dirs_exists_ok=True,
                                anaconda=os.environ.get(
                                    "LINGUA_CONDA_ENV_PATH",
                                    os.path.join(
                                        os.environ.get("HOME", ""),
                                        "miniconda3",
                                        "envs",
                                        os.environ.get("LINGUA_CONDA_ENV", "lingua"),
                                    ),
                                ),
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)

    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
