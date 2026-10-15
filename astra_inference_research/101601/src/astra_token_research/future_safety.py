"""Explicit future-rollout risk definitions; no fabricated training results.

Teacher/student outputs must be from matching starting conditions/actions.
The loss is differentiable if upstream latents are still connected through a
soft-gated rollout. Hard top-k masks are non-differentiable; the trainable future
objective requires a soft/straight-through gate *and* multiple chunk unrolls.
"""
from __future__ import annotations


def rollout_distillation_loss(student_chunks, teacher_chunks,
                              *, weights=None, distant_weight: float=1.0):
    import torch
    if len(student_chunks)!=len(teacher_chunks) or not student_chunks:
        raise ValueError("Matching nonempty future chunk sequences are required")
    if distant_weight<0:
        raise ValueError("distant_weight must be nonnegative")
    n=len(student_chunks)
    if weights is None:
        weights=[1 + distant_weight*i/max(1,n-1) for i in range(n)]
    if len(weights)!=n or any(w<0 for w in weights):
        raise ValueError("Invalid chunk loss weights")
    values=[]
    for student,teacher,weight in zip(student_chunks,teacher_chunks,weights):
        if student.shape!=teacher.shape:
            raise ValueError("Teacher/student latent shapes must match exactly")
        values.append(float(weight)*torch.nn.functional.mse_loss(student,teacher.detach()))
    return torch.stack(values).sum()/sum(weights)


def budget_regularizer(soft_gate_probabilities, desired_ratio:float):
    if not 0<desired_ratio<1:
        raise ValueError("desired_ratio must be in (0,1)")
    return (soft_gate_probabilities.mean()-desired_ratio).square()
