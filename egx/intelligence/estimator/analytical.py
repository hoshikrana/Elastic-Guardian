"""
EGX Analytical Estimator — Layer 3.

Calculates memory usage based on theoretical formulas for LLM training.
Speed: <1ms (pure arithmetic).
Accuracy: ~90-95% (depends on model implementation details like KV cache).

Formulas:
  P = total_params
  P_train = trainable_params
  B = batch_size
  S = seq_len
  H = hidden_dim
  L = num_layers
  
  M_weights    = P * weight_dtype.byte_size()
  M_gradients  = P_train * grad_dtype.byte_size()
  M_optimizer  = P_train * optimizer.bytes_per_param()
  M_activations = B * S * H * L * activation_factor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from egx.core.enums import EstimationMethod
from egx.core.models import MemoryReport
from egx.intelligence.estimator.base import BaseEstimator

if TYPE_CHECKING:
    from egx.core.models import HardwareTopology, ModelProfile, TrainingPlan


class AnalyticalEstimator(BaseEstimator):
    """
    Formula-based estimator for zero-config planning.
    Used for initial strategy pruning and fallbacks.
    """

    # Multipliers for activation memory (varies by strategy)
    ACTIVATION_FACTOR_DEFAULT = 34.0  # approximate bits per token-layer
    
    def estimate(
        self, 
        topology: HardwareTopology, 
        profile: ModelProfile, 
        plan: TrainingPlan
    ) -> MemoryReport:
        
        # 1. Weights
        # Law: call TrainingMode.weight_dtype() to get actual storage dtype
        storage_dtype = plan.mode.weight_dtype(profile.weight_dtype)
        weights_bytes = int(profile.total_params * storage_dtype.byte_size())
        
        # 2. Trainable params (P_lora or P)
        p_train = profile.trainable_params
        if plan.mode.uses_peft() and plan.lora_rank:
            # Approximate LoRA params: 2 * L * H * rank
            # (Very rough, depends on target modules, using heuristic)
            p_train = int(2 * profile.num_layers * profile.hidden_dim * plan.lora_rank)
            
        # 3. Gradients
        # Usually same as trainable weights, or FP32 if mixed precision
        grad_bytes = int(p_train * 4)  # conservatively assume FP32 grads
        
        # 4. Optimizer States
        opt_bytes = int(p_train * plan.optimizer.bytes_per_param())
        
        # 5. Activations
        # A = batch * seq * hidden * layers * factor
        # Mixed precision and gradient checkpointing reduce this
        act_factor = self.ACTIVATION_FACTOR_DEFAULT
        if plan.gradient_checkpointing:
            act_factor *= 0.15  # checkpointing reduces activations significantly
            
        # Bits -> Bytes conversion (/8)
        act_bytes = int((plan.batch_size * plan.seq_len * profile.hidden_dim * 
                        profile.num_layers * act_factor) / 8)
        
        # 6. Overhead (CUDA context, temporary buffers)
        # Context is measured by gpu_probe, we add a safety buffer
        overhead_bytes = int(weights_bytes * 0.05 + 512 * 1024 * 1024) # 5% + 512MB
        
        return MemoryReport(
            weights_bytes     = weights_bytes,
            activations_bytes = act_bytes,
            gradients_bytes   = grad_bytes,
            optimizer_bytes   = opt_bytes,
            overhead_bytes    = overhead_bytes,
            method            = EstimationMethod.ANALYTICAL,
            confidence        = 0.90,
            error_bound_pct   = 10.0
        )
