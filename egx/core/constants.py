"""
EGX system-wide constants.

Law: no logic, no classes, no functions — constants only.
All memory constants are int (Law 10).
"""

# ---------------------------------------------------------------------------
# Memory unit constants (int bytes — Law 10)
# ---------------------------------------------------------------------------
KB: int = 1_024
MB: int = 1_048_576
GB: int = 1_073_741_824
TB: int = 1_099_511_627_776

# ---------------------------------------------------------------------------
# CUDA overhead — DEFAULT multiplier only
# The real overhead is MEASURED by infrastructure/gpu_probe.py at startup:
#   actual_overhead = vram_total - vram_free_after_context_init
# This constant is the FALLBACK used only before the first probe completes.
# It is intentionally conservative (high) so the estimator never underestimates
# before real measurement is available.
# Do NOT treat this as a calibrated value — it is a safe upper-bound default.
# ---------------------------------------------------------------------------
CUDA_OVERHEAD_FACTOR_DEFAULT: float = 1.10  # 10% — conservative fallback only

# ---------------------------------------------------------------------------
# Strategy safety thresholds
# Max fraction of VRAM that each strategy is allowed to use.
# Stored here as the canonical source — enums.TrainingMode.safety_threshold()
# reads from this dict to avoid duplication.
# ---------------------------------------------------------------------------
SAFETY_THRESHOLDS: dict[str, float] = {
    "full_finetune": 0.72,
    "lora_plus":     0.80,
    "lora":          0.80,
    "dora":          0.82,
    "qlora":         0.90,
}

# ---------------------------------------------------------------------------
# Strategy selection order (highest quality first)
# Strategy selector traverses this list and picks first that fits.
# ---------------------------------------------------------------------------
STRATEGY_PRIORITY_ORDER: tuple[str, ...] = (
    "full_finetune",
    "lora_plus",
    "lora",
    "dora",
    "qlora",
)

# ---------------------------------------------------------------------------
# Scoring weights for strategy selection (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORING_WEIGHT_MEMORY_SAFETY:    float = 0.40
SCORING_WEIGHT_TRAINING_SPEED:   float = 0.25
SCORING_WEIGHT_PARAM_EFFICIENCY: float = 0.20
SCORING_WEIGHT_USER_PREFERENCE:  float = 0.15

# ---------------------------------------------------------------------------
# Batch size search bounds
# ---------------------------------------------------------------------------
BATCH_SEARCH_LOW:  int = 1
BATCH_SEARCH_HIGH: int = 512

# ---------------------------------------------------------------------------
# Pressure thresholds (VRAM fraction)
# ---------------------------------------------------------------------------
PRESSURE_GREEN:     float = 0.72
PRESSURE_YELLOW:    float = 0.85
PRESSURE_ORANGE:    float = 0.92
PRESSURE_RED:       float = 0.97
PRESSURE_EMERGENCY: float = 0.99

# ---------------------------------------------------------------------------
# Elastic batch sizer
# ---------------------------------------------------------------------------
ELASTIC_STABILITY_WINDOW:  int   = 3
ELASTIC_GROWTH_PHASE1:     float = 1.05
ELASTIC_GROWTH_PHASE2:     float = 1.10

# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------
WATCHDOG_HEARTBEAT_TIMEOUT_S: float = 30.0
MAX_OOM_RETRIES:              int   = 5
MAX_NAN_RETRIES:              int   = 10
MAX_INF_RETRIES:              int   = 5
MAX_DEADLOCK_RETRIES:         int   = 3

# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
DRYRUN_TIMEOUT_S:          float = 45.0
DRYRUN_WARMUP_STEPS:       int   = 2
DRYRUN_MEASURE_STEPS:      int   = 3
HYBRID_DRYRUN_WEIGHT:      float = 0.70
HYBRID_ANALYTICAL_WEIGHT:  float = 0.30
CONFIDENCE_THRESHOLD:      float = 0.85   # below this, attempt dry-run

# ---------------------------------------------------------------------------
# Infrastructure probing
# ---------------------------------------------------------------------------
NVIDIA_SMI_TIMEOUT_S:      float = 5.0
BANDWIDTH_WARMUP_ITERS:    int   = 3
BANDWIDTH_MEASURE_ITERS:   int   = 5
NVME_BENCHMARK_SIZE_BYTES: int   = 128 * MB   # 128 MB test file

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
CHECKPOINT_TIME_INTERVAL_MIN: int = 30
CHECKPOINT_STEP_INTERVAL:     int = 500
CHECKPOINT_KEEP_TOP_N_LOSS:   int = 2
CHECKPOINT_KEEP_LAST_N_TIME:  int = 2

# ---------------------------------------------------------------------------
# Throughput tracking
# ---------------------------------------------------------------------------
THROUGHPUT_WINDOW_SIZE: int = 50   # steps in rolling average
