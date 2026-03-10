"""Reusable GPU mock factories for EGX test suite."""
from egx.core.models import GPUSpec


def make_gpu(
    device_id: int = 0,
    name: str = "MockGPU",
    vram_gb: int = 8,
    cc_major: int = 8,
    cc_minor: int = 0,
    bw_gbps: float = 400.0,
    fp16: float = 20.0,
    bf16: float = 20.0,
    flash2: bool = True,
    fp8: bool = False,
    nvlink: tuple = (),
    vendor: str = "nvidia",
) -> GPUSpec:
    return GPUSpec(
        device_id=device_id,
        name=name,
        vram_bytes=vram_gb * 1024**3,
        compute_capability=(cc_major, cc_minor),
        memory_bandwidth_gbps=bw_gbps,
        fp16_tflops=fp16,
        bf16_tflops=bf16,
        supports_flash_attn2=flash2,
        supports_fp8=fp8,
        nvlink_peer_ids=nvlink,
        vendor=vendor,
    )


LAPTOP_GPU = lambda: make_gpu(name="RTX 4060 Laptop", vram_gb=8, cc_major=8, cc_minor=9)
WORKSTATION_GPU = lambda: make_gpu(name="RTX A6000", vram_gb=48, cc_major=8, cc_minor=6, bw_gbps=768.0, fp16=38.7, bf16=38.7)
A100_GPU = lambda: make_gpu(name="A100-80GB", vram_gb=80, bw_gbps=2039.0, fp16=312.0, bf16=312.0, nvlink=(1,))
H100_GPU = lambda: make_gpu(name="H100-SXM", vram_gb=80, cc_major=9, cc_minor=0, bw_gbps=3350.0, fp16=989.0, bf16=989.0, fp8=True, nvlink=(1,))
CPU_ONLY = lambda: make_gpu(device_id=-1, name="System CPU", vram_gb=0, cc_major=0, cc_minor=0, bw_gbps=50.0, fp16=1.0, bf16=1.0, flash2=False, vendor="cpu")
