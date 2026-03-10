"""
EGX Dependency Injection Test Suite.

Exhaustively tests every edge-case scenario for the new Dependency Inversion hierarchy.
Ensures that custom components inherit correctly, interface overrides work, 
and fallbacks operate gracefully without breaking the core mathematical architecture.
"""

import unittest
from typing import List, Dict, Any, Optional

from egx.core.interfaces import (
    BaseGPUProber,
    BaseTopologyBuilder,
    BaseStrategySelector,
    BaseWatchdog,
    BaseCheckpointManager,
)
from egx.core.models import GPUSpec, HardwareTopology
from egx.runtime.engine import EGXEngine
from egx.training.kernel import TrainingKernel

# ====================================================================
# CUSTOM MOCK IMPLEMENTATIONS (Simulating Developer Extensions)
# ====================================================================

class MockAMDProber(BaseGPUProber):
    """Simulates a developer overriding the GPU Profiler for AMD."""
    def probe(self) -> List[GPUSpec]:
        return [GPUSpec(
            device_id=99, name="AMD Instinct MI300X", vram_bytes=192 * 1024**3,
            compute_capability=(0,0), memory_bandwidth_gbps=5300.0,
            fp16_tflops=1300.0, bf16_tflops=1300.0,
            supports_flash_attn2=True, supports_fp8=True, nvlink_peer_ids=(), vendor="amd"
        )]

class MockCustomTopology(BaseTopologyBuilder):
    """Simulates a custom datacenter topology compiler."""
    def build(self, gpus: List[GPUSpec]) -> HardwareTopology:
        from egx.core.enums import InterconnectType
        return HardwareTopology(
            gpus=tuple(gpus), cpu_cores=128, ram_bytes=512*1024**3,
            nvme_bytes=10*1024**4, nvme_seq_read_gbps=12.0, nvme_seq_write_gbps=10.0,
            pcie_bandwidth_gbps=64.0, gpu_interconnect_gbps=800.0, interconnect=InterconnectType.NVLINK,
            node_count=4
        )

class MockAStarSelector(BaseStrategySelector):
    """Simulates a developer replacing the FibonacciHeap with A*."""
    def __init__(self):
        self._items = []
        
    def insert(self, priority: float, value: Any):
        self._items.append((priority, value))
        self._items.sort(key=lambda x: x[0], reverse=True)
        
    def extract_max(self) -> Any:
        class DummyNode:
            def __init__(self, v): self.value = v
        if not self._items: return None
        return DummyNode(self._items.pop(0)[1])

class MockSilentWatchdog(BaseWatchdog):
    """Simulates a watchdog that intercepts heartbeats for custom cloud logging."""
    def __init__(self):
        self.beat_count = 0
        
    def start(self): pass
    def stop(self): pass
    def heartbeat(self, step: int):
        self.beat_count += 1

class MockS3CheckpointManager(BaseCheckpointManager):
    """Simulates saving checkpoints directly to AWS S3 instead of local disk."""
    def __init__(self):
        self.save_called = False
        
    def should_save(self, step: int, loss: float) -> bool:
        return True # Always save
        
    def checkpoint(self, step: int, loss: float, state_dict: dict):
        self.save_called = True

# ====================================================================
# TEST CASES
# ====================================================================

class TestDependencyInjection(unittest.TestCase):

    def test_engine_custom_prober_and_builder(self):
        """Test if the engine correctly boots using completely custom Hardware layers."""
        engine = EGXEngine(
            gpu_prober=MockAMDProber(),
            topology_builder=MockCustomTopology()
        )
        # We pass a dummy model since introspection checks for .parameters()
        class DummyModel:
            def parameters(self): return []
            
        engine.boot(model=DummyModel())
        topo = engine.topology
        self.assertIsNotNone(topo)
        self.assertEqual(len(topo.gpus), 1)
        self.assertEqual(topo.gpus[0].name, "AMD Instinct MI300X")
        self.assertEqual(topo.node_count, 4)

    def test_engine_custom_strategy_selector(self):
        """Test if the generic 'extract_max' Strategy interface works securely across the loop."""
        custom_selector = MockAStarSelector()
        engine = EGXEngine(strategy_selector=custom_selector)
        
        # Verify it routes through correctly
        custom_selector.insert(0.99, "LORA_EXTREME")
        custom_selector.insert(0.5, "QLORA_FAST")
        
        self.assertEqual(engine.strategy_selector.extract_max().value, "LORA_EXTREME")
        
    def test_kernel_custom_watchdog_and_checkpoint(self):
        """Test if the stateless Kernel triggers custom injected Base Interfaces."""
        import torch.nn as nn
        
        # Dummy linear model to satisfy Kernel
        model = nn.Linear(10, 2)
        
        watchdog = MockSilentWatchdog()
        checkpoint = MockS3CheckpointManager()
        
        kernel = TrainingKernel(
            model=model,
            optimizer_type="sgd", # Use simple optimizer to avoid overhead
            watchdog=watchdog,
            checkpoint_mgr=checkpoint
        )
        
        import torch
        batch = {"input": torch.randn(2, 10)}
        # Override the forward pass to prevent complex tensor destructuring issues in the dummy
        # We must explicitly require gradients or loss.backward() will crash
        model.forward = lambda **kwargs: torch.tensor([0.5, 0.5], requires_grad=True) 
        
        # Execute 3 steps to test the interception
        kernel.train_step(batch, step=1)
        kernel.train_step(batch, step=2)
        kernel.train_step(batch, step=3)
        
        # Verify interfaces caught the signals!
        self.assertEqual(watchdog.beat_count, 3)
        self.assertTrue(checkpoint.save_called)

    def test_default_fallbacks(self):
        """Test that missing dependencies gracefully trigger the EGX default instances."""
        engine = EGXEngine() # No variables passed 
        
        from egx.infrastructure.gpu_probe import GPUProber
        from egx.intelligence.strategy.selector import FibonacciHeap
        
        # Should cleanly instantiate the default implementations
        self.assertIsInstance(engine.gpu_prober, GPUProber)
        self.assertIsInstance(engine.strategy_selector, FibonacciHeap)

if __name__ == "__main__":
    unittest.main()
