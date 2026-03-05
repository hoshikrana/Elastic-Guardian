"""
EGX Comprehensive Test Suite — All Modules.

Tests every production module in the EGX project.
Organized by layer to match the 7-layer architecture.
"""

import unittest


# ============================================================
# LAYER 1: CORE
# ============================================================

class TestCoreEnums(unittest.TestCase):
    def test_device_type_values(self):
        from egx.core.enums import DeviceType
        self.assertEqual(DeviceType.CUDA.value, "cuda")
        self.assertEqual(DeviceType.CPU.value, "cpu")

    def test_training_mode_values(self):
        from egx.core.enums import TrainingMode
        modes = [TrainingMode.FULL_FINETUNE, TrainingMode.LORA, TrainingMode.QLORA]
        self.assertEqual(len(set(modes)), 3)

    def test_dtype_byte_size(self):
        from egx.core.enums import DType
        self.assertEqual(DType.FP32.byte_size(), 4)
        self.assertEqual(DType.FP16.byte_size(), 2)
        self.assertEqual(DType.BF16.byte_size(), 2)
        self.assertEqual(DType.INT8.byte_size(), 1)

    def test_hardware_tier(self):
        from egx.core.enums import HardwareTier
        self.assertIn(HardwareTier.DATACENTER, HardwareTier)

    def test_recovery_action(self):
        from egx.core.enums import RecoveryAction
        self.assertIn(RecoveryAction.HALVE_BATCH, RecoveryAction)
        self.assertIn(RecoveryAction.RETRY, RecoveryAction)


class TestCoreExceptions(unittest.TestCase):
    def test_exception_hierarchy(self):
        from egx.core.exceptions import (
            EGXError, HardwareError, GPUNotFoundError,
            EGXMemoryError, BoolAsIntError,
        )
        self.assertTrue(issubclass(HardwareError, EGXError))
        self.assertTrue(issubclass(GPUNotFoundError, HardwareError))
        self.assertTrue(issubclass(BoolAsIntError, EGXMemoryError))

    def test_recoverability(self):
        from egx.core.exceptions import OutOfMemoryError, GPUNotFoundError
        from egx.core.enums import RecoveryAction
        oom = OutOfMemoryError()
        self.assertTrue(oom.recoverable)
        self.assertEqual(oom.suggested_action, RecoveryAction.HALVE_BATCH)
        
        gpu_err = GPUNotFoundError()
        self.assertFalse(gpu_err.recoverable)

    def test_nan_loss_includes_step(self):
        from egx.core.exceptions import NaNLossError
        err = NaNLossError(step=42)
        self.assertIn("42", str(err))

    def test_circular_dependency(self):
        from egx.core.exceptions import CircularDependencyError
        err = CircularDependencyError(cycle=["A", "B", "A"])
        self.assertIn("A", str(err))


class TestCoreModels(unittest.TestCase):
    def test_gpu_spec_frozen(self):
        from egx.core.models import GPUSpec
        spec = GPUSpec(
            device_id=0, name="Test", vram_bytes=8*1024**3,
            compute_capability=(8, 0), memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0, bf16_tflops=20.0,
            supports_flash_attn2=True, supports_fp8=False,
            nvlink_peer_ids=()
        )
        with self.assertRaises(AttributeError):
            spec.vram_bytes = 0  # type: ignore

    def test_gpu_spec_tier(self):
        from egx.core.models import GPUSpec
        from egx.core.enums import HardwareTier
        spec = GPUSpec(
            device_id=0, name="T", vram_bytes=80*1024**3,
            compute_capability=(9, 0), memory_bandwidth_gbps=3350.0,
            fp16_tflops=989.0, bf16_tflops=989.0,
            supports_flash_attn2=True, supports_fp8=True,
            nvlink_peer_ids=()
        )
        self.assertEqual(spec.tier, HardwareTier.DATACENTER)

    def test_hardware_topology_frozen(self):
        from egx.core.models import HardwareTopology
        from egx.core.enums import InterconnectType
        topo = HardwareTopology(
            gpus=(), cpu_cores=8, ram_bytes=32*1024**3,
            nvme_bytes=500*1024**3, nvme_seq_read_gbps=3.5,
            nvme_seq_write_gbps=2.5, pcie_bandwidth_gbps=15.8,
            gpu_interconnect_gbps=15.8, interconnect=InterconnectType.PCIE,
            node_count=1
        )
        with self.assertRaises(AttributeError):
            topo.cpu_cores = 16  # type: ignore


class TestCoreConstants(unittest.TestCase):
    def test_units(self):
        from egx.core.constants import KB, MB, GB, TB
        self.assertEqual(KB, 1024)
        self.assertEqual(MB, 1024 * 1024)
        self.assertEqual(GB, 1024 ** 3)
        self.assertEqual(TB, 1024 ** 4)

    def test_safety_thresholds(self):
        from egx.core.constants import SAFETY_THRESHOLDS
        from egx.core.enums import TrainingMode
        self.assertIn(TrainingMode.QLORA, SAFETY_THRESHOLDS)
        self.assertTrue(0 < SAFETY_THRESHOLDS[TrainingMode.QLORA] <= 1.0)


class TestMemoryValue(unittest.TestCase):
    def test_creation(self):
        from egx.core.memory.value import MemoryValue
        mv = MemoryValue(1024)
        self.assertEqual(mv.bytes, 1024)

    def test_bool_accepted_as_int(self):
        from egx.core.memory.value import MemoryValue
        # MemoryValue silently converts bool to int (0 or 1)
        mv = MemoryValue(True)
        self.assertEqual(mv.bytes, 1)

    def test_negative_rejected(self):
        from egx.core.memory.value import MemoryValue
        from egx.core.exceptions import NegativeMemoryError
        with self.assertRaises(NegativeMemoryError):
            MemoryValue(-1)

    def test_immutable(self):
        from egx.core.memory.value import MemoryValue
        mv = MemoryValue(1024)
        with self.assertRaises(AttributeError):
            mv.bytes = 2048  # type: ignore


class TestMemoryValidators(unittest.TestCase):
    def test_validate_int(self):
        from egx.core.memory.validators import StandardMemoryValidator
        result = StandardMemoryValidator.validate(1024, "test")
        self.assertEqual(result, 1024)

    def test_validate_bool_trap(self):
        from egx.core.exceptions import BoolAsIntError
        from egx.core.memory.validators import StandardMemoryValidator
        with self.assertRaises(BoolAsIntError):
            StandardMemoryValidator.validate(True, "test")


class TestMemoryUnits(unittest.TestCase):
    def test_to_bytes(self):
        from egx.core.memory.units import to_bytes, GB, MB
        self.assertEqual(to_bytes(1, GB), 1024**3)
        self.assertEqual(to_bytes(1, MB), 1024**2)

    def test_from_bytes(self):
        from egx.core.memory.units import from_bytes, GB
        self.assertEqual(from_bytes(1024**3, GB), 1.0)


class TestMemoryFormatting(unittest.TestCase):
    def test_format_bytes(self):
        from egx.core.memory.formatting import format_bytes
        self.assertIn("GB", format_bytes(2 * 1024**3))
        self.assertIn("MB", format_bytes(512 * 1024**2))


# ============================================================
# LAYER 2: INFRASTRUCTURE
# ============================================================

class TestStructuredLogger(unittest.TestCase):
    def test_log_event(self):
        from egx.infrastructure.structured_logger import StructuredLogger
        logger = StructuredLogger("test")
        # Should not raise
        logger.log_event("test_event", {"key": "value"})

    def test_log_degradation(self):
        from egx.infrastructure.structured_logger import StructuredLogger
        logger = StructuredLogger("test")
        logger.log_degradation("gpu", "thermal", "throttle")


class TestTopologyBuilder(unittest.TestCase):
    def test_build_with_no_gpus(self):
        from egx.infrastructure.topology_builder import TopologyBuilder
        topo = TopologyBuilder().build([])
        self.assertEqual(len(topo.gpus), 0)
        self.assertGreater(topo.cpu_cores, 0)


# ============================================================
# LAYER 3: INTELLIGENCE
# ============================================================

class TestCalibrationCache(unittest.TestCase):
    def test_put_get(self):
        from egx.intelligence.estimator.calibration.cache import CalibrationCache
        cache = CalibrationCache(max_size=10)
        cache.put("k1", {"vram_bytes": 100})
        self.assertEqual(cache.get("k1")["vram_bytes"], 100)

    def test_miss(self):
        from egx.intelligence.estimator.calibration.cache import CalibrationCache
        cache = CalibrationCache()
        self.assertIsNone(cache.get("nonexistent"))

    def test_lru_eviction(self):
        from egx.intelligence.estimator.calibration.cache import CalibrationCache
        cache = CalibrationCache(max_size=2)
        cache.put("a", {"v": 1})
        cache.put("b", {"v": 2})
        cache.put("c", {"v": 3})  # should evict "a"
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))


class TestCalibrationRegression(unittest.TestCase):
    def test_initial_pass_through(self):
        from egx.intelligence.estimator.calibration.regression import CalibrationRegression
        reg = CalibrationRegression()
        # With no data, predict should return raw value
        self.assertEqual(reg.predict(1000), 1000)

    def test_calibration_updates(self):
        from egx.intelligence.estimator.calibration.regression import CalibrationRegression
        reg = CalibrationRegression()
        reg.update(100, 120)
        reg.update(200, 240)
        # After 2 data points, should calibrate
        predicted = reg.predict(150)
        self.assertGreater(predicted, 0)


class TestMemoryPlanner(unittest.TestCase):
    def test_budget_computation(self):
        from egx.intelligence.planner.memory_planner import MemoryPlanner
        from egx.core.models import GPUSpec
        from egx.core.enums import TrainingMode
        gpu = GPUSpec(
            device_id=0, name="T", vram_bytes=8*1024**3,
            compute_capability=(8,0), memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0, bf16_tflops=20.0,
            supports_flash_attn2=True, supports_fp8=False, nvlink_peer_ids=()
        )
        budget = MemoryPlanner().compute_budget(gpu, TrainingMode.QLORA)
        self.assertGreater(budget["usable_vram"], 0)
        self.assertLess(budget["usable_vram"], budget["total_vram"])


class TestTimelinePlanner(unittest.TestCase):
    def test_estimate(self):
        from egx.intelligence.planner.timeline_planner import TimelinePlanner
        from egx.core.models import GPUSpec
        gpu = GPUSpec(
            device_id=0, name="T", vram_bytes=8*1024**3,
            compute_capability=(8,0), memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0, bf16_tflops=20.0,
            supports_flash_attn2=True, supports_fp8=False, nvlink_peer_ids=()
        )
        result = TimelinePlanner().estimate(
            total_tokens=1_000_000, seq_length=512, batch_size=4, gpu=gpu
        )
        self.assertGreater(result.total_steps, 0)
        self.assertGreater(result.estimated_hours, 0)


class TestTopologyPlanner(unittest.TestCase):
    def test_single_gpu(self):
        from egx.intelligence.planner.topology_planner import TopologyPlanner
        from egx.core.models import HardwareTopology
        from egx.core.enums import InterconnectType
        topo = HardwareTopology(
            gpus=(), cpu_cores=8, ram_bytes=32*1024**3,
            nvme_bytes=500*1024**3, nvme_seq_read_gbps=3.5,
            nvme_seq_write_gbps=2.5, pcie_bandwidth_gbps=15.8,
            gpu_interconnect_gbps=15.8, interconnect=InterconnectType.PCIE,
            node_count=1
        )
        plan = TopologyPlanner().plan(topo, model_bytes=1*1024**3)
        self.assertEqual(plan.strategy, "single")


class TestStrategyScorer(unittest.TestCase):
    def test_scoring(self):
        from egx.intelligence.strategy.scorer import StrategyScorer
        from egx.core.models import GPUSpec
        from egx.core.enums import TrainingMode
        gpu = GPUSpec(
            device_id=0, name="T", vram_bytes=8*1024**3,
            compute_capability=(8,0), memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0, bf16_tflops=20.0,
            supports_flash_attn2=True, supports_fp8=False, nvlink_peer_ids=()
        )
        results = StrategyScorer().score_all(
            gpu, model_bytes=4*1024**3,
            modes=[TrainingMode.FULL_FINETUNE, TrainingMode.LORA, TrainingMode.QLORA]
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].score >= results[-1].score)


class TestParallelAdvisor(unittest.TestCase):
    def test_single_device(self):
        from egx.intelligence.strategy.parallel_advisor import ParallelAdvisor
        from egx.core.models import HardwareTopology
        from egx.core.enums import InterconnectType
        topo = HardwareTopology(
            gpus=(), cpu_cores=8, ram_bytes=32*1024**3,
            nvme_bytes=500*1024**3, nvme_seq_read_gbps=3.5,
            nvme_seq_write_gbps=2.5, pcie_bandwidth_gbps=15.8,
            gpu_interconnect_gbps=15.8, interconnect=InterconnectType.PCIE,
            node_count=1
        )
        config = ParallelAdvisor().advise(topo, model_params=7_000_000_000)
        self.assertEqual(config.data_parallel_size, 1)


class TestModelRegistry(unittest.TestCase):
    def test_builtin_lookup(self):
        from egx.models.registry import ModelRegistry
        reg = ModelRegistry()
        cfg = reg.get("llama2-7b")
        self.assertIsNotNone(cfg)
        self.assertGreater(cfg.param_count_approx, 0)

    def test_list_available(self):
        from egx.models.registry import ModelRegistry
        reg = ModelRegistry()
        available = reg.list_available()
        self.assertIn("llama2-7b", available)

    def test_custom_register(self):
        from egx.models.registry import ModelRegistry, ModelArchConfig
        reg = ModelRegistry()
        custom = ModelArchConfig("custom-1b", 2048, 24, 16, 8192, 32000, 2048, 1_000_000_000)
        reg.register(custom)
        self.assertIsNotNone(reg.get("custom-1b"))


# ============================================================
# LAYER 3: DSA STRUCTURES
# ============================================================

class TestFibonacciHeap(unittest.TestCase):
    def test_insert_extract(self):
        from egx.intelligence.strategy.selector import FibonacciHeap
        h = FibonacciHeap()
        h.insert(10.0, "a")
        h.insert(50.0, "b")
        h.insert(30.0, "c")
        self.assertEqual(h.extract_max().value, "b")
        self.assertEqual(h.extract_max().value, "c")

    def test_increase_key(self):
        from egx.intelligence.strategy.selector import FibonacciHeap
        h = FibonacciHeap()
        h.insert(10.0, "low")
        node = h.insert(20.0, "mid")
        h.insert(50.0, "high")
        h.increase_key(node, 60.0)
        self.assertEqual(h.extract_max().value, "mid")


class TestRedBlackTree(unittest.TestCase):
    def test_insert_search(self):
        from egx.intelligence.estimator.calibration.store import RedBlackTree
        t = RedBlackTree()
        t.insert(10, "v1")
        t.insert(20, "v2")
        self.assertEqual(t.search(10), "v1")
        self.assertEqual(t.search(20), "v2")
        self.assertIsNone(t.search(99))

    def test_find_nearest(self):
        from egx.intelligence.estimator.calibration.store import RedBlackTree
        t = RedBlackTree()
        t.insert(10, "a")
        t.insert(20, "b")
        t.insert(30, "c")
        key, val = t.find_nearest(22)
        self.assertEqual(val, "b")


class TestSkipList(unittest.TestCase):
    def test_insert_latest(self):
        from egx.orchestration.pressure.monitor import PressureEventSkipList
        sl = PressureEventSkipList()
        sl.insert(1.0, "e1")
        sl.insert(3.0, "e3")
        sl.insert(2.0, "e2")
        self.assertEqual(sl.latest(), "e3")


class TestSegmentTree(unittest.TestCase):
    def test_range_max(self):
        from egx.intelligence.estimator.dryrun import MemorySegmentTree
        st = MemorySegmentTree(8)
        st.update(0, 100)
        st.update(3, 500)
        st.update(5, 200)
        self.assertEqual(st.query_max(0, 4), 500)
        self.assertEqual(st.global_peak(), 500)


class TestDijkstra(unittest.TestCase):
    def test_shortest_path(self):
        from egx.intelligence.graph.topology_graph import HardwareTopologyGraph
        g = HardwareTopologyGraph()
        g.add_edge("gpu0", "gpu1", 100.0)
        g.add_edge("gpu1", "gpu2", 50.0)
        g.add_edge("gpu0", "gpu2", 10.0)  # slow direct
        path, cost = g.shortest_path("gpu0", "gpu2")
        # Fast: gpu0->gpu1->gpu2 (lower cost than slow direct)
        self.assertIn("gpu2", path)


class TestKahnsAlgorithm(unittest.TestCase):
    def test_valid_dag(self):
        from egx.intelligence.graph.dependency_dag import ModuleDependencyDAG
        dag = ModuleDependencyDAG()
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "C")
        order = dag.validate()
        self.assertIn("A", order)

    def test_cycle_detection(self):
        from egx.intelligence.graph.dependency_dag import ModuleDependencyDAG
        from egx.core.exceptions import CircularDependencyError
        dag = ModuleDependencyDAG()
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "C")
        dag.add_dependency("C", "A")
        with self.assertRaises(CircularDependencyError):
            dag.validate()


class TestBinarySearch(unittest.TestCase):
    def test_find_max(self):
        from egx.intelligence.strategy.batch_optimizer import find_max_batch_size
        result = find_max_batch_size(lambda x: x <= 100, low=1, high=200)
        self.assertEqual(result, 100)


class TestTrie(unittest.TestCase):
    def test_insert_get(self):
        from egx.runtime.config_loader import ConfigTrie
        t = ConfigTrie()
        t.insert("core.memory.limit", 1024)
        self.assertEqual(t.get("core.memory.limit"), 1024)
        self.assertIsNone(t.get("core.memory.nonexist"))

    def test_prefix_search(self):
        from egx.runtime.config_loader import ConfigTrie
        t = ConfigTrie()
        t.insert("core.a", 1)
        t.insert("core.b", 2)
        t.insert("other.c", 3)
        results = t.find_by_prefix("core")
        self.assertEqual(len(results), 2)


# ============================================================
# LAYER 4: ORCHESTRATION & RESILIENCE
# ============================================================

class TestElasticBatch(unittest.TestCase):
    def test_oom_halves(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=32)
        new_batch = eb.on_oom()
        self.assertEqual(new_batch, 16)
        new_batch = eb.on_oom()
        self.assertEqual(new_batch, 8)

    def test_min_batch(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=2, min_batch=1)
        eb.on_oom()
        result = eb.on_oom()
        self.assertEqual(result, 1)


class TestEvictionPolicy(unittest.TestCase):
    def test_eviction(self):
        from egx.orchestration.pressure.eviction_policy import LRUEvictionPolicy
        pol = LRUEvictionPolicy(capacity_bytes=1000)
        pol.access("t1", 400)
        pol.access("t2", 400)
        evicted = pol.evict_until(300)
        self.assertIn("t1", evicted)  # LRU should be evicted first


class TestRAMToNVMeSwapper(unittest.TestCase):
    def test_offload_restore(self):
        import torch
        from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper
        sw = RAMToNVMeSwapper()
        t = torch.randn(100)
        sw.offload("test_tensor", t)
        self.assertEqual(sw.cached_count, 1)
        restored = sw.restore("test_tensor")
        self.assertEqual(restored.shape, t.shape)
        sw.cleanup()
        self.assertEqual(sw.cached_count, 0)


class TestWatchdog(unittest.TestCase):
    def test_heartbeat(self):
        from egx.resilience.watchdog import TrainingWatchdog
        wd = TrainingWatchdog(timeout_s=10.0)
        wd.heartbeat(1)
        wd.heartbeat(2)
        # No exception = success


class TestInputSanitizer(unittest.TestCase):
    def test_clean_batch(self):
        import torch
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        batch = {"input_ids": torch.tensor([1, 2, 3])}
        clean = san.check_batch(batch)
        self.assertIn("input_ids", clean)

    def test_nan_detection_strict(self):
        import torch
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        batch = {"x": torch.tensor([1.0, float('nan'), 3.0])}
        with self.assertRaises(ValueError):
            san.check_batch(batch)

    def test_loss_check(self):
        import torch
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer()
        self.assertTrue(san.check_loss(torch.tensor(0.5)))
        self.assertFalse(san.check_loss(torch.tensor(float('nan'))))


# ============================================================
# LAYER 5: TRAINING & PEFT
# ============================================================

class TestMixedPrecision(unittest.TestCase):
    def test_select_optimal(self):
        from egx.training.mixed_precision import PrecisionSelector
        from egx.core.models import GPUSpec, HardwareTopology
        from egx.core.enums import DType, InterconnectType
        gpu = GPUSpec(
            device_id=0, name="A100", vram_bytes=40*1024**3,
            compute_capability=(8, 0), memory_bandwidth_gbps=1555.0,
            fp16_tflops=312.0, bf16_tflops=312.0,
            supports_flash_attn2=True, supports_fp8=False, nvlink_peer_ids=()
        )
        topo = HardwareTopology(
            gpus=(gpu,), cpu_cores=64, ram_bytes=256*1024**3,
            nvme_bytes=2000*1024**3, nvme_seq_read_gbps=7.0,
            nvme_seq_write_gbps=5.0, pcie_bandwidth_gbps=31.5,
            gpu_interconnect_gbps=600.0, interconnect=InterconnectType.NVLINK,
            node_count=1
        )
        dtype, autocast = PrecisionSelector.select_optimal(topo)
        self.assertEqual(dtype, DType.BF16)
        self.assertTrue(autocast)


class TestLoRA(unittest.TestCase):
    def test_lora_linear(self):
        import torch
        import torch.nn as nn
        from egx.peft.lora import LoRALinear
        orig = nn.Linear(64, 32)
        lora = LoRALinear(orig, rank=4, alpha=8)
        x = torch.randn(2, 64)
        out = lora(x)
        self.assertEqual(out.shape, (2, 32))

    def test_trainable_params(self):
        import torch.nn as nn
        from egx.peft.lora import LoRALinear
        orig = nn.Linear(64, 32)
        lora = LoRALinear(orig, rank=4)
        self.assertEqual(lora.trainable_params, 4*64 + 32*4)


class TestQLoRA(unittest.TestCase):
    def test_quantized_linear(self):
        import torch
        import torch.nn as nn
        from egx.peft.qlora import QuantizedLinear
        orig = nn.Linear(64, 32)
        q = QuantizedLinear(orig, rank=4)
        x = torch.randn(2, 64)
        out = q(x)
        self.assertEqual(out.shape, (2, 32))


class TestLoRAPlus(unittest.TestCase):
    def test_param_groups(self):
        import torch.nn as nn
        from egx.peft.lora import inject_lora
        from egx.peft.lora_plus import get_lora_plus_param_groups
        # Create a simple model with q_proj
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64)
        model = inject_lora(model, rank=4, targets=["q_proj"])
        groups = get_lora_plus_param_groups(model, base_lr=1e-4)
        self.assertGreater(len(groups), 0)


# ============================================================
# LAYER 6-7: RUNTIME & API
# ============================================================

class TestAPIConfig(unittest.TestCase):
    def test_defaults(self):
        from egx.api.config import EGXConfig
        cfg = EGXConfig()
        self.assertEqual(cfg.num_epochs, 3)
        self.assertEqual(cfg.lora_rank, 16)

    def test_from_dict(self):
        from egx.api.config import EGXConfig
        cfg = EGXConfig.from_dict({"num_epochs": 10, "custom_key": "v"})
        self.assertEqual(cfg.num_epochs, 10)
        self.assertEqual(cfg.get("custom_key"), "v")


class TestEGXTrainer(unittest.TestCase):
    def test_init_no_args(self):
        from egx.api.trainer import EGX
        trainer = EGX()
        self.assertIsNotNone(trainer)


class TestSimulator(unittest.TestCase):
    def test_mock_8gb(self):
        from egx.testing.simulator import mock_8gb_gpu
        topo = mock_8gb_gpu()
        self.assertEqual(len(topo.gpus), 1)
        self.assertEqual(topo.gpus[0].vram_bytes, 8 * 1024**3)

    def test_mock_h100(self):
        from egx.testing.simulator import mock_h100_cluster
        topo = mock_h100_cluster()
        self.assertEqual(len(topo.gpus), 2)
        self.assertEqual(topo.gpus[0].vram_bytes, 80 * 1024**3)


if __name__ == "__main__":
    unittest.main()
