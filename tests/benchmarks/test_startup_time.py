"""Benchmark: Full boot sequence < 5 seconds."""

import time
import unittest


class TestStartupTime(unittest.TestCase):
    def test_engine_boot_speed(self):
        from egx.runtime.engine import EGXEngine
        from egx.api.config import EGXConfig

        engine = EGXEngine()

        class DummyModel:
            def parameters(self):
                return []

            def named_parameters(self):
                return []

        start = time.perf_counter()
        engine.boot(DummyModel(), EGXConfig())
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"Boot took {elapsed:.2f}s, expected < 5s")

    def test_config_creation_speed(self):
        from egx.api.config import EGXConfig

        start = time.perf_counter()
        for _ in range(1000):
            EGXConfig()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"1K config creations took {elapsed:.2f}s")

    def test_topology_build_speed(self):
        from egx.infrastructure.topology_builder import TopologyBuilder
        from tests.mocks.mock_gpu import make_gpu

        gpus = [make_gpu(device_id=i) for i in range(8)]
        start = time.perf_counter()
        TopologyBuilder().build(gpus)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"Topology build took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
