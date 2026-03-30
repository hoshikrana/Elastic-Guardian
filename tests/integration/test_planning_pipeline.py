"""Integration: GPU probe → topology → strategy selection → plan."""

import unittest


class TestPlanningPipeline(unittest.TestCase):
    def test_full_planning_flow(self):
        from egx.infrastructure.topology_builder import TopologyBuilder
        from egx.intelligence.strategy.selector import FibonacciHeap
        from egx.intelligence.planner.topology_planner import TopologyPlanner
        from egx.core.enums import TrainingMode
        from tests.mocks.mock_gpu import make_gpu

        gpus = [make_gpu(vram_gb=80), make_gpu(vram_gb=80, device_id=1)]
        topo = TopologyBuilder().build(gpus)
        self.assertEqual(len(topo.gpus), 2)

        heap = FibonacciHeap()
        heap.insert(0.9, TrainingMode.LORA)
        heap.insert(0.8, TrainingMode.QLORA)
        heap.insert(0.5, TrainingMode.FULL_FINETUNE)
        best = heap.extract_max()
        self.assertEqual(best.value, TrainingMode.LORA)

        plan = TopologyPlanner().plan(topo, model_bytes=14 * 1024**3)
        self.assertIn(plan.strategy, ("single", "dp", "tp", "3d"))

    def test_strategy_scorer_ranking(self):
        from egx.intelligence.strategy.scorer import StrategyScorer
        from egx.core.enums import TrainingMode
        from tests.mocks.mock_gpu import make_gpu

        gpu = make_gpu(vram_gb=8)
        results = StrategyScorer().score_all(
            gpu,
            model_bytes=4 * 1024**3,
            modes=[TrainingMode.FULL_FINETUNE, TrainingMode.LORA, TrainingMode.QLORA],
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].score >= results[-1].score)

    def test_memory_planner(self):
        from egx.intelligence.planner.memory_planner import MemoryPlanner
        from egx.core.enums import TrainingMode
        from tests.mocks.mock_gpu import make_gpu

        budget = MemoryPlanner().compute_budget(
            make_gpu(vram_gb=16), TrainingMode.QLORA
        )
        self.assertGreater(budget["usable_vram"], 0)


if __name__ == "__main__":
    unittest.main()
