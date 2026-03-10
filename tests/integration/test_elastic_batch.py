"""Integration: OOM simulation → batch halving → recovery."""
import unittest


class TestElasticBatch(unittest.TestCase):
    def test_cascading_oom_recovery(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=64)
        sizes = []
        for _ in range(5):
            sizes.append(eb.on_oom())
        self.assertEqual(sizes, [32, 16, 8, 4, 2])
        self.assertEqual(eb.oom_count, 5)

    def test_batch_never_below_minimum(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=4, min_batch=2)
        eb.on_oom()  # 2
        eb.on_oom()  # still 2
        eb.on_oom()  # still 2
        self.assertEqual(eb.current_batch, 2)

    def test_batch_increase_on_low_pressure(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=8, max_batch=64)
        new = eb.try_increase(vram_usage_pct=0.3)
        self.assertEqual(new, 16)
        new = eb.try_increase(vram_usage_pct=0.3)
        self.assertEqual(new, 32)

    def test_no_increase_on_high_pressure(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=8)
        new = eb.try_increase(vram_usage_pct=0.9)
        self.assertEqual(new, 8)


if __name__ == "__main__":
    unittest.main()
