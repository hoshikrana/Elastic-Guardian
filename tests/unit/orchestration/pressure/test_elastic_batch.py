"""Unit tests for ElasticBatchResizer."""

import unittest


class TestElasticBatchResizer(unittest.TestCase):
    def test_initial_batch(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

        eb = ElasticBatchResizer(initial_batch=32)
        self.assertEqual(eb.current_batch, 32)

    def test_oom_halves(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

        eb = ElasticBatchResizer(initial_batch=16)
        self.assertEqual(eb.on_oom(), 8)
        self.assertEqual(eb.on_oom(), 4)

    def test_min_floor(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

        eb = ElasticBatchResizer(initial_batch=2, min_batch=2)
        eb.on_oom()
        self.assertEqual(eb.current_batch, 2)

    def test_try_increase(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

        eb = ElasticBatchResizer(initial_batch=4, max_batch=16)
        self.assertEqual(eb.try_increase(0.3), 8)

    def test_no_increase_high_usage(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

        eb = ElasticBatchResizer(initial_batch=4)
        self.assertEqual(eb.try_increase(0.9), 4)

    def test_oom_count_property(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

        eb = ElasticBatchResizer(initial_batch=8)
        eb.on_oom()
        eb.on_oom()
        self.assertEqual(eb.oom_count, 2)


if __name__ == "__main__":
    unittest.main()
