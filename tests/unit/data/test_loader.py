"""Unit tests for NVMeDataLoader."""
import unittest


class TestNVMeDataLoader(unittest.TestCase):
    def test_auto_worker_count(self):
        from egx.data.loader import NVMeDataLoader
        from tests.mocks.mock_dataloader import MockDataset
        ds = MockDataset(size=20)
        loader = NVMeDataLoader(ds, batch_size=4)
        self.assertGreaterEqual(loader.num_workers, 0)

    def test_topology_aware_workers(self):
        from egx.data.loader import NVMeDataLoader
        from tests.mocks.mock_dataloader import MockDataset
        from tests.mocks.mock_topology import datacenter_topology
        ds = MockDataset(size=20)
        topo = datacenter_topology()
        loader = NVMeDataLoader(ds, batch_size=4, topology=topo)
        self.assertGreaterEqual(loader.num_workers, 1)

    def test_nvme_prefetch_boost(self):
        from egx.data.loader import NVMeDataLoader
        from tests.mocks.mock_dataloader import MockDataset
        from tests.mocks.mock_topology import datacenter_topology
        ds = MockDataset(size=20)
        topo = datacenter_topology()
        loader = NVMeDataLoader(ds, batch_size=4, topology=topo)
        if loader.num_workers > 0:
            self.assertGreaterEqual(loader.prefetch_factor, 4)


if __name__ == "__main__":
    unittest.main()
