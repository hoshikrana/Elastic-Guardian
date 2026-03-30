"""GPU Validation: NVLink detection tests."""

import unittest


class TestNVLinkDetection(unittest.TestCase):
    def test_nvlink_peers_detected(self):
        from tests.mocks.mock_gpu import make_gpu

        gpu = make_gpu(nvlink=(1, 2))
        self.assertEqual(gpu.nvlink_peer_ids, (1, 2))

    def test_no_nvlink_empty(self):
        from tests.mocks.mock_gpu import make_gpu

        gpu = make_gpu(nvlink=())
        self.assertEqual(gpu.nvlink_peer_ids, ())

    def test_topology_interconnect_nvlink(self):
        from egx.infrastructure.topology_builder import TopologyBuilder
        from tests.mocks.mock_gpu import make_gpu

        gpus = [make_gpu(nvlink=(1,), device_id=0), make_gpu(nvlink=(0,), device_id=1)]
        topo = TopologyBuilder().build(gpus)
        from egx.core.enums import InterconnectType

        self.assertEqual(topo.interconnect, InterconnectType.NVLINK)

    def test_topology_interconnect_pcie_fallback(self):
        from egx.infrastructure.topology_builder import TopologyBuilder
        from tests.mocks.mock_gpu import make_gpu

        gpus = [make_gpu(nvlink=(), device_id=0), make_gpu(nvlink=(), device_id=1)]
        topo = TopologyBuilder().build(gpus)
        from egx.core.enums import InterconnectType

        self.assertEqual(topo.interconnect, InterconnectType.PCIE)


if __name__ == "__main__":
    unittest.main()
