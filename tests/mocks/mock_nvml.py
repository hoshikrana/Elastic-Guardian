"""pynvml stub for running GPU tests on CI without CUDA hardware."""


class MockMemInfo:
    def __init__(self, total: int = 8 * 1024**3, used: int = 2 * 1024**3):
        self.total = total
        self.used = used
        self.free = total - used


class MockNVMLHandle:
    def __init__(
        self, idx: int = 0, name: str = "MockGPU", vram_total: int = 8 * 1024**3
    ):
        self.idx = idx
        self.name = name
        self.vram_total = vram_total


def mock_nvml_init():
    """Replacement for pynvml.nvmlInit()."""
    pass


def mock_nvml_device_count():
    return 1


def mock_nvml_get_handle(idx: int):
    return MockNVMLHandle(idx=idx)


def mock_nvml_get_name(handle):
    return handle.name


def mock_nvml_get_memory(handle):
    return MockMemInfo(total=handle.vram_total)


def mock_nvml_get_compute_capability(handle):
    return 8, 0
