import pytest

from common.memory import (
    free_gpu_memory,
    get_gpu_memory,
    gpu_memory_usage,
    is_memory_exceeded,
)


def test_gpu_memory_usage():
    gpu_memory_usage()


def test_get_gpu_memory():
    result = get_gpu_memory()
    assert isinstance(result, float)
    print(f"GPU Memory: {result} GiB")


def test_free_gpu_memory():
    free_gpu_memory()


@pytest.mark.parametrize("estimated_size_gb", [12, 16, 24, 32, 48, 64])
def test_needs_offloading(estimated_size_gb: int):
    result = is_memory_exceeded(estimated_size_gb)
    assert isinstance(result, bool)
