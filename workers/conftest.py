import pytest

from common.memory import free_gpu_memory


def pytest_sessionfinish(session, exitstatus):
    """Called after all tests have finished."""
    free_gpu_memory()
