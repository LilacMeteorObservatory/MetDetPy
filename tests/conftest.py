"""Keep each test independent of the process-lifetime backend lock registry."""
import pytest


@pytest.fixture(autouse=True)
def isolated_backend_device_locks(monkeypatch):
    from MetLib.model import ONNXBackend
    monkeypatch.setattr(ONNXBackend, "_device_locks", {})
