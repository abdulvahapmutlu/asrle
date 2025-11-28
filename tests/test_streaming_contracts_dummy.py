from asrle.backends.registry import BackendRegistry
from asrle.contracts.streaming_contracts import validate_streaming_contract


def test_dummy_streaming_contract():
    reg = BackendRegistry()
    b = reg.create("dummy")
    ok, issues = validate_streaming_contract(b)
    assert ok, issues
