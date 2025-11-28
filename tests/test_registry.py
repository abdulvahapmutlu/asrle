from asrle.backends.registry import BackendRegistry


def test_registry_lists_dummy():
    reg = BackendRegistry()
    names = reg.list()
    assert "dummy" in names


def test_registry_create_dummy():
    reg = BackendRegistry()
    b = reg.create("dummy")
    assert b.backend_name() == "dummy"
