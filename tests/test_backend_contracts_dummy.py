import os
import tempfile
import numpy as np

from asrle.backends.registry import BackendRegistry
from asrle.contracts.backend_contracts import validate_backend_contract
from asrle.utils.audio import write_wav


def test_dummy_backend_contract_passes():
    reg = BackendRegistry()
    b = reg.create("dummy")

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "silence.wav")
        write_wav(wav, np.zeros((16000,), dtype=np.float32), 16000)

        ok, issues = validate_backend_contract(b, wav)
        assert ok, issues
