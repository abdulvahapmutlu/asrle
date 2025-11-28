import os
import tempfile
import numpy as np

from asrle.backends.registry import BackendRegistry
from asrle.core.engine import Engine
from asrle.utils.audio import write_wav


def test_engine_dummy_runs():
    engine = Engine(BackendRegistry())
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "silence.wav")
        write_wav(wav, np.zeros((16000,), dtype=np.float32), 16000)

        report = engine.analyze(
            audio_path=wav,
            backend_name="dummy",
            backend_params={"text": "hello world", "seconds": 1.0},
            reference_text="hello world",
            repeats=1,
            out_dir=None,
        )
    assert report.transcript.text == "hello world"
    assert report.wer is not None
    assert report.wer.wer == 0.0
