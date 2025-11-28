from asrle.core.metrics import compute_wer


def test_wer_identity():
    r = compute_wer("hello world", "hello world")
    assert r.wer == 0.0


def test_wer_nonzero():
    r = compute_wer("hello world", "hello there")
    assert r.wer > 0.0
