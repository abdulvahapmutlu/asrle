from __future__ import annotations

import os

from asrle.types import AnalysisReport
from asrle.utils.io import ensure_dir, write_json


def write_artifacts(report: AnalysisReport, out_dir: str) -> dict[str, str]:
    ensure_dir(out_dir)
    artifacts_dir = os.path.join(out_dir, "artifacts")
    ensure_dir(artifacts_dir)

    paths: dict[str, str] = {}

    if report.attribution:
        p = os.path.join(artifacts_dir, "attribution.json")
        write_json(
            p,
            {"top": sorted([a.__dict__ for a in report.attribution], key=lambda x: x["wer"], reverse=True)[:50]},
        )
        paths["attribution"] = p

    if report.latency:
        p = os.path.join(artifacts_dir, "latency.json")
        write_json(p, report.latency)
        paths["latency"] = p

    if report.drift:
        p = os.path.join(artifacts_dir, "drift.json")
        write_json(p, report.drift)
        paths["drift"] = p

    if report.word_attribution:
        p = os.path.join(artifacts_dir, "word_attribution.json")
        write_json(p, report.word_attribution)
        paths["word_attribution"] = p

    return paths
