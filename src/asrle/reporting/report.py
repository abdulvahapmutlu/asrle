from __future__ import annotations

from asrle.types import AnalysisReport
from asrle.utils.io import ensure_dir, write_json


def write_report_json(report: AnalysisReport, out_dir: str) -> str:
    ensure_dir(out_dir)
    path = f"{out_dir}/report.json"
    write_json(path, report)
    return path
