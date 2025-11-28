from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class BackendConfig(BaseModel):
    name: str = Field(default="hf-whisper")
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    enable_profiling: bool = True
    compute_wer: bool = True
    compute_attribution: bool = True
    compute_drift: bool = True
    enable_suggestions: bool = True
    # repeat run count for latency stats (wall-clock)
    repeats: int = 1


class ReportingConfig(BaseModel):
    write_markdown: bool = True
    write_json: bool = True
    out_dir: str = "runs/latest"


class ASRLEConfig(BaseSettings):
    backend: BackendConfig = Field(default_factory=BackendConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    class Config:
        env_prefix = "ASRLE_"
