# ASR-LE (Automatic Speech Recognition – Latency & Error Explorer)

ASR-LE is an advanced ASR evaluation toolkit that goes beyond WER by adding **time-aware analysis**:

- **Word/token timelines** (timestamps + confidence)
- **Streaming latency simulation** (chunking/overlap/right-context)
- **Word-level error attribution** (sub/ins/del bursts by time window)
- **“Moments”**: automatically surfaces the *worst* error windows so you can jump directly to problem segments
- **Backend contract tests** so community backends must meet the same interface & quality gates
- **Streamlit dashboard** for exploration, comparisons, and batch runs

> Think of it as: **“perf + quality observability for ASR pipelines”**.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
  - [PyPI (Recommended)](#pypi-recommended)
  - [Minimal](#minimal)
  - [With Whisper Backends](#with-whisper-backends)
  - [With Alignment (CTC forced alignment)](#with-alignment-ctc-forced-alignment)
- [Quickstart (Streamlit)](#quickstart-streamlit)
- [Quickstart (CLI)](#quickstart-cli)
- [How the Analysis Works](#how-the-analysis-works)
  - [WER](#wer)
  - [Tokens & Confidence](#tokens--confidence)
  - [Streaming p95 First-Word Latency](#streaming-p95-first-word-latency)
  - [Word-Level Error Attribution & Timeline Heatmap](#word-level-error-attribution--timeline-heatmap)
  - [Error Moments](#error-moments)
  - [Timestamp Drift Checks](#timestamp-drift-checks)
- [Dataset Runner (Batch)](#dataset-runner-batch)
  - [Manifest Format](#manifest-format)
  - [What You Get](#what-you-get)
- [Backend System](#backend-system)
  - [Backends Included](#backends-included)
  - [Backend Contract Tests](#backend-contract-tests)
  - [Streaming Interface (Optional)](#streaming-interface-optional)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

### 1) True ASR Observability (not just one scalar)
ASR-LE produces a run folder containing:
- `report.json` (machine-readable)
- `report.md` (human-readable)
- `artifacts/` exports for visualization (timeline bins, moments, tokens, etc.)

You can compare runs, batch runs, and analyze worst segments quickly.

### 2) Token-level introspection: confidence + timestamps
For backends that expose tokens (e.g., faster-whisper), you can inspect:
- per-token `word`, `start_s`, `end_s`
- `confidence` (when available)

Even when your alignment-based heatmap is missing, the Streamlit app can build a **confidence heatmap** from tokens as a fallback.

### 3) Streaming p95 first-word latency estimator
ASR-LE can simulate streaming by chunking the audio and measuring:
- first decoded word time
- p50/p95 across repeated simulated runs

This helps you answer real production questions like:
> “Which knobs buy the biggest p95 improvements without retraining?”

### 4) Word-level error attribution with time windows
When reference text is provided, ASR-LE can compute:
- WER (sub/ins/del/hits)
- attribution bursts: *where* errors occur in time
- timeline bins: e.g. each 1 second window gets counts of sub/ins/del

This enables targeted debugging (noise bursts, far-field reverberation zones, etc.).

### 5) Error “moments”
ASR-LE auto-detects the worst 1s windows (with padding) and stores them as **moments** so the dashboard can jump directly.

### 6) Backend contract tests
Community backends must satisfy baseline correctness and shape requirements via a minimal contract.

---

## Installation

### PyPI (Recommended)

```
pip install asrle
```

Optional extras (recommended)
If your package defines extras, install like:

``` 
pip install "asrle[whisper,alignment]" 
```

Typical extras:

- whisper: faster-whisper / whisper backend deps

- alignment: transformers + torchaudio forced alignment deps

### Minimal

```
python -m venv .venv
# Windows PowerShell:
. .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .
````

### With Whisper Backends

Depending on your repo extras, you may expose extras like `[whisper]`. If not, install typical deps manually:

```
pip install faster-whisper
pip install openai-whisper   # optional, HF whisper alternative
pip install ffmpeg-python    # if needed
```

Also ensure **FFmpeg** is available:

* Windows: install FFmpeg and add to PATH
* Linux: `sudo apt-get install ffmpeg`

### With Alignment (CTC forced alignment)

Alignment can use HuggingFace + torchaudio forced alignment when available:

```
pip install transformers torchaudio
```

---

## Quickstart (Streamlit)

Start the dashboard:

```powershell
cd C:\path\to\asrle
streamlit run .\src\asrle\dashboard\streamlit_app.py
```

Open:

* [http://localhost:8501](http://localhost:8501)

### Single Run Workflow

1. Upload audio (or server path)
2. Pick backend (e.g. `faster-whisper`)
3. Optionally paste/upload reference transcript
4. Enable:

   * Word alignment (CTC) (for timestamped attribution)
   * Word-level attribution (moments + bins)
   * Streaming simulation (optional)
5. Run analysis

### What to expect in UI

* **WER** (if reference provided)
* **Latency p95** (estimated using repeats)
* **Streaming first-word latency p95** (streaming mode)
* **Tokens/confidence explorer** (if backend provides tokens)
* **Error heatmap** (if timeline artifacts exist)

> If you don’t see the error heatmap: it means `artifacts/timeline.json` wasn’t created. This usually happens when CTC alignment fails or reference wasn’t provided.

---

## Quickstart (CLI)

If your repo exposes a CLI entrypoint, you can add examples here. A minimal canonical pattern:

```
python -m asrle <command> ...
```

If you don’t have a CLI command yet, Streamlit is the fastest interface.

---

## How the Analysis Works

### WER

WER is computed **word-level** using:

* `jiwer.process_words()` when available (preferred)
* else a Levenshtein fallback

Outputs include:

* `wer`
* `substitutions`, `insertions`, `deletions`, `hits`
* `ref_words`, `hyp_words`

### Tokens & Confidence

Some backends produce per-word tokens with timestamps and confidence.
ASR-LE exposes them in:

* `report.json -> transcript -> segments[*] -> tokens[*]`
* and Streamlit provides a token table + confidence summaries.

### Streaming p95 First-Word Latency

With streaming enabled, ASR-LE:

* chunks audio into overlapping blocks
* runs backend decode loop multiple times (`repeats`)
* estimates p50/p95 `(first_word_latency_s)` and stores percentiles

### Word-Level Error Attribution & Timeline Heatmap

To build time-aware substitution windows, ASR-LE needs:

* reference text
* hypothesis timestamps (from backend tokens)
* reference word timestamps (from CTC alignment)

If CTC alignment fails (e.g. produces `<unk>`/garbage), then timeline bins may be missing.

Artifacts:

* `artifacts/word_attribution.json`
* `artifacts/timeline.json`

### Error Moments

Moments are the **worst time windows** by error density, saved in:

* `artifacts/error_moments.json`

The UI uses these to jump straight to problematic spans.

### Timestamp Drift Checks

ASR-LE can run a drift check against transcript timestamps to detect:

* non-monotonic segments
* overlaps, gaps, impossible ordering

---

## Dataset Runner (Batch)

You can run a dataset by uploading a manifest CSV in the dashboard.

### Manifest Format

Required columns:

* `audio_path`

Optional:

* `ref_text` (inline reference)
* `ref_path` (path to a .txt file reference)
* any metadata columns (SNR, device, far-field, noise_type, etc.)

Example:

```csv
audio_path,ref_text,snr,far_field
C:\data\a.wav,"hello world",20,false
C:\data\b.wav,"this is a test",5,true
```

### What You Get

The dataset run creates:

```
runs/dataset_<id>/
  manifest.csv
  dataset_summary.json
  items/
    item_00000/
      report.json
      artifacts/...
    item_00001/
      ...
```

The UI can summarize:

* WER distribution (mean/p50/p90)
* latency p50/p95
* first-word latency p50/p95 (streaming mode)

---

## Backend System

### Backends Included

Typical set:

* `dummy` (testing)
* `hf-whisper` (transformers pipeline)
* `faster-whisper` (high-performance Whisper)

### Backend Contract Tests

The **Backend Validator** page runs checks like:

* does transcribe return required fields?
* timestamps monotonic?
* streaming capability validation (if claimed)

### Streaming Interface (Optional)

Some backends can support *true incremental decoding*.
If a backend supports it, ASR-LE can validate and use it.

---

## Docker

Build and run:

```bash
docker build -t asrle:local .
docker run --rm -p 8501:8501 -v "$(pwd)/runs:/app/runs" asrle:local
```

Then open:

* [http://localhost:8501](http://localhost:8501)

---

## CI/CD

GitHub Actions included:

* `.github/workflows/ci.yml` – tests + lint/format checks (best-effort)
* `.github/workflows/docker.yml` – builds and pushes to GHCR on main/tags
* `.github/workflows/release.yml` – PyPI Trusted Publisher

---

## Troubleshooting

### 1) “Heatmap is missing”

The error heatmap relies on `artifacts/timeline.json`.
If it doesn’t exist:

* Provide reference text
* Enable word attribution
* Enable word alignment (CTC)
* Ensure alignment deps are installed (`transformers`, `torchaudio`)
* Check CTC didn’t collapse into `<unk>` outputs (alignment failure)

### 2) WER = 0 even with noisy audio

WER uses the reference transcript. If your reference equals the hypothesis after normalization, WER can still be 0.
Verify:

* the reference text is correct and not accidentally identical
* your normalization is not overly aggressive
* backend isn’t outputting identical transcript due to VAD trimming or normalization

### 3) No tokens/confidence showing

Not all backends return token-level outputs.
Use `faster-whisper` and ensure your backend exposes tokens into:
`transcript.segments[*].tokens`.

### 4) FFmpeg errors

Install FFmpeg and ensure it is in PATH.

---

## Contributing

Contributions are welcome:

* add a backend (must pass contract tests)
* improve alignment robustness
* add dashboards / better visualizations
* improve streaming accuracy and incremental decoding support

Recommended dev flow:

1. Create a feature branch
2. Add tests or a minimal reproducible case
3. Ensure CI passes
4. Open a PR

---

## License

This project is licensed under MIT License.

