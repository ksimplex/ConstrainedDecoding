# Setup

Three levels. Most of the repository needs only the first.

| Level | What it runs | Needs |
|---|---|---|
| 1. Core | the package, all tests, the notes' tables (`experiments/notes/`), the oracle studies (`experiments/studies/`, `experiments/archive/`) | Python, PyTorch on CPU. No model weights, no datasets, no GPU. |
| 2. Benchmark preparation | compiling benchmark constraints for a tokenizer, scoring stored outputs (`experiments/benchmarks/*_compile.py`, `*_constraints.py`, `*_score.py`, `bfcl_eval_outputs.py`, `llada_pilot_check.py`) | level 1, a few extra packages, the datasets, the model's tokenizer file. CPU only. |
| 3. Real-model runs | LLaDA and MDLM (`experiments/benchmarks/llada_json.py`, `mdlm_*.py`) | level 2, the model weights, one GPU. |

## 1. Core (CPU)

```bash
conda create -n twistroute python=3.11
conda activate twistroute
pip install -e ".[test]"        # numpy, scipy, torch >= 2.1, pytest; the package is installed in place
```

`pip install -e .` makes `import twistroute` work from anywhere. The experiment scripts add their own folders to the path, so
they run as `python experiments/<folder>/<script>.py` from the repository root.

Versions this was last run with: Python 3.11.18, torch 2.14.1, numpy 2.4.6, scipy 1.17.1, pytest 9.2.0 (macOS, 8 cores, 16 GB).
The cluster environment used torch 2.5.1 with CUDA 12.4.

Check the installation:

```bash
pytest tests/core                          # the code that follows the notes; about 2 minutes
python experiments/notes/table3.py         # Table 3 of the notes; about 20 seconds
```

All tests of this work (481 tests in `tests/core`, `tests/algorithms`, `tests/legacy`, `tests/experiments`) take 25 to 35 minutes on
the laptop above. Cap the threads on a shared machine, otherwise torch takes every core:

```bash
OMP_NUM_THREADS=4 pytest tests/core tests/algorithms tests/legacy tests/experiments
```

Two tests compare exactly against outputs frozen on one platform and torch version (macOS arm64, torch 2.14.1), because
last-bit floating-point differences can flip a random draw: `tests/experiments/test_pu_regression.py` and the frozen-file part of
`tests/core/test_core_matches_legacy.py`. Elsewhere they skip themselves and say so. The side-by-side comparison of the new and
the earlier sampler in the same file runs everywhere.

Style of the new code (line length 100, one statement per line) is checked with `ruff format` and `ruff check`
(`pip install ruff`; settings in `pyproject.toml`, which lists the files it applies to).

## 2. Benchmark preparation (CPU)

Extra packages:

```bash
pip install transformers tokenizers interegular outlines-core pandas pyarrow jsonschema
pip install --no-deps bfcl-eval            # the official BFCL checker; --no-deps because its dependencies are not needed for scoring
```

Versions last used: transformers 5.23.0, tokenizers 0.23.3, interegular 0.3.3, outlines-core 0.2.14, pandas 3.1.1, pyarrow 26.0.1,
jsonschema 4.26.0, bfcl-eval 2026.3.23.

Datasets (Hugging Face; download once, the scripts take a local path):

| Benchmark | Dataset | Access | What the scripts expect |
|---|---|---|---|
| JSON-Mode-Eval | `NousResearch/json-mode-eval` | open | `--dataset <path>/train.parquet` |
| BFCL v3, single turn | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | open | `--data <folder>` holding `BFCL_v3_<category>.json` and `possible_answer/BFCL_v3_<category>.json` |
| xLAM | `Salesforce/xlam-function-calling-60k` | gated: accept the terms on the dataset page while logged in, then download | `--data <path>/xlam_function_calling_60k.json` |

For the gated dataset, log in with `huggingface-cli login` (a read token is enough) or download the file in the browser. Do not
put a token in the repository or in a job script.

The compile scripts need the model's tokenizer, not its weights: `--tokenizer <path>/tokenizer.json`, the file of that name in
the model repository (`GSAI-ML/LLaDA-8B-Instruct`). Their defaults for the vocabulary size, the range of ordinary tokens and the
end tokens are LLaDA's.

```bash
python experiments/benchmarks/json_mode_compile.py --tokenizer tokenizer.json --dataset train.parquet --out json_mode_routes
python experiments/benchmarks/bfcl_compile.py --tokenizer tokenizer.json --data bfcl/ --per-split 4 --out bfcl_compile.json --routes bfcl_routes
python experiments/benchmarks/xlam_constraints.py --data xlam_function_calling_60k.json --n 1000
```

Each compile writes `<name>.npz` (token classes and automata) and `<name>.json` (prompts and references); these two files are all
a GPU job needs besides the code. Compiling one BFCL example takes 6 to 9 seconds at the median and up to about 80 seconds.

## 3. Real-model runs (GPU)

| Model | Checkpoint (Hugging Face) | Notes |
|---|---|---|
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Loaded by `twistroute/models/llada.py` with `trust_remote_code=True` in bfloat16; 8 billion parameters, so about 16 GB of weights in memory. Run on one L40S (48 GB), 16 rows per model call. |
| MDLM, OpenWebText | `kuleshov-group/mdlm-owt` | Loaded by `twistroute/models/mdlm_owt.py`; its remote code imports `flash_attn`, and it uses the `gpt2` tokenizer. On torch 2.5.1 with CUDA 12.4 the prebuilt wheel `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE` works; building from source failed against a CUDA 12.9 toolkit. |

The weights download on first use into the Hugging Face cache (`HF_HOME`). On a cluster, point `HF_HOME` at storage the compute
nodes can read and download once from a login node.

What needs the GPU and what does not:

- GPU: `llada_json.py` (every stage), `mdlm_first_look.py`, `mdlm_orders_output.py`, `mdlm_order_consistency.py`, `mdlm_timing.py`.
- `llada_json.py --synthetic` and `mdlm_first_look.py --synthetic` replace the model by a small random one and run on CPU; use them
  to test a job script before submitting it.
- Everything else in the repository runs on CPU.

Sizes seen so far, for planning:

- LLaDA at 256 positions: about 1.4 seconds per model call of 16 rows on an L40S; a constrained decode of one BFCL example with
  one particle is 128 steps in Dang and Ermon's setting (blocks of 32, two positions per step).
- An MDLM timing job covering up to 256 positions and 256 particles ran out of GPU memory (12 GiB requested with 35 GiB held);
  bound the number of rows per model call.

A typical run, after the compile step of level 2:

```bash
python experiments/benchmarks/llada_json.py --routes bfcl_routes --stage compare --orders block_confident --K 1 --nrun 16 --out bfcl_run.json
python experiments/benchmarks/bfcl_eval_outputs.py --routes bfcl_routes --results bfcl_run.json
```

The cluster used for the results in `docs/parallel_unmasking_status.md` was MIT ORCD Engaging: partition `mit_normal_gpu`, one L40S,
8 cores, 64 GB, at most 6 hours per job. `docs/cluster_gpu_log.md` records the incidents there (some GPU nodes could not reach the
storage that held the environment).

## Where things are

- `README.md`: what the package does.
- `docs/refactor_map.md`: how the code follows the notes, section by section.
- `experiments/INDEX.md`: every experiment script, what it does, and where its result is recorded.
- `docs/parallel_unmasking_status.md`: the results log.
