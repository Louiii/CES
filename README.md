# Causal Edit Serialization (CES)

[![arXiv](https://img.shields.io/badge/arXiv-2610.22820-b31b1b.svg)](https://arxiv.org/abs/2610.22820)
[![Hugging Face: Data and checkpoints](https://img.shields.io/badge/Hugging%20Face-Data%20%26%20checkpoints-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/lorob/CES)

CES represents text edits as causal programs with a localization head and dual-index positional encoding. This repository contains the checkpoint-compatible implementation, training and evaluation workflows, and figure-building code for the CES paper.

The [lorob/CES artifact repository](https://huggingface.co/datasets/lorob/CES) contains 66 checkpoints, the canonical tokenizer, public data components and CoEdIT reconstruction recipes. Checkpoints use this implementation, not Transformers `AutoModel`. Download individual components rather than the whole repository. Detailed evaluation records and the full results archive are not part of this artifact release.

## Install

From this code checkout, install [uv](https://docs.astral.sh/uv/) and use Python 3.10 or newer:

```sh
uv sync --locked
uv run ces --help
```

This installs the inference dependencies. For training, evaluation and figure generation, install the corresponding extras:

```sh
uv sync --locked --extra train --extra eval --extra report --extra dev
```

CPU inference is supported; the reported training runs require GPUs. NanoChat's MIT attribution is preserved in `LICENSE`; source provenance is in `provenance.json`.

## Download one checkpoint and edit

The revision below pins the published data, checkpoints and tokenizer. Run these commands from the code repository root:

```sh
export CES_REVISION='6e0145fc0e277bc7ad7d72a670d679dd61224367'
uv run ces download --revision "$CES_REVISION" --destination ./artifacts \
  --include 'tokenizers/ces32k_v1/*' \
  --include 'checkpoints/rl-posttraining/ces/d12/posttrain-seed-42/*' \
  --include checkpoints/models.json \
  --include SHA256SUMS \
  --include LICENSES.md --include NOTICE.md --include 'licenses/*'
uv run ces verify ./artifacts --include 'tokenizers/ces32k_v1/*'
uv run ces verify ./artifacts \
  --include 'checkpoints/rl-posttraining/ces/d12/posttrain-seed-42/*'
uv run ces edit \
  --checkpoint ./artifacts/checkpoints/rl-posttraining/ces/d12/posttrain-seed-42 \
  --tokenizer ./artifacts/tokenizers/ces32k_v1 \
  --text 'She went to store.' --device cpu
```

For this example, the d12 checkpoint inserts the missing article:

```text
She went to store. → She went to the store.
```

The command returns JSON containing the corrected `text`, `status: "valid"`, and the generated token/address program.

Use `--device cuda` for GPU inference. The d12 checkpoint is a smaller demonstration model; the main confirmation result uses d24. To download that model, replace `d12` with `d24` in the checkpoint paths above. Other methods, stages and seeds are listed in `checkpoints/models.json` and [model names and training lineage](docs/MODELS.md).

`edit` generates a greedy token/address program and reports execution status. It does not apply the paper's selective confirmation policy. A valid program is not necessarily a good correction; use the evaluation workflows and recorded policies when measuring performance.

## Prepare data and train

[Data preparation and training](docs/DATA_AND_TRAINING.md) gives selective downloads, CoEdIT reconstruction, optional fine-tuning corpus serialization, and commands for pretraining and the SFT/RL curriculum. It explains the distinct d12 post-training base and the required upstream ClimbMix shards.

The CLI provides `prepare`, `train`, `evaluate`, `report`, `download`, `verify`, and `edit`. Inspect a workflow or a saved recipe without launching training:

```sh
uv run ces evaluate controlled -- --help
uv run ces train pretrain \
  --config configs/methods-sweep__seed-m42-d1234__ces__d12__pretrained.json \
  --dry-run
```

`configs/index.json` maps checkpoint paths to recipes. [Reproducing figures and results](reproduce/README.md) describes the bundled figure inputs, metric definitions and the scope of the available evidence.

## Code map and NanoChat relationship

CES builds on NanoChat's decoder, tokenizer, optimizer, checkpoint and evaluation infrastructure. Files under `nanochat/` include both adapted infrastructure and CES-specific modules; the directory name does not imply that those files are unchanged upstream code. The main additions are:

- **Model and loss:** `gpt.py` adds source-address positional encoding alongside stream positions, a localization head, and address supervision. Configuration selects standard attention, split dual RoPE, or the appended address dimensions used for adapted-CES.
- **Edit programs and decoding:** serializers represent edits as token/address streams; the CES engine constrains operation grammar and pointer choices during generation.
- **Data and training:** edit-aware loaders construct corruptions, pack examples and track token/address targets. SFT and RL workflows train the edit-program policy and its localization decisions.
- **Comparisons and evaluation:** dedicated modules implement the text-editing baselines, iterative editor, oracle likelihoods, GEC scoring and inference-cost measurements.

```text
CES/
├── nanochat/
│   ├── gpt.py                          # NanoChat-derived decoder + CES dual RoPE/head
│   ├── causal_serialisation_extended.py # Five-operation CES grammar and training targets
│   ├── causal_serialisation.py         # Two-operation format + shared sampling utilities
│   ├── engine_extended.py              # CES generation, grammar and pointer constraints
│   ├── optimized_inference.py          # Optional cached batch-one CES inference
│   ├── edit_dataloader.py              # Corruptions, example packing and training streams
│   ├── edit_dataset.py                 # Edit-candidate data access
│   ├── ces_rl.py                       # Rewards and policy log-probability replay
│   ├── rl_pair_dataloader.py           # Paired source/target training inputs
│   ├── pretrained_adapter.py           # Initialization of adapted-CES from clean AR
│   ├── tokenizer.py                    # Base tokenizer and edit-token extensions
│   ├── tokenizer_contract.py           # Frozen CES tokenizer vocabulary/hash checks
│   ├── edit_dsl.py, flat_relative_dsl.py # Autoregressive edit-language baselines
│   ├── text_edit_baselines.py, predicted_fim.py # Text and span-infilling comparisons
│   ├── levenshtein.py, levenshtein_data.py # Iterative editing baseline
│   ├── oracle_ces.py, oracle_edit_likelihood.py # Oracle program evaluation
│   └── optim.py, fp8.py, flash_attention.py, ... # NanoChat-derived training infrastructure
├── ces/
│   ├── cli.py                          # Public commands and saved-config dispatch
│   ├── inference.py                    # Checkpoint loading and text-editing interface
│   └── workflows/
│       ├── base_train.py, base_eval.py  # NanoChat-derived training/evaluation with CES modes
│       ├── ces_sft.py, ces_balanced_sft.py, ces_rl.py # CES post-training stages
│       ├── build_*.py, prepare_*.py     # Benchmark and corpus preparation
│       ├── *eval*.py, score_*.py        # Editing evaluations and metric aggregation
│       └── benchmark_*.py, ces_cache_timing.py # Inference/tokenization measurements
├── configs/                            # Training/evaluation recipes and checkpoint index
├── reproduce/                          # Figure builders and bundled aggregate inputs
├── tests/                              # Serializer, loader, model and workflow checks
└── docs/                               # Data preparation and model-lineage guides
```

Start with `causal_serialisation_extended.py` for the edit representation, `gpt.py` for the architecture, and `engine_extended.py` for decoding. Both serializer modules are required: the five-operation pipeline still imports reservoir and corruption helpers from `causal_serialisation.py`, while the two-operation loader, decoder and tokenizer retain their own types and vocabulary.

NanoChat's MIT notice is retained in [LICENSE](LICENSE). [provenance.json](provenance.json) records the research-fork source revision and the source-to-release file mapping; it is not a claim that all NanoChat-derived files match upstream NanoChat.

## Validation and reproducibility

Release checks cover installation in a fresh Python environment, download and checksum verification of the pinned public tokenizer and d12 RL checkpoint, CPU execution of the editing example, 175 passing CPU tests (16 skips), parsing of all 88 workflow configurations, a two-update CES training smoke test with token and localization loss, and wheel/source-package contents. The wheel is also tested outside the source checkout. Earlier checks established exact fixed-input CPU logits against the development implementation for all eight method families plus final RL. These checks do not constitute a rerun of the full training grid or every documented workflow.

The tests require the downloaded tokenizer from the inference example. After installing the development extras, expose it to the training/test loaders:

```sh
export NANOCHAT_BASE_DIR="$PWD/runtime"
mkdir -p "$NANOCHAT_BASE_DIR"
if [ ! -e "$NANOCHAT_BASE_DIR/tokenizer" ]; then
  ln -s "$PWD/artifacts/tokenizers/ces32k_v1" "$NANOCHAT_BASE_DIR/tokenizer"
fi
uv run pytest tests -q
```

Artifact manifests record SHA256 hashes. Public pretraining candidates omit five quarantined rows, so historical data sampling cannot be reproduced bit-for-bit. Raw ClimbMix, optimizer states and intermediate balanced-SFT/broad-RL weights are not distributed. Component licences and source-data attribution are in the artifact repository's `LICENSES.md` and `NOTICE.md`.

## Citation

If you use CES or its released artifacts, please cite the paper:

```bibtex
@misc{robinson2026causaleditserialization,
  title         = {Causal Edit Serialization},
  author        = {Louis Robinson},
  year          = {2026},
  eprint        = {2610.22820},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.22820}
}
```
