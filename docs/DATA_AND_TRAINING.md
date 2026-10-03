# Data preparation and training

Run commands from the code repository root. Install the training and evaluation dependencies with `uv sync --locked --extra train --extra eval`. The examples use the same immutable artifact revision as the README:

```sh
export CES_REVISION='6e0145fc0e277bc7ad7d72a670d679dd61224367'
export CES_PUBLIC="$PWD/artifacts"
export CES_ARTIFACTS="$PWD/../ces-local"
export CES_RUNS="$PWD/runs"
export CES_RUNTIME="$PWD/runtime"
```

`CES_PUBLIC` holds downloaded release files. `CES_ARTIFACTS` is a separate local training tree containing reconstructed text and links to public components. Keep upstream source text and reconstructed outputs out of the public code/artifact repositories.

## Reconstruct CoEdIT inputs

Download the small reconstruction recipes and tokenizer:

```sh
uv run ces download --revision "$CES_REVISION" --destination "$CES_PUBLIC" \
  --include 'reconstruction/*' --include 'tokenizers/ces32k_v1/*' \
  --include SHA256SUMS --include LICENSES.md --include NOTICE.md \
  --include 'licenses/*'
uv run ces verify "$CES_PUBLIC" --include 'reconstruction/*'
uv run ces verify "$CES_PUBLIC" --include 'tokenizers/ces32k_v1/*'
uv run python "$CES_PUBLIC/reconstruction/reconstruct_coedit.py" \
  --download --upstream-dir "$PWD/../ces-upstream/coedit" \
  --output "$CES_ARTIFACTS"
```

Choose an empty `CES_ARTIFACTS` directory for the first reconstruction; the script refuses to overwrite an existing nonempty output. It fetches the pinned official CoEdIT files, validates hashes and reconstructs training/evaluation pairs, identity inputs, clean targets and approved corruption training fields. No generation or judge API calls are required. To use already-obtained source files, omit `--download` and put the pinned `train.jsonl` and `validation.jsonl` in the upstream directory.

The release supplies reconstruction recipes rather than complete CoEdIT source/target sentences. Read the source-use terms in the downloaded `LICENSES.md` and `NOTICE.md`; reconstruction does not change them.

## Assemble the training inputs

For pretraining and the complete d12 post-training curriculum, download these additional components. This is larger than the inference-only download:

```sh
uv run ces download --revision "$CES_REVISION" --destination "$CES_PUBLIC" \
  --include 'data/pretraining/*' \
  --include 'data/posttraining/ces/v1/balanced-sft/*' \
  --include 'checkpoints/rl-posttraining/ces/d12/pretrained/*' \
  --include SHA256SUMS
uv run ces verify "$CES_PUBLIC" --include 'data/pretraining/*'
uv run ces verify "$CES_PUBLIC" \
  --include 'data/posttraining/ces/v1/balanced-sft/*'
uv run ces verify "$CES_PUBLIC" \
  --include 'checkpoints/rl-posttraining/ces/d12/pretrained/*'

mkdir -p "$CES_RUNTIME" "$CES_RUNS" "$CES_ARTIFACTS/data/posttraining/ces/v1"
ln -s "$CES_PUBLIC/checkpoints" "$CES_ARTIFACTS/checkpoints"
ln -s "$CES_PUBLIC/data/pretraining" "$CES_ARTIFACTS/data/pretraining"
ln -s "$CES_PUBLIC/data/posttraining/ces/v1/balanced-sft" \
  "$CES_ARTIFACTS/data/posttraining/ces/v1/balanced-sft"
ln -s "$CES_PUBLIC/tokenizers/ces32k_v1" "$CES_RUNTIME/tokenizer"
export NANOCHAT_BASE_DIR="$CES_RUNTIME"
```

Create these links once. CoEdIT reconstruction already supplies the other evaluation and GEC-corruption paths expected by the recipes. Download additional checkpoint families explicitly if using their configs.

Pretraining and broad RL also require independently obtained ClimbMix source shards. Set their actual local directory, then create the runtime link:

```sh
export CES_CLIMBMIX='/path/to/upstream/climbmix/shards'
ln -s "$CES_CLIMBMIX" "$CES_RUNTIME/base_data_climbmix"
```

The release does not redistribute raw ClimbMix. Public edit candidates are not a replacement for those source shards. Preserve the canonical tokenizer; do not retrain it.

## Optional CoEdIT fine-tuning corpus

With the reconstructed manifests and runtime tokenizer configured above:

```sh
uv run ces prepare corpus \
  --train-manifest "$CES_ARTIFACTS/data/evaluations/coedit-gec-transfer-v1/train.jsonl" \
  --validation-manifest "$CES_ARTIFACTS/data/evaluations/coedit-gec-v1/development.jsonl" \
  --output-dir "$CES_ARTIFACTS/data/finetuning/coedit-gec-transfer-corpus" \
  --train-shards 8 --seed 20260727
```

This builds the tokenized shards used by the methods-grid CoEdIT fine-tuning recipes. Balanced SFT and RL use the reconstructed pairs/corruption inputs directly and do not need this serialization step.

## Run a training recipe

Preview the d12 methods-grid pretraining command:

```sh
uv run ces train pretrain \
  --config configs/methods-sweep__seed-m42-d1234__ces__d12__pretrained.json \
  --dry-run
```

Remove `--dry-run` to launch training after preparing the data and suitable GPU resources. Dry-run prints the command; it does not check that data are present or that GPU memory is sufficient. Saved batch sizes, learning rates and update counts describe the paper experiments, not a small tutorial run.

For post-training, execute these stages in order, removing `--dry-run` from each command when ready:

```sh
uv run ces train balanced-sft --config configs/posttraining/d12-balanced-sft.json --dry-run
uv run ces train rl --config configs/posttraining/d12-seed42-broad.json --dry-run
uv run ces train rl --config configs/posttraining/d12-seed42-gec.json --dry-run
```

Later stages read the previous stage's outputs under `CES_RUNS`. The d12 curriculum starts from the separately released step-74800 post-training base downloaded above, not the step-26880 methods-grid checkpoint. For d18/d24, download the respective methods-grid CES pretrained checkpoint. All three post-training seeds share the balanced-SFT seed-42 parent; run that stage once per depth before the seed-specific RL recipes.

Fine-tuning configs may create a step-zero hard-link view before training. Put the source checkpoint and output run on the same filesystem and keep outputs separate from released weights. Adapted-CES includes an internal clean-pretraining stage; running its adaptation recipe separately requires that stage through `CES_INITIAL_CHECKPOINT`, not a standalone clean-AR comparison checkpoint.

See [model lineage](MODELS.md) for stage identities and [reproduction limits](../reproduce/README.md#limits) for unavailable intermediates and differences from historical data sampling.
