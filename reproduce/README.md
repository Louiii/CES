# Reproducing figures and results

Run from the repository root, with the `report` extra installed:

```sh
uv run python reproduce/figures/build_method_scaling.py
uv run python reproduce/figures/build_rl_figure.py
uv run python reproduce/figures/build_dataset_distributions.py
uv run python reproduce/figures/build_dataset_type.py
uv run python reproduce/figures/build_position_weight.py
uv run python reproduce/figures/build_rho_and_temperature.py
uv run python reproduce/figures/build_policy_heatmaps.py
uv run python reproduce/figures/build_continuation.py
uv run python reproduce/figures/build_cache_cost_table.py
```

These commands use frozen inputs bundled in this code repository under `reproduce/inputs/` and write PDFs/PNGs under `reproduce/output/`. They do not require the unpublished full results archive. The RL command renders the six aggregate panels. The source-text example panel is excluded from this release; `--examples /path/to/selection.json` can add it from an independently obtained local selection (5 improved, 3 neutral, 4 worsened examples). No CoEdIT source sentences or model-output copies are bundled as figure inputs.

Figure 1 additionally needs system LaTeX (`latexmk`, TikZ, Latin Modern, Lato), Lato fonts and Poppler: `uv run python reproduce/figures/build_ces_figure.py`. Do not pass `--install`: that switch is retained only for manuscript integration and is not needed here.

## Result map

The paths below identify the experiment evidence used for the paper. `results/` and `analysis/` collections are not included in the pinned public Hub release. Use the bundled `reproduce/inputs/` aggregates for figure generation; the table is not a list of currently downloadable result directories. Public evaluation data are available directly where distributable, or through the CoEdIT reconstruction workflow described in [Data preparation and training](../docs/DATA_AND_TRAINING.md).

| Claim | Artifact inputs | Evaluator / analysis |
|---|---|---|
| Method scaling and controlled repair | `results/methods-sweep/`, `data/evaluations/controlled-edit-val-v1/` | `ces evaluate controlled`, `text`, `levenshtein`; `build_method_scaling.py` |
| CoEdIT transfer | `results/methods-sweep/`, `data/evaluations/coedit-gec-transfer-v1/` | `ces evaluate gec`; `ces report gec` |
| Clean AR references | `results/clean-ar/`, historical `adapted-ces/*/initial-clean-lm/` | `ces evaluate base`; `inputs/paper/evidence/clean_ar_reference_runs.csv` |
| Selective and unfiltered confirmation | `results/rl-posttraining/ces/sealed-confirmation/`, `unfiltered-confirmation/` | `ces evaluate ces-gec`; `build_rl_figure.py` |
| Source supervision / position weight / corruption exposure | retained sweep results and `inputs/paper/evidence/` | `build_dataset_type.py`, `build_position_weight.py`, `build_rho_and_temperature.py` |
| Oracle program likelihood | `results/` oracle collections | `ces evaluate oracle`, `oracle-gec` |
| Natural-text clean-tail likelihood | `results/post-edit-tail-likelihood/` | `ces evaluate tail`; `ces report tail`; `build_continuation.py` |
| Cache and editing latency | `results/system-cost/` | `ces evaluate latency`; `build_cache_cost_table.py` |

Method error bars are sample standard deviations across paired model/data seeds, not standard errors. The frozen audited metric inputs include the iterative-editor character-similarity correction. Invalid/rejected predictions use the source for unconditional metrics. The complete confirmation judge inputs and per-example decisions are not distributed in this artifact release. The bundled aggregate figure inputs support offline plotting; rebuilding those figures does not call an API or purchase judging.

## Limits

- Bundled figure aggregates cover three-seed results, but method-grid weights are retained for model/data seed 42/1234. Final RL has three post-training seeds per depth, sharing the pretrained lineage.
- The d24 clean reference is the historical 4,675-update large-batch run. New d12/d18 references have matched update counts; do not describe all depths as optimizer-matched.
- Historical adapted-CES initialization checkpoints are not all present in the curated bundle. Inference from final weights is supported; retraining that lineage requires its original initialization or rerunning clean pretraining.
- Intermediate balanced-SFT/broad-RL weights and optimizer states are excluded. Their exact recovered recipes are included; rerun these stages in order.
- The public pretraining candidates omit five quarantined rows. This prevents bit-for-bit reproduction of historical data sampling. Dataset content hashes and the actual upstream shard inventory are included; the historical upstream Hub commit was not retained.
- The compute ledger contains measured completed segments, not a deduplicated total for the research project. Failed experiments, some evaluation overhead and full API accounting remain outside that ledger.
