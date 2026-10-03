"""Aggregate held-out post-edit tail likelihood runs with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROGRAM_CONDITIONS = (
    "clean_noop_program",
    "corrupt_noop_program",
    "corrupt_sham_program",
    "corrupt_gold_program",
)
PROGRAM_LABELS = ("Clean + no-op", "Corrupt + no-op", "Corrupt + sham", "Corrupt + repair")
COLORS = ("#4C78A8", "#E45756", "#F2CF5B", "#54A24B")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_run(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Run must be LABEL=SHORT_LABEL=PATH")
    return parts[0], parts[1], Path(parts[2])


def ci(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values[np.isfinite(values)], [0.025, 0.975])
    return [float(low), float(high)]


def condition_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        condition: np.asarray([
            float(row["conditions"][condition]["tail_nll"])
            / int(row["conditions"][condition]["tail_tokens"])
            for row in rows
        ])
        for condition in rows[0]["conditions"]
    }


def paired_bootstrap(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    arrays = condition_arrays(rows)
    n = len(rows)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(draws, n))

    def estimate_delta(left: str, right: str) -> dict[str, Any]:
        values = arrays[left] - arrays[right]
        return {
            "estimate": float(values.mean()),
            "ci95": ci(values[indices].mean(axis=1)),
        }

    output: dict[str, Any] = {
        "examples": n,
        "conditions": {},
    }
    byte_counts = np.asarray([
        int(row["conditions"][next(iter(row["conditions"]))]["tail_bytes"]) for row in rows
    ])
    token_counts = np.asarray([
        int(row["conditions"][next(iter(row["conditions"]))]["tail_tokens"]) for row in rows
    ])
    for condition, values in arrays.items():
        total_nll = np.asarray([
            float(row["conditions"][condition]["tail_nll"]) for row in rows
        ])
        boot_mean = values[indices].mean(axis=1)
        boot_bpb = (
            total_nll[indices].sum(axis=1)
            / np.log(2.0)
            / byte_counts[indices].sum(axis=1)
        )
        output["conditions"][condition] = {
            "mean_nll": float(total_nll.sum() / token_counts.sum()),
            "mean_nll_ci95": ci(boot_mean),
            "bits_per_byte": float(total_nll.sum() / np.log(2.0) / byte_counts.sum()),
            "bits_per_byte_ci95": ci(boot_bpb),
        }
    if "corrupt_gold_program" in arrays:
        output.update({
            "corruption_damage": estimate_delta(
                "corrupt_noop_program", "clean_noop_program"
            ),
            "repair_gain_vs_noop": estimate_delta(
                "corrupt_noop_program", "corrupt_gold_program"
            ),
            "repair_gain_vs_sham": estimate_delta(
                "corrupt_sham_program", "corrupt_gold_program"
            ),
            "repair_advantage_vs_clean_noop": estimate_delta(
                "clean_noop_program", "corrupt_gold_program"
            ),
            "clean_program_overhead": estimate_delta(
                "clean_noop_program", "clean_plain"
            ),
            "corrupt_program_overhead": estimate_delta(
                "corrupt_noop_program", "corrupt_plain"
            ),
            "repair_win_rate_vs_noop": float(
                np.mean(arrays["corrupt_gold_program"] < arrays["corrupt_noop_program"])
            ),
            "repair_win_rate_vs_sham": float(
                np.mean(arrays["corrupt_gold_program"] < arrays["corrupt_sham_program"])
            ),
        })
        damage = output["corruption_damage"]["estimate"]
        gain = output["repair_gain_vs_noop"]["estimate"]
        output["aggregate_recovery_fraction"] = gain / damage if damage > 0 else None
    return output


def select_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "overall":
        return rows
    field, value = group.split(":", 1)
    if field in {"span_length", "edit_distance"}:
        numeric = int(value)
        return [row for row in rows if int(row[field]) == numeric]
    return [row for row in rows if str(row[field]) == value]


def analyse_run(
    label: str,
    short_label: str,
    path: Path,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rows = read_jsonl(path / "records.jsonl")
    groups = ["overall"]
    groups += [f"operation:{value}" for value in ("replace", "delete", "insert")]
    groups += [f"span_length:{value}" for value in (1, 4, 16)]
    groups += [f"edit_distance:{value}" for value in (16, 64, 128)]
    analysis = {
        group: paired_bootstrap(
            select_rows(rows, group), draws=draws, seed=seed + group_index
        )
        for group_index, group in enumerate(groups)
    }
    position_arrays = {
        condition: np.asarray([
            row["conditions"][condition]["tail_token_nll"] for row in rows
        ], dtype=np.float64)
        for condition in rows[0]["conditions"]
    }
    position_nll = {
        condition: values.mean(axis=0).tolist()
        for condition, values in position_arrays.items()
    }
    position_excess_smoothed: dict[str, Any] = {}
    if "clean_noop_program" in position_arrays:
        clean = position_arrays["clean_noop_program"]
        for condition in PROGRAM_CONDITIONS[1:]:
            differences = position_arrays[condition] - clean
            smoothed = np.asarray([rolling(row, 8) for row in differences])
            mean = smoothed.mean(axis=0)
            half_width = 1.96 * smoothed.std(axis=0, ddof=1) / np.sqrt(len(smoothed))
            position_excess_smoothed[condition] = {
                "positions": (np.arange(1, len(mean) + 1) + 3.5).tolist(),
                "mean": mean.tolist(),
                "ci95_low": (mean - half_width).tolist(),
                "ci95_high": (mean + half_width).tolist(),
            }
    return {
        "label": label,
        "short_label": short_label,
        "path": str(path),
        "analysis": analysis,
        "mean_token_nll_by_tail_position": position_nll,
        "smoothed_excess_nll_by_tail_position": position_excess_smoothed,
    }


def fmt_delta(metric: dict[str, Any], *, scale: float = 1000.0, digits: int = 2) -> str:
    estimate = scale * metric["estimate"]
    low, high = (scale * value for value in metric["ci95"])
    return f"{estimate:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def render_report(runs: list[dict[str, Any]], standard: dict[str, Any] | None) -> str:
    lines = [
        "# Held-out natural-text continuation after CES edit programs",
        "",
        "The evaluation uses 540 unique ClimbMix validation documents: 20 cases",
        "for every combination of three corruption operations (replace, delete,",
        "insert), three span lengths (1, 4, 16 tokens), and three distances from",
        "the clean tail (16, 64, 128 tokens). Every condition predicts the identical",
        "128-token held-out tail. Intervals are paired 95% bootstrap intervals with",
        "20,000 resamples. NLL differences are reported in millinats per token.",
        "",
        "## Overall result",
        "",
        "| CES checkpoint | Clean no-op BPB | Corrupt no-op BPB | Sham BPB | Repair BPB | Repair gain vs no-op | Repair gain vs sham | Recovery | Win vs no-op / sham |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        overall = run["analysis"]["overall"]
        conditions = overall["conditions"]
        recovery = overall["aggregate_recovery_fraction"]
        lines.append(
            f"| {run['label']} | "
            f"{conditions['clean_noop_program']['bits_per_byte']:.4f} | "
            f"{conditions['corrupt_noop_program']['bits_per_byte']:.4f} | "
            f"{conditions['corrupt_sham_program']['bits_per_byte']:.4f} | "
            f"{conditions['corrupt_gold_program']['bits_per_byte']:.4f} | "
            f"{fmt_delta(overall['repair_gain_vs_noop'])} | "
            f"{fmt_delta(overall['repair_gain_vs_sham'])} | "
            f"{100 * recovery:.1f}% | "
            f"{100 * overall['repair_win_rate_vs_noop']:.1f}% / "
            f"{100 * overall['repair_win_rate_vs_sham']:.1f}% |"
        )

    if standard is not None:
        standard_overall = standard["analysis"]["overall"]
        adapted = runs[0]["analysis"]["overall"]["conditions"]
        std_clean = standard_overall["conditions"]["clean_plain"]["bits_per_byte"]
        std_corrupt = standard_overall["conditions"]["corrupt_plain"]["bits_per_byte"]
        lines.extend([
            "",
            "## Matched standard-AR reference",
            "",
            "The adapted CES checkpoint was initialized from this exact standard AR",
            "checkpoint. This makes their clean plain-text likelihoods comparable; it",
            "does not make the standard model capable of consuming CES programs.",
            "",
            "| History/model | Tail BPB |",
            "|---|---:|",
            f"| Standard AR, clean prefix | {std_clean:.4f} |",
            f"| Standard AR, corrupt prefix | {std_corrupt:.4f} |",
            f"| Adapted CES, clean plain prefix | {adapted['clean_plain']['bits_per_byte']:.4f} |",
            f"| Adapted CES, clean no-op program | {adapted['clean_noop_program']['bits_per_byte']:.4f} |",
        ])

    lines.extend([
        "",
        "## Stratified repair gain versus corrupt no-op",
        "",
        "Positive values mean the supplied correction lowers clean-tail NLL.",
        "",
        "| Checkpoint | Stratum | N | Gain, millinats/token [95% CI] | Recovery |",
        "|---|---|---:|---:|---:|",
    ])
    group_labels = [
        ("operation:replace", "replace"),
        ("operation:delete", "delete"),
        ("operation:insert", "insert"),
        ("span_length:1", "span 1"),
        ("span_length:4", "span 4"),
        ("span_length:16", "span 16"),
        ("edit_distance:16", "distance 16"),
        ("edit_distance:64", "distance 64"),
        ("edit_distance:128", "distance 128"),
    ]
    for run in runs:
        for key, label in group_labels:
            group = run["analysis"][key]
            recovery = group["aggregate_recovery_fraction"]
            lines.append(
                f"| {run['short_label']} | {label} | {group['examples']} | "
                f"{fmt_delta(group['repair_gain_vs_noop'])} | {100 * recovery:.1f}% |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The mature from-scratch CES checkpoint shows lower tail NLL after the gold",
        "repair than after either the corrupt no-op or the valid same-family sham",
        "program. The effect is positive across every operation, span, and edit-distance",
        "aggregate. Its recovery exceeds 100% because the explicit correct program",
        "makes the relevant local content more salient than an empty program over an",
        "otherwise clean prefix; the paired NLL gains and sham comparison are therefore",
        "more stable headline statistics than the recovery ratio.",
        "",
        "The short adapted checkpoint retains almost the same clean plain-text BPB as",
        "its standard-AR source but does not improve aggregate tail NLL versus corrupt",
        "no-op. This is useful negative evidence: architecture conversion plus a short",
        "adaptation is insufficient for reliable downstream use of arbitrary programs.",
        "",
        "The validation corruption is synthetic but held out, balanced, and independent",
        "across documents. No held-out language-model corruption set was available",
        "locally, and compatible three-seed final checkpoints were not stored locally.",
        "These are the two most important follow-up extensions before treating the",
        "result as a final multi-seed natural-edit benchmark.",
        "",
    ])
    return "\n".join(lines)


def rolling(values: np.ndarray, width: int = 8) -> np.ndarray:
    if width <= 1:
        return values
    return np.convolve(values, np.ones(width) / width, mode="valid")


def plot_overview(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    x = np.arange(len(runs), dtype=float)
    width = 0.19
    for condition_index, (condition, label, color) in enumerate(
        zip(PROGRAM_CONDITIONS, PROGRAM_LABELS, COLORS)
    ):
        values = [
            1000 * (
                run["analysis"]["overall"]["conditions"][condition]["mean_nll"]
                - run["analysis"]["overall"]["conditions"]["clean_noop_program"]["mean_nll"]
            )
            for run in runs
        ]
        axes[0].bar(
            x + (condition_index - 1.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, [run["short_label"] for run in runs])
    axes[0].set_ylabel("Excess clean-tail NLL (millinats/token)")
    axes[0].set_title("Format-matched continuation likelihood")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)

    mature = runs[-1]
    group_specs = [
        ("operation:replace", "replace"),
        ("operation:delete", "delete"),
        ("operation:insert", "insert"),
        ("span_length:1", "span 1"),
        ("span_length:4", "span 4"),
        ("span_length:16", "span 16"),
        ("edit_distance:16", "dist. 16"),
        ("edit_distance:64", "dist. 64"),
        ("edit_distance:128", "dist. 128"),
    ]
    gains, low_errors, high_errors = [], [], []
    for key, _label in group_specs:
        metric = mature["analysis"][key]["repair_gain_vs_noop"]
        estimate = 1000 * metric["estimate"]
        low, high = (1000 * value for value in metric["ci95"])
        gains.append(estimate)
        low_errors.append(estimate - low)
        high_errors.append(high - estimate)
    positions = np.arange(len(group_specs))
    axes[1].barh(
        positions,
        gains,
        xerr=[low_errors, high_errors],
        color="#54A24B",
        capsize=3,
    )
    axes[1].set_yticks(positions, [label for _key, label in group_specs])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("Repair NLL gain (millinats/token)")
    axes[1].set_title(f"{mature['short_label']}: all strata")
    fig.suptitle("Held-out natural-text continuation after CES repair", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"post_edit_tail_overview.{suffix}", dpi=220)
    plt.close(fig)


def plot_positions(runs: list[dict[str, Any]], output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(runs), figsize=(11.5, 4.1), sharey=True, constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]
    for axis, run in zip(axes, runs):
        curves = run["smoothed_excess_nll_by_tail_position"]
        for condition, label, color in zip(PROGRAM_CONDITIONS[1:], PROGRAM_LABELS[1:], COLORS[1:]):
            curve = curves[condition]
            positions = np.asarray(curve["positions"])
            mean = 1000 * np.asarray(curve["mean"])
            low = 1000 * np.asarray(curve["ci95_low"])
            high = 1000 * np.asarray(curve["ci95_high"])
            axis.plot(positions, mean, label=label, color=color, linewidth=1.8)
            axis.fill_between(positions, low, high, color=color, alpha=0.13, linewidth=0)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title(run["short_label"])
        axis.set_xlabel("Tail-token distance after edit-end")
    axes[0].set_ylabel("Excess NLL, 8-token moving mean (millinats/token)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Continuation recovery by token position", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"post_edit_tail_by_position.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ces-run", action="append", required=True, type=parse_run)
    parser.add_argument("--standard-run", type=parse_run)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        analyse_run(
            label, short_label, path,
            draws=args.bootstrap_draws,
            seed=args.seed + run_index * 100,
        )
        for run_index, (label, short_label, path) in enumerate(args.ces_run)
    ]
    standard = None
    if args.standard_run is not None:
        label, short_label, path = args.standard_run
        standard = analyse_run(
            label, short_label, path,
            draws=args.bootstrap_draws,
            seed=args.seed + 10_000,
        )
    payload = {"runs": runs, "standard_run": standard}
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_report(runs, standard), encoding="utf-8"
    )
    plot_overview(runs, args.output_dir)
    plot_positions(runs, args.output_dir)


if __name__ == "__main__":
    main()
