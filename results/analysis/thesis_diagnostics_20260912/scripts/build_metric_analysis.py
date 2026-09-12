#!/usr/bin/env python3
"""Build a private, presentation-only Base/Full effectiveness analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PACKAGE = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parents[4]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
RAW = (
    ROOT
    / "results/comparisons/formal_static_rgb_final_20260911/evidence"
    / "final_raw_rd_points.csv"
)
POINT_ORDER = ["512", "1k", "2k", "4k", "8k", "16k", "32k"]
METRICS = ["Y", "U", "V", "YUV611"]
EXACT_TOL = 1e-12
NEAR_ZERO_DB = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(delta: float) -> str:
    if abs(delta) <= EXACT_TOL:
        return "equal_exact_tolerance"
    if abs(delta) <= NEAR_ZERO_DB:
        return "near_zero_positive" if delta > 0 else "near_zero_negative"
    return "positive" if delta > 0 else "negative"


def summarize(group: pd.DataFrame) -> pd.Series:
    result: dict[str, float | int] = {"pairs": len(group)}
    for column in ["delta_bpp", *[f"delta_{metric}" for metric in METRICS]]:
        values = group[column]
        result[f"{column}_mean"] = values.mean()
        result[f"{column}_median"] = values.median()
        result[f"{column}_min"] = values.min()
        result[f"{column}_max"] = values.max()
    for metric in METRICS:
        values = group[f"delta_{metric}"]
        result[f"{metric}_positive"] = int((values > EXACT_TOL).sum())
        result[f"{metric}_equal"] = int((values.abs() <= EXACT_TOL).sum())
        result[f"{metric}_negative"] = int((values < -EXACT_TOL).sum())
        result[f"{metric}_near_zero"] = int((values.abs() <= NEAR_ZERO_DB).sum())
    return pd.Series(result)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW)
    ours = raw.loc[raw["method"].eq("Ours")].copy()
    key = ["dataset", "sample", "operating_point"]
    base = ours.loc[ours["endpoint"].eq("Ours Base")].set_index(key).sort_index()
    full = ours.loc[ours["endpoint"].eq("Ours Full")].set_index(key).sort_index()
    if len(base) != 140 or len(full) != 140 or not base.index.equals(full.index):
        missing_base = full.index.difference(base.index).tolist()
        missing_full = base.index.difference(full.index).tolist()
        raise RuntimeError(
            f"Base/Full pairing failed: base={len(base)} full={len(full)} "
            f"missing_base={missing_base} missing_full={missing_full}"
        )

    rows = []
    for identity in base.index:
        b = base.loc[identity]
        f = full.loc[identity]
        row = dict(zip(key, identity))
        row.update(
            base_bits=int(b["physical_bits"]),
            full_bits=int(f["physical_bits"]),
            enhancement_bits=int(f["physical_bits"] - b["physical_bits"]),
            base_bpp=b["physical_bpp"],
            full_bpp=f["physical_bpp"],
            delta_bpp=f["physical_bpp"] - b["physical_bpp"],
            base_reconstruction_sha256=b["reconstruction_sha256"],
            full_reconstruction_sha256=f["reconstruction_sha256"],
        )
        if row["enhancement_bits"] < 0 or row["delta_bpp"] < -EXACT_TOL:
            raise RuntimeError(f"Negative enhancement rate for {identity}")
        for metric in METRICS:
            delta = f[metric] - b[metric]
            row[f"base_{metric}"] = b[metric]
            row[f"full_{metric}"] = f[metric]
            row[f"delta_{metric}"] = delta
            row[f"class_{metric}"] = classify(delta)
        rows.append(row)

    pairs = pd.DataFrame(rows)
    pairs["operating_point"] = pd.Categorical(
        pairs["operating_point"], POINT_ORDER, ordered=True
    )
    pairs = pairs.sort_values(["dataset", "sample", "operating_point"])
    pairs.to_csv(TABLES / "enhancement_effectiveness_all_pairs.csv", index=False)

    pairs.groupby("dataset", observed=True).apply(
        summarize, include_groups=False
    ).to_csv(TABLES / "enhancement_summary_by_dataset.csv")
    pairs.groupby("operating_point", observed=True, sort=True).apply(
        summarize, include_groups=False
    ).to_csv(TABLES / "enhancement_summary_by_point.csv")

    negatives = []
    for _, row in pairs.iterrows():
        for metric in METRICS:
            delta = row[f"delta_{metric}"]
            if delta <= EXACT_TOL:
                negatives.append(
                    {
                        "dataset": row["dataset"],
                        "sample": row["sample"],
                        "operating_point": row["operating_point"],
                        "metric": metric,
                        "delta_db": delta,
                        "classification": classify(delta),
                        "enhancement_bits": row["enhancement_bits"],
                        "delta_bpp": row["delta_bpp"],
                        "delta_Y": row["delta_Y"],
                        "delta_U": row["delta_U"],
                        "delta_V": row["delta_V"],
                        "delta_YUV611": row["delta_YUV611"],
                    }
                )
    not_better = pd.DataFrame(negatives).sort_values(
        ["metric", "delta_db", "dataset", "sample"]
    )
    not_better.to_csv(TABLES / "full_not_better_than_base.csv", index=False)

    yuv_negative = pairs.loc[pairs["delta_YUV611"] < -EXACT_TOL].copy()
    controls = []
    for _, negative in yuv_negative.iterrows():
        candidates = pairs.loc[
            pairs["dataset"].eq(negative["dataset"])
            & pairs["operating_point"].eq(negative["operating_point"])
            & (pairs["delta_YUV611"] > EXACT_TOL)
        ].copy()
        if candidates.empty:
            control = None
        else:
            candidates["distance"] = (
                candidates["delta_YUV611"] - abs(negative["delta_YUV611"])
            ).abs()
            control = candidates.sort_values(["distance", "sample"]).iloc[0]
        controls.append(
            {
                "negative_dataset": negative["dataset"],
                "negative_sample": negative["sample"],
                "operating_point": negative["operating_point"],
                "negative_delta_YUV611": negative["delta_YUV611"],
                "negative_delta_bpp": negative["delta_bpp"],
                "control_sample": None if control is None else control["sample"],
                "control_delta_YUV611": None
                if control is None
                else control["delta_YUV611"],
                "control_delta_bpp": None if control is None else control["delta_bpp"],
            }
        )
    pd.DataFrame(controls).to_csv(TABLES / "negative_case_matched_controls.csv", index=False)

    q3 = yuv_negative[
        [
            "dataset",
            "sample",
            "operating_point",
            "enhancement_bits",
            "delta_bpp",
            "delta_Y",
            "delta_U",
            "delta_V",
            "delta_YUV611",
        ]
    ].copy()
    q3["most_negative_channel"] = q3[["delta_Y", "delta_U", "delta_V"]].idxmin(axis=1)
    colors = {"8iVFB": "#1f77b4", "Owlii": "#ff7f0e", "CTC": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for dataset, group in pairs.groupby("dataset", observed=True):
        ax.scatter(
            group["delta_bpp"], group["delta_YUV611"], s=24, alpha=0.78,
            label=dataset, color=colors[dataset]
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Enhancement rate, Δbpp")
    ax.set_ylabel("Enhancement quality, ΔYUV611 (dB)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("svg",):
        fig.savefig(FIGURES / f"delta_yuv611_vs_delta_bpp.{suffix}")
    plt.close(fig)

    point_data = [
        pairs.loc[pairs["operating_point"].eq(point), "delta_YUV611"].to_numpy()
        for point in POINT_ORDER
    ]
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.boxplot(point_data, tick_labels=POINT_ORDER, showfliers=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Operating point")
    ax.set_ylabel("ΔYUV611 (dB)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("svg",):
        fig.savefig(FIGURES / f"delta_yuv611_by_operating_point.{suffix}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.3 * len(q3))))
    labels = [
        f"{row.dataset}/{row.sample}/{row.operating_point}"
        for row in q3.itertuples()
    ]
    positions = range(len(q3))
    width = 0.25
    for offset, metric, color in [
        (-width, "delta_Y", "#4c78a8"),
        (0, "delta_U", "#f58518"),
        (width, "delta_V", "#54a24b"),
    ]:
        ax.barh(
            [position + offset for position in positions], q3[metric],
            height=width, label=metric.removeprefix("delta_"), color=color
        )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Full − Base PSNR (dB)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("svg",):
        fig.savefig(FIGURES / f"negative_yuv611_channel_deltas.{suffix}")
    plt.close(fig)

    counts = {
        metric: {
            "positive": int((pairs[f"delta_{metric}"] > EXACT_TOL).sum()),
            "equal": int((pairs[f"delta_{metric}"].abs() <= EXACT_TOL).sum()),
            "negative": int((pairs[f"delta_{metric}"] < -EXACT_TOL).sum()),
            "near_zero_abs_le_0_01_db": int(
                (pairs[f"delta_{metric}"].abs() <= NEAR_ZERO_DB).sum()
            ),
        }
        for metric in METRICS
    }
    evidence = {
        "source": str(RAW.relative_to(ROOT)),
        "source_sha256": sha256(RAW),
        "pairs": len(pairs),
        "exact_tolerance": EXACT_TOL,
        "near_zero_db": NEAR_ZERO_DB,
        "counts": counts,
        "frozen_package_modified": False,
    }
    (PACKAGE / "metric_analysis_manifest.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
