#!/usr/bin/env python3
"""Summarize and plot the 4K/2K content-distribution diagnosis."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRICS = [
    "retention_level1", "retention_level2", "retention_level3",
    "retention_level4", "retention_level5", "var_Y", "var_U", "var_V",
    "Y_variance_share", "UV_variance_share", "tv6_D111",
    "tv6_mean_degree", "r5_E_D111", "r4_E_D111", "r3_E_D111",
    "r2_E_D111", "r1_E_D111", "r5_energy_share", "r4_energy_share",
    "r45_energy_share", "r5_over_r4",
]


def percentile(reference, value):
    clean = reference[np.isfinite(reference)]
    return 100.0 * np.searchsorted(np.sort(clean), value, side="right") / len(clean)


def robust_z(reference, value):
    median = np.nanmedian(reference)
    mad = np.nanmedian(np.abs(reference - median))
    return (value - median) / (1.4826 * mad) if mad else np.nan


def aggregate_sequence(group):
    result = {"dataset": group.dataset.iloc[0], "sequence": group.sequence.iloc[0],
              "num_blocks": len(group), "point_count": group.point_count.sum()}
    n = result["point_count"]
    for channel in "YUV":
        total = group["sum_" + channel].sum()
        total2 = group["sumsq_" + channel].sum()
        result["mean_" + channel] = total / n
        result["var_" + channel] = total2 / n - (total / n) ** 2
    var_total = sum(result["var_" + c] for c in "YUV")
    result["Y_variance_share"] = result["var_Y"] / var_total
    result["UV_variance_share"] = (result["var_U"] + result["var_V"]) / var_total
    edges = group.tv6_edge_count.sum()
    result["tv6_mean_degree"] = 2.0 * edges / n
    result["tv6_D111"] = sum(group["tv6_sse_" + c].sum() for c in "YUV") / (3 * edges)
    previous = group.point_count
    energy = {}
    for down in range(1, 6):
        current = group["N_level{}".format(down)]
        result["retention_level{}".format(down)] = current.sum() / previous.sum()
        stage = 6 - down
        denom = previous.sum()
        sse = sum(group["r{}_sse_{}".format(stage, c)].sum() for c in "YUV")
        result["r{}_E_D111".format(stage)] = sse / (3 * denom)
        energy[stage] = result["r{}_E_D111".format(stage)]
        previous = current
    total_e = sum(energy.values())
    for stage in range(1, 6):
        result["r{}_energy_share".format(stage)] = energy[stage] / total_e
    result["r45_energy_share"] = (energy[4] + energy[5]) / total_e
    result["r5_over_r4"] = energy[5] / energy[4]
    return pd.Series(result)


def read_degradation(path8i, path_owlii):
    a = pd.read_csv(path8i)
    a = a[a.point.isin(["4k", "2k"])][
        ["sequence", "point", "delta_matched_db"]]
    a.sequence = a.sequence.str.lower().map({
        "longdress": "Longdress", "loot": "Loot",
        "redandblack": "Redandblack", "soldier": "Soldier"})
    b = pd.read_csv(path_owlii)
    b = b[(b.endpoint == "Base") & b.point.str.lower().isin(["4k", "2k"])][
        ["sequence", "point", "delta_matched_db"]]
    b.sequence = b.sequence.str.lower().map({
        "basketball_player": "Basketball", "dancer": "Dancer",
        "exercise": "Exercise", "model": "Model"})
    b.point = b.point.str.lower()
    return pd.concat([a, b], ignore_index=True).pivot(
        index="sequence", columns="point", values="delta_matched_db").reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--external", required=True)
    p.add_argument("--whole")
    p.add_argument("--degradation-8i", required=True)
    p.add_argument("--degradation-owlii", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "top_metric_distributions").mkdir(exist_ok=True)
    (out / "correlation_plots").mkdir(exist_ok=True)
    train, val, ext = map(pd.read_csv, (args.train, args.val, args.external))
    if len(train) != 14098 or len(val) != 792:
        raise RuntimeError("RWTT split count mismatch")

    quantiles = train[METRICS].quantile([.05, .25, .5, .75, .95]).T
    quantiles.columns = ["p05", "p25", "median", "p75", "p95"]
    quantiles["iqr"] = quantiles.p75 - quantiles.p25
    quantiles.to_csv(out / "rwtt_content_statistics.csv", index_label="metric")
    for label, frame in (("train", train), ("val", val)):
        per_model = pd.DataFrame([
            aggregate_sequence(group).rename({"sequence": "original_model"})
            for _, group in frame.groupby("original_model", sort=True)
        ]).reset_index(drop=True)
        per_model.to_csv(out / ("rwtt_{}_per_original_model.csv".format(label)),
                         index=False)

    val_rows = []
    for metric in METRICS:
        reference = train[metric].to_numpy(float)
        values = val[metric].to_numpy(float)
        val_rows.append({
            "metric": metric, "val_median": np.nanmedian(values),
            "train_percentile_of_val_median": percentile(reference, np.nanmedian(values)),
            "val_fraction_inside_train_p05_p95": np.mean(
                (values >= np.nanpercentile(reference, 5)) &
                (values <= np.nanpercentile(reference, 95))),
        })
    pd.DataFrame(val_rows).to_csv(out / "rwtt_val_sanity.csv", index=False)

    seq = pd.DataFrame([
        aggregate_sequence(group)
        for _, group in ext.groupby(["dataset", "sequence"], sort=True)
    ]).reset_index(drop=True)
    dist_rows = []
    for (dataset, sequence), group in ext.groupby(["dataset", "sequence"]):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            dist_rows.append({"dataset": dataset, "sequence": sequence,
                              "metric": metric, "num_blocks": len(group),
                              "p25": np.nanpercentile(values, 25),
                              "median": np.nanmedian(values),
                              "p75": np.nanpercentile(values, 75),
                              "p95": np.nanpercentile(values, 95)})
    pd.DataFrame(dist_rows).to_csv(out / "external_block_distributions.csv", index=False)
    seq.to_csv(out / "external_content_statistics.csv", index=False)
    if args.whole:
        pd.read_csv(args.whole).to_csv(out / "external_whole_frame_auxiliary.csv", index=False)

    comparisons = []
    for _, row in seq.iterrows():
        for metric in METRICS:
            ref = train[metric].to_numpy(float)
            comparisons.append({
                "dataset": row.dataset, "sequence": row.sequence,
                "metric": metric, "sequence_value": row[metric],
                "rwtt_percentile": percentile(ref, row[metric]),
                "robust_z": robust_z(ref, row[metric]),
            })
    compare = pd.DataFrame(comparisons)
    compare.to_csv(out / "external_vs_rwtt_percentiles.csv", index=False)
    degradation = read_degradation(args.degradation_8i, args.degradation_owlii)
    wide = compare.pivot(index="sequence", columns="metric", values="robust_z").reset_index()
    joined = degradation.merge(wide, on="sequence", how="inner")
    if len(joined) != 8:
        raise RuntimeError("Expected eight external sequences, found {}".format(len(joined)))
    correlations = []
    for metric in METRICS:
        for point in ("4k", "2k"):
            rho, _ = spearmanr(joined[metric], joined[point])
            correlations.append({"metric": metric, "point": point,
                                 "spearman_rho": rho, "n_sequences": len(joined),
                                 "interpretation": "exploratory/descriptive only"})
    corr = pd.DataFrame(correlations)
    corr.to_csv(out / "metric_degradation_correlations.csv", index=False)

    top = (corr.assign(abs_rho=corr.spearman_rho.abs()).groupby("metric").abs_rho.max()
           .sort_values(ascending=False).head(8).index.tolist())
    heat = joined.set_index("sequence")[top + ["4k", "2k"]]
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat)), heat.index)
    fig.colorbar(image, ax=ax, label="RWTT robust z (metrics); delta dB (last columns)")
    ax.set_title("External content statistics and 4K/2K matched-rate degradation")
    fig.tight_layout(); fig.savefig(out / "distribution_heatmap.png", dpi=180); plt.close(fig)

    for metric in top[:6]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(train[metric].dropna(), bins=60, density=True, alpha=.65, label="RWTT Train blocks")
        for _, row in seq.iterrows():
            ax.axvline(row[metric], linewidth=1, label=row.sequence)
        ax.set_xlabel(metric); ax.set_ylabel("density"); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(out / "top_metric_distributions" / (metric + ".png"), dpi=160)
        plt.close(fig)
        for point in ("4k", "2k"):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(joined[metric], joined[point])
            for _, row in joined.iterrows():
                ax.annotate(row.sequence, (row[metric], row[point]), fontsize=8)
            rho = corr[(corr.metric == metric) & (corr.point == point)].spearman_rho.iloc[0]
            ax.set_xlabel(metric + " (RWTT robust z)"); ax.set_ylabel(point + " delta matched (dB)")
            ax.set_title("Exploratory Spearman rho={:.3f}".format(rho))
            fig.tight_layout(); fig.savefig(out / "correlation_plots" / (metric + "_" + point + ".png"), dpi=160)
            plt.close(fig)
    print("top metrics:", ", ".join(top))


if __name__ == "__main__":
    main()
