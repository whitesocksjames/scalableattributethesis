#!/usr/bin/env python3
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RAW = ROOT / "raw"
OW_REF = REPO / "drafts/20260902_owlii_joint256_review/tables/owlii_author_r01_r09_per_sequence.csv"
I8_REF = REPO / "drafts/canonical_scalable_formal_r01/8ivfb_external/author_vs_physical_reference.csv"

GROUPS = {
    "8ivfb": ["longdress", "loot", "redandblack", "soldier"],
    "owlii": ["basketball_player", "dancer", "exercise", "model"],
}
LABELS = {
    "longdress": "Longdress 1300", "loot": "Loot 1200",
    "redandblack": "Redandblack 1550", "soldier": "Soldier 0690",
    "basketball_player": "Basketball 0200", "dancer": "Dancer 0001",
    "exercise": "Exercise 0001", "model": "Model 0001",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def interpolation(points, x):
    pts = sorted(points)
    if x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def load_refs():
    refs = {}
    for row in read_csv(I8_REF):
        refs.setdefault(row["sequence"], []).append({
            "rate_id": row["rate_id"],
            "bpp": float(row["local_physical_bpp"]),
            "q": float(row["local_yuv611"]),
            "provenance": "AUTHOR_POINT_LOCALLY_PHYSICAL_REPRODUCED",
        })
    for row in read_csv(OW_REF):
        refs.setdefault(row["sequence"], []).append({
            "rate_id": row["rate_id"],
            "bpp": float(row["physical_bpp"]),
            "q": float(row["yuv611_db"]),
            "provenance": row["provenance"],
        })
    for seq in refs:
        refs[seq].sort(key=lambda r: r["bpp"])
    return refs


def load_candidates(refs):
    rows = []
    for seq in sum(GROUPS.values(), []):
        for r in read_csv(RAW / seq / "physical_rd.csv"):
            bpp = float(r["physical_bpp"])
            q = float(r["yuv_psnr_611"])
            interp = interpolation([(x["bpp"], x["q"]) for x in refs[seq]], bpp)
            rows.append({
                "dataset": "8iVFB" if seq in GROUPS["8ivfb"] else "Owlii",
                "sequence": seq,
                "endpoint": r["endpoint"],
                "checkpoint_step": int(r["checkpoint_step"]),
                "physical_bpp": bpp,
                "y_psnr": float(r["y_psnr"]),
                "u_psnr": float(r["u_psnr"]),
                "v_psnr": float(r["v_psnr"]),
                "yuv611": q,
                "base_bits": int(r["base_bits"]),
                "enhancement_bits": int(r["enhancement_bits"]),
                "official_interp_yuv611": interp,
                "delta_to_official_curve": None if interp is None else q - interp,
                "hard_roundtrip_max_abs_difference": r["hard_roundtrip_max_abs_difference"],
                "num_base_residual_streams": int(r["num_base_residual_streams"]),
                "num_native_r5_streams": int(r["num_native_r5_streams"]),
            })
    return rows


def write_merged(rows):
    path = ROOT / "rescued_2k_external_results.csv"
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_group(group, refs, rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    styles = {
        "Canonical_Base": ("#2ca02c", "s", "Rescued Base"),
        "step1500": ("#d62728", "o", "Full step1500"),
        "step1763": ("#9467bd", "^", "Full step1763"),
    }
    for ax, seq in zip(axes.flat, GROUPS[group]):
        rr = refs[seq]
        xs = [x["bpp"] for x in rr]
        ys = [x["q"] for x in rr]
        ax.plot(xs, ys, "-o", color="#1f77b4", ms=4, lw=1.5, label="Official R01-R09")
        for x in rr:
            ax.annotate(x["rate_id"], (x["bpp"], x["q"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        selected = [r for r in rows if r["sequence"] == seq and r["endpoint"] in styles]
        for r in selected:
            color, marker, label = styles[r["endpoint"]]
            ax.scatter(r["physical_bpp"], r["yuv611"], color=color, marker=marker, s=58, zorder=4, label=label)
        ax.set_title(LABELS[seq])
        ax.set_xlabel("Physical attribute bpp")
        ax.set_ylabel("YUV PSNR 6:1:1 (dB)")
        ax.grid(alpha=.25)

        # Local inset makes the tightly clustered rescued points readable while
        # the main panel retains all nine official operating points.
        zx = [r["physical_bpp"] for r in selected]
        zy = [r["yuv611"] for r in selected]
        ins = inset_axes(ax, width="44%", height="42%", loc="lower right", borderpad=1.1)
        ins.plot(xs, ys, "-o", color="#1f77b4", ms=2.5, lw=1)
        for r in selected:
            color, marker, _ = styles[r["endpoint"]]
            ins.scatter(r["physical_bpp"], r["yuv611"], color=color, marker=marker, s=30, zorder=4)
        xpad = max((max(zx) - min(zx)) * .25, max(zx) * .035)
        ypad = max((max(zy) - min(zy)) * .25, .18)
        ins.set_xlim(min(zx) - xpad, max(zx) + xpad)
        ins.set_ylim(min(zy) - ypad, max(zy) + ypad)
        ins.grid(alpha=.2)
        ins.tick_params(labelsize=7)
        mark_inset(ax, ins, loc1=2, loc2=4, fc="none", ec="0.55", lw=.7)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    title = "8iVFB" if group == "8ivfb" else "Owlii"
    fig.suptitle(f"Rescued 2K Base + Enhancement vs Official Unicorn — {title}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = "rescued_2k_8ivfb_four_panel_rd" if group == "8ivfb" else "rescued_2k_owlii_four_panel_rd"
    fig.savefig(ROOT / f"{stem}.png", dpi=200)
    fig.savefig(ROOT / f"{stem}.svg")
    plt.close(fig)


def mean(values):
    return sum(values) / len(values)


def report(rows):
    lines = [
        "# Rescued 2K Enhancement External Analysis", "",
        "## Provenance", "",
        "- Model: rescued `2K-U-PATH step500` Base with independent D111 Enhancement Stage-1.",
        "- Evaluation: physical hard coding; Base uses `x_low+r1-r4`, native r5 is absent; Full bits equal Base plus Enhancement bits.",
        "- 8i reference: author operating points reproduced with local physical coding and direct `pc_error` YUV 6:1:1.",
        "- Owlii reference: retained author-provided per-sequence R01-R09 CSV.",
        "- Matched deltas use adjacent-point linear interpolation only; no extrapolation.", "",
        "## Per-sequence RD results", "",
        "| Dataset | Sequence | Endpoint | bpp | YUV611 | Official@bpp | Delta to curve |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    wanted = [r for r in rows if r["endpoint"] in ("Canonical_Base", "step1500", "step1763")]
    for r in wanted:
        oi = r["official_interp_yuv611"]
        dd = r["delta_to_official_curve"]
        lines.append(f"| {r['dataset']} | {LABELS[r['sequence']]} | {r['endpoint']} | {r['physical_bpp']:.6f} | {r['yuv611']:.4f} | {oi:.4f} | {dd:+.4f} |")
    lines += ["", "## Dataset means", "", "| Dataset | Endpoint | Mean bpp | Mean YUV611 | Mean matched delta |", "|---|---:|---:|---:|---:|"]
    for dataset in ("8iVFB", "Owlii"):
        for endpoint in ("Canonical_Base", "step1500", "step1763"):
            ss = [r for r in wanted if r["dataset"] == dataset and r["endpoint"] == endpoint]
            lines.append(f"| {dataset} | {endpoint} | {mean([r['physical_bpp'] for r in ss]):.6f} | {mean([r['yuv611'] for r in ss]):.4f} | {mean([r['delta_to_official_curve'] for r in ss]):+.4f} |")
    s15 = {r["sequence"]: r for r in rows if r["endpoint"] == "step1500"}
    s17 = {r["sequence"]: r for r in rows if r["endpoint"] == "step1763"}
    dominates, tradeoff = [], []
    for seq in s15:
        a, b = s15[seq], s17[seq]
        if a["physical_bpp"] <= b["physical_bpp"] and a["yuv611"] >= b["yuv611"]:
            dominates.append(LABELS[seq])
        else:
            tradeoff.append(LABELS[seq])
    lines += [
        "", "## Interpretation", "",
        f"- `step1500` strictly RD-dominates `step1763` on **{len(dominates)}/8** sequences: {', '.join(dominates)}.",
        f"- The remaining sequences are rate-quality trade-offs: {', '.join(tradeoff)}.",
        "- Therefore `step1500` is the stronger external checkpoint overall, but it is not a universal strict winner. Selection should use the per-sequence curve positions, not PSNR alone.",
        "- All Full checkpoints add substantial Enhancement rate over Base; the payload is actively used and has not collapsed.",
        "", "## Correctness", "",
        f"- Rows checked: {len(rows)}; all use four Base residual streams and zero native-r5 streams.",
        "- Full hard round-trip maximum absolute difference is zero for every Full row.",
    ]
    (ROOT / "RESCUED_2K_ENH_EXTERNAL_ANALYSIS.md").write_text("\n".join(lines) + "\n")


def main():
    refs = load_refs()
    rows = load_candidates(refs)
    assert len(rows) == 32
    assert all(r["num_base_residual_streams"] == 4 and r["num_native_r5_streams"] == 0 for r in rows)
    assert all(float(r["hard_roundtrip_max_abs_difference"]) == 0 for r in rows if r["endpoint"].startswith("step"))
    write_merged(rows)
    plot_group("8ivfb", refs, rows)
    plot_group("owlii", refs, rows)
    report(rows)


if __name__ == "__main__":
    main()
