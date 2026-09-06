#!/usr/bin/env python3
from pathlib import Path
import csv
from collections import defaultdict


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "all_checkpoints_rd_review"
OUT.mkdir(exist_ok=True)
STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3525)
I8 = ("longdress", "loot", "redandblack", "soldier")
OWL = ("basketball_player", "dancer", "exercise", "model")
LABEL = {
    "longdress": "Longdress1300", "loot": "Loot1200",
    "redandblack": "Redandblack1550", "soldier": "Soldier0690",
    "basketball_player": "Basketball200", "dancer": "Dancer1",
    "exercise": "Exercise1", "model": "Model1",
}


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def one(path, endpoint):
    rows = [r for r in read(path) if r["endpoint"] == endpoint]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {endpoint} in {path}, got {len(rows)}")
    r = rows[0]
    return {
        "bpp": float(r["physical_bpp"]), "q": float(r["yuv_psnr_611"]),
        "physical_bits": int(r["physical_bits"]), "base_bits": int(r["base_bits"]),
        "enhancement_bits": int(r["enhancement_bits"]),
        "streams": int(r["num_base_residual_streams"]), "r5": int(r["num_native_r5_streams"]),
        "hard_diff": None if not r["hard_roundtrip_max_abs_difference"] else float(r["hard_roundtrip_max_abs_difference"]),
    }


def official_curves():
    curves = defaultdict(list)
    p = REPO / "drafts/canonical_scalable_formal_r01/8ivfb_external/author_vs_physical_reference.csv"
    for r in read(p): curves[r["sequence"]].append((float(r["local_physical_bpp"]), float(r["local_yuv611"])))
    p = REPO / "drafts/20260902_owlii_joint256_review/tables/owlii_author_r01_r09_per_sequence.csv"
    for r in read(p): curves[r["sequence"]].append((float(r["physical_bpp"]), float(r["yuv611_db"])))
    return {k: sorted(v) for k, v in curves.items()}


def interp(points, x):
    if x < points[0][0] or x > points[-1][0]: return None
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            return y0 + (y1-y0) * (x-x0) / (x1-x0)
    return points[-1][1]


def dominates(a, b):
    return a["bpp"] <= b["bpp"] and a["q"] >= b["q"] and (a["bpp"] < b["bpp"] or a["q"] > b["q"])


def load_neighbors(seq):
    p2 = REPO / "results/rescued_2k_enh_external_20260903/raw" / seq / "physical_rd.csv"
    if seq in I8:
        p8 = REPO / "drafts/20260901_multirate_review/03_raw_evidence/highrate_contract_completion/8ivfb/8k" / seq / "physical_rd.csv"
        full8 = "8k_step1763"
    else:
        p8 = REPO / "drafts/20260902_owlii_joint256_review/raw_n30/owlii/8k" / seq / "physical_rd.csv"
        full8 = "8k_stage1_step1763"
    return {
        "Base": {"2k": one(p2, "Canonical_Base"), "8k": one(p8, "Canonical_Base")},
        "Full": {"2k": one(p2, "step1500"), "8k": one(p8, full8)},
    }


def fmt(x, digits=4, signed=False):
    if x is None: return "N/A"
    return f"{x:{'+' if signed else ''}.{digits}f}"


def main():
    curves = official_curves()
    masters = []
    paired = {}
    candidates = defaultdict(dict)
    neighbors = {}
    for seq in I8 + OWL:
        neighbors[seq] = load_neighbors(seq)
        p = ROOT / "raw" / seq / "physical_rd.csv"
        for step in STEPS:
            b = one(p, f"step{step}_Base"); f = one(p, f"step{step}_Full")
            paired[(seq, step)] = (b, f)
            candidates[(seq, "Base")][step] = b
            candidates[(seq, "Full")][step] = f

    for seq in I8 + OWL:
        dataset = "8iVFB" if seq in I8 else "Owlii"
        for step in STEPS:
            b, f = paired[(seq, step)]
            bit_identity = b["base_bits"] == f["base_bits"] and f["physical_bits"] == f["base_bits"] + f["enhancement_bits"]
            for ep, x in (("Base", b), ("Full", f)):
                n2, n8 = neighbors[seq][ep]["2k"], neighbors[seq][ep]["8k"]
                official = interp(curves[seq], x["bpp"])
                in_range = min(n2["bpp"], n8["bpp"]) <= x["bpp"] <= max(n2["bpp"], n8["bpp"])
                chord = None
                if in_range:
                    chord = n2["q"] + (n8["q"]-n2["q"]) * (x["bpp"]-n2["bpp"]) / (n8["bpp"]-n2["bpp"])
                other_dom = any(dominates(y, x) for s, y in candidates[(seq, ep)].items() if s != step)
                correctness = x["streams"] == 4 and x["r5"] == 0 and x["physical_bits"] == x["base_bits"] + x["enhancement_bits"]
                if ep == "Full": correctness = correctness and bit_identity and x["hard_diff"] == 0
                else: correctness = correctness and x["physical_bits"] == x["base_bits"]
                delta = None if official is None else x["q"] - official
                masters.append({
                    "checkpoint": step, "dataset": dataset, "sequence": seq, "endpoint": ep,
                    "physical_bpp": x["bpp"], "YUV611": x["q"],
                    "official_interp_YUV611": official, "delta_to_official_curve": delta,
                    "neighbor_2k_bpp": n2["bpp"], "neighbor_2k_YUV611": n2["q"],
                    "neighbor_8k_bpp": n8["bpp"], "neighbor_8k_YUV611": n8["q"],
                    "rate_between_2k_8k": in_range,
                    "rate_overshoot_vs_8k_abs": x["bpp"]-n8["bpp"],
                    "rate_overshoot_vs_8k_pct": 100*(x["bpp"]/n8["bpp"]-1),
                    "local_2k8k_RD_residual": None if chord is None else x["q"]-chord,
                    "dominated_by_2k": dominates(n2, x), "dominated_by_8k": dominates(n8, x),
                    "dominated_by_other_4k_checkpoint": other_dom,
                    "enhancement_bpp": None if ep == "Base" else f["bpp"]-b["bpp"],
                    "full_minus_base_dB": None if ep == "Base" else f["q"]-b["q"],
                    "enhancement_RD_gain": None if ep == "Base" else (None if delta is None or interp(curves[seq], b["bpp"]) is None else delta-(b["q"]-interp(curves[seq], b["bpp"]))),
                    "correctness_pass": correctness,
                })

    fields = list(masters[0])
    with (OUT / "joint_4k_all_checkpoints_rd_review.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(masters)

    summary = []
    for step in STEPS:
        z = [r for r in masters if r["checkpoint"] == step]
        def subset(ep, ds=None): return [r for r in z if r["endpoint"] == ep and (ds is None or r["dataset"] == ds)]
        b, full = subset("Base"), subset("Full")
        ovs = [r["rate_overshoot_vs_8k_pct"] for r in b]
        summary.append({
            "step": step,
            "base_rate_in_range": sum(r["rate_between_2k_8k"] for r in b),
            "full_rate_in_range": sum(r["rate_between_2k_8k"] for r in full),
            "base_delta_8i": sum(r["delta_to_official_curve"] for r in subset("Base", "8iVFB"))/4,
            "base_delta_owlii": sum(r["delta_to_official_curve"] for r in subset("Base", "Owlii"))/4,
            "full_delta_8i": sum(r["delta_to_official_curve"] for r in subset("Full", "8iVFB"))/4,
            "full_delta_owlii": sum(r["delta_to_official_curve"] for r in subset("Full", "Owlii"))/4,
            "mean_base_overshoot_pct": sum(ovs)/8, "max_base_overshoot_pct": max(ovs),
            "base_pareto_violations": sum(r["dominated_by_2k"] or r["dominated_by_8k"] or r["dominated_by_other_4k_checkpoint"] for r in b),
            "full_pareto_violations": sum(r["dominated_by_2k"] or r["dominated_by_8k"] or r["dominated_by_other_4k_checkpoint"] for r in full),
            "enhancement_rd_positive": sum((r["enhancement_RD_gain"] or 0)>0 for r in full),
            "negative_full_gain": sum(r["full_minus_base_dB"] <= 0 for r in full),
            "correctness": all(r["correctness_pass"] for r in z),
        })
    with (OUT / "joint_4k_checkpoint_summary.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)

    lines = ["# Joint 4K All-Checkpoint RD Review", "",
             "All values use existing physical hard evaluations. Official deltas use adjacent-point interpolation only; no extrapolation.", "",
             "## Seven-checkpoint summary", "",
             "| Step | Base in range | Full in range | Base Δofficial 8i / Owlii | Full Δofficial 8i / Owlii | Mean / max Base vs 8K rate | Pareto violations B/F | Enhancement RD+ | Negative Full gain |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['step']} | {s['base_rate_in_range']}/8 | {s['full_rate_in_range']}/8 | {s['base_delta_8i']:+.3f} / {s['base_delta_owlii']:+.3f} | {s['full_delta_8i']:+.3f} / {s['full_delta_owlii']:+.3f} | {s['mean_base_overshoot_pct']:+.1f}% / {s['max_base_overshoot_pct']:+.1f}% | {s['base_pareto_violations']}/{s['full_pareto_violations']} | {s['enhancement_rd_positive']}/8 | {s['negative_full_gain']}/8 |")
    lines += ["", "## Screening interpretation", "",
              "- `STRICT PASS` requires Base and Full rates inside the frozen 2K–8K brackets on all sequences, no Pareto violation, positive Enhancement RD gain on all sequences, and all correctness gates.",
              "- `USABLE CANDIDATE` may retain a Base placement warning when its RD remains competitive and it is not systematically dominated.",
              "- `REJECT` denotes clear multi-sequence Pareto domination, ineffective Full refinement, or correctness failure.", "",
              "The numerical shortlist and trade-offs are reported after inspecting this table; no checkpoint is automatically made canonical.", ""]
    (OUT / "JOINT_4K_ALL_CHECKPOINTS_RD_REVIEW.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
