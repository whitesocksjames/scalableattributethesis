#!/usr/bin/env python3
from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "selected_step3000_review"
OUT.mkdir(exist_ok=True)

I8 = ["longdress", "loot", "redandblack", "soldier"]
OWL = ["basketball_player", "dancer", "exercise", "model"]
LABEL = {
    "longdress": "Longdress 1300", "loot": "Loot 1200",
    "redandblack": "Redandblack 1550", "soldier": "Soldier 0690",
    "basketball_player": "Basketball 0200", "dancer": "Dancer 0001",
    "exercise": "Exercise 0001", "model": "Model 0001",
}


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def endpoint(path, name):
    found = [x for x in rows(path) if x["endpoint"] == name]
    assert len(found) == 1, (path, name, len(found))
    x = found[0]
    return {
        "bpp": float(x["physical_bpp"]), "q": float(x["yuv_psnr_611"]),
        "base_bits": int(x["base_bits"]), "enh_bits": int(x["enhancement_bits"]),
        "physical_bits": int(x["physical_bits"]),
        "hard": x["hard_roundtrip_max_abs_difference"],
        "streams": int(x["num_base_residual_streams"]),
        "r5": int(x["num_native_r5_streams"]),
    }


def references():
    result = {}
    p = REPO / "drafts/canonical_scalable_formal_r01/8ivfb_external/author_vs_physical_reference.csv"
    for x in rows(p):
        result.setdefault(x["sequence"], []).append((x["rate_id"], float(x["local_physical_bpp"]), float(x["local_yuv611"])))
    p = REPO / "drafts/20260902_owlii_joint256_review/tables/owlii_author_r01_r09_per_sequence.csv"
    for x in rows(p):
        result.setdefault(x["sequence"], []).append((x["rate_id"], float(x["physical_bpp"]), float(x["yuv611_db"])))
    for seq in result:
        result[seq].sort(key=lambda x: x[1])
    return result


def load():
    all_rows = []
    for seq in I8 + OWL:
        p4 = ROOT / "raw" / seq / "physical_rd.csv"
        p2 = REPO / "results/rescued_2k_enh_external_20260903/raw" / seq / "physical_rd.csv"
        if seq in I8:
            p8 = REPO / "drafts/20260901_multirate_review/03_raw_evidence/highrate_contract_completion/8ivfb/8k" / seq / "physical_rd.csv"
            n8f = "8k_step1763"
        else:
            p8 = REPO / "drafts/20260902_owlii_joint256_review/raw_n30/owlii/8k" / seq / "physical_rd.csv"
            n8f = "8k_stage1_step1763"
        points = {
            "8K Base": endpoint(p8, "Canonical_Base"),
            "8K Full": endpoint(p8, n8f),
            "4K Base": endpoint(p4, "step3000_Base"),
            "4K Full": endpoint(p4, "step3000_Full"),
            "2K Base": endpoint(p2, "Canonical_Base"),
            "2K Full": endpoint(p2, "step1500"),
        }
        for name, value in points.items():
            all_rows.append({"dataset": "8iVFB" if seq in I8 else "Owlii", "sequence": seq,
                             "endpoint": name, **value})
    return all_rows


def plot(seq, rr, ref):
    d = {x["endpoint"]: x for x in rr if x["sequence"] == seq}
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    xs = [x[1] for x in ref[seq]]; ys = [x[2] for x in ref[seq]]
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=1.5, ms=4, label="Official R01–R09")
    for rid, x, y in ref[seq]:
        ax.annotate(rid, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    colors = {"8K": "#9467bd", "4K": "#d62728", "2K": "#2ca02c"}
    for rate in ("8K", "4K", "2K"):
        b, f = d[f"{rate} Base"], d[f"{rate} Full"]
        ax.plot([b["bpp"], f["bpp"]], [b["q"], f["q"]], "--", color=colors[rate], lw=1.2)
        ax.scatter(b["bpp"], b["q"], marker="s", s=65, color=colors[rate], label=f"{rate} Base")
        ax.scatter(f["bpp"], f["q"], marker="^", s=70, color=colors[rate], label=f"{rate} Full")
    focus = [d[x] for x in d]
    xmin, xmax = min(x["bpp"] for x in focus), max(x["bpp"] for x in focus)
    ymin, ymax = min(x["q"] for x in focus), max(x["q"] for x in focus)
    ax.set_xlim(max(0, xmin - .14 * (xmax-xmin)), xmax + .14 * (xmax-xmin))
    ax.set_ylim(ymin - .12 * (ymax-ymin), ymax + .12 * (ymax-ymin))
    ax.set_title(f"{LABEL[seq]} — selected scalable endpoints")
    ax.set_xlabel("Physical attribute bpp"); ax.set_ylabel("YUV PSNR 6:1:1 (dB)")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{seq}_selected_8k_4k_2k_rd.png", dpi=200)
    fig.savefig(OUT / f"{seq}_selected_8k_4k_2k_rd.svg")
    plt.close(fig)


def main():
    rr = load(); ref = references()
    with (OUT / "selected_8k_4k_2k_endpoints.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    for seq in I8 + OWL: plot(seq, rr, ref)

    lines = ["# Selected 4K Joint Step3000 Review", "", "## Endpoint ladder and scalability", "",
             "| Dataset | Sequence | 4K Base between 2K/8K | 4K Full between 2K/8K | Δbpp Full−Base | ΔdB Full−Base | Scalable |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for seq in I8 + OWL:
        d = {x["endpoint"]: x for x in rr if x["sequence"] == seq}
        bb = d["2K Base"]["bpp"] < d["4K Base"]["bpp"] < d["8K Base"]["bpp"] and d["2K Base"]["q"] < d["4K Base"]["q"] < d["8K Base"]["q"]
        ff = d["2K Full"]["bpp"] < d["4K Full"]["bpp"] < d["8K Full"]["bpp"] and d["2K Full"]["q"] < d["4K Full"]["q"] < d["8K Full"]["q"]
        dbpp = d["4K Full"]["bpp"] - d["4K Base"]["bpp"]
        dq = d["4K Full"]["q"] - d["4K Base"]["q"]
        bitok = d["4K Full"]["physical_bits"] == d["4K Full"]["base_bits"] + d["4K Full"]["enh_bits"] and d["4K Full"]["base_bits"] == d["4K Base"]["base_bits"]
        exact = float(d["4K Full"]["hard"]) == 0 and d["4K Full"]["streams"] == 4 and d["4K Full"]["r5"] == 0
        scalable = bitok and exact and dbpp > 0 and dq > 0
        lines.append(f"| {d['4K Base']['dataset']} | {LABEL[seq]} | {'PASS' if bb else 'FAIL'} | {'PASS' if ff else 'FAIL'} | {dbpp:.6f} | {dq:+.4f} | {'PASS' if scalable else 'WARNING'} |")
    lines += ["", "Scalable legality requires exact hard decoding, four Base residual streams, no native r5, identical Base prefix bits in Base/Full, and `Full_bits = Base_bits + Enhancement_bits`. A positive quality increment is reported separately as RD usefulness.", ""]
    (OUT / "SELECTED_4K_STEP3000_REVIEW.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
