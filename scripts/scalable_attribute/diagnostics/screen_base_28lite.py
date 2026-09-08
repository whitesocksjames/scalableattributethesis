#!/usr/bin/env python3
"""Rank one point's Base checkpoints after a shared RWTT-28Lite hard run."""

import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", required=True)
    parser.add_argument("--endpoint-summary", required=True)
    parser.add_argument("--author-curve", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shortlist", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate_bracket(rate, official):
    ordered = sorted(official, key=lambda row: float(row["mean_model_bpp"]))
    if rate <= float(ordered[0]["mean_model_bpp"]):
        return "at_or_below_" + ordered[0]["rate_id"]
    if rate >= float(ordered[-1]["mean_model_bpp"]):
        return "at_or_above_" + ordered[-1]["rate_id"]
    for lower, upper in zip(ordered, ordered[1:]):
        if (float(lower["mean_model_bpp"]) <= rate <=
                float(upper["mean_model_bpp"])):
            return "{}_to_{}".format(lower["rate_id"], upper["rate_id"])
    raise RuntimeError("Could not locate rate bracket")


def plot(official, candidates, path, point):
    import matplotlib.pyplot as plt

    ox = [float(row["mean_model_bpp"]) for row in official]
    oy = [float(row["mean_model_yuv_psnr_611"]) for row in official]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(ox, oy, "o-", color="#3568b8", label="Official Unicorn")
    for x, y, row in zip(ox, oy, official):
        axis.annotate(row["rate_id"], (x, y), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    for row in candidates:
        axis.scatter(
            [float(row["mean_model_bpp"])],
            [float(row["mean_model_yuv_psnr_611"])], marker="*", s=120,
            label="{} step{}".format(point, row["checkpoint_step"]))
    axis.set_xlabel("Physical attribute rate (bpp)")
    axis.set_ylabel("pc_error YUV-PSNR 6:1:1 (dB)")
    axis.set_title("Canonical Base {} on RWTT-28Lite".format(point))
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)


def main():
    args = parse_args()
    if args.max_shortlist < 1 or args.max_shortlist > 2:
        raise ValueError("max-shortlist must be 1 or 2")
    candidates = read_csv(args.endpoint_summary)
    official = read_csv(args.author_curve)
    if len(candidates) < 1:
        raise ValueError("No Base candidates")
    rates = [float(row["mean_model_bpp"]) for row in candidates]
    if max(rates) != min(rates):
        raise RuntimeError(
            "Same-point Base checkpoints changed physical rate: {}".format(rates))
    candidates.sort(
        key=lambda row: (-float(row["mean_model_yuv_psnr_611"]),
                         int(row["checkpoint_step"])))
    bracket = rate_bracket(rates[0], official)
    for rank, row in enumerate(candidates, start=1):
        row["quality_rank"] = rank
        row["author_rate_bracket"] = bracket
        row["shortlisted"] = rank <= args.max_shortlist
    shortlist = candidates[:args.max_shortlist]
    promoted = []
    for row in shortlist:
        promoted.append({
            **row,
            "role": "candidate",
            "candidate_base_checkpoint": row.get("checkpoint_path"),
            "canonical_selected": False,
            "requires_manager_review_for_full28_or_enhancement": True,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = "BASE_{}_28LITE".format(args.point.upper())
    csv_path = os.path.join(args.output_dir, prefix + "_RESULTS.csv")
    json_path = os.path.join(args.output_dir, prefix + "_SHORTLIST.json")
    md_path = os.path.join(args.output_dir, prefix + "_SHORTLIST.md")
    paths = [csv_path, json_path, md_path]
    plot_path = os.path.join(args.output_dir, prefix + "_VS_AUTHOR.png")
    if args.plot:
        paths.append(plot_path)
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    write_csv(csv_path, candidates)
    with open(json_path, "x", encoding="utf-8") as handle:
        json.dump({
            "point": args.point,
            "primary_benchmark": "RWTT-28Lite",
            "all_candidates_share_exact_physical_rate": True,
            "author_rate_bracket": bracket,
            "selection_rule": (
                "rank by RWTT-28Lite YUV611; retain the configured top {} "
                "candidate(s) for 8i tie-break".format(args.max_shortlist)),
            "input_role": "diagnostic",
            "result_role": "candidate",
            "automatic_canonical_promotion": False,
            "shortlist": promoted,
        }, handle, indent=2)
        handle.write("\n")
    with open(md_path, "x", encoding="utf-8") as handle:
        handle.write("# Base {} RWTT-28Lite shortlist\n\n".format(args.point))
        handle.write("All checkpoints share physical rate `{:.9f}` bpp; rate region `{}`.\n\n".format(
            rates[0], bracket))
        handle.write("| Rank | Candidate | Step | Y | U | V | YUV611 |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for row in candidates:
            handle.write(
                "| {quality_rank} | {candidate} | {checkpoint_step} | "
                "{mean_model_y_psnr} | {mean_model_u_psnr} | "
                "{mean_model_v_psnr} | {mean_model_yuv_psnr_611} |\n"
                .format(**row))
        handle.write("\nShortlisted for the 8i fixed-4 tie-break: {}.\n".format(
            ", ".join(row["candidate"] for row in shortlist)))
    if args.plot:
        plot(official, candidates, plot_path, args.point)


if __name__ == "__main__":
    main()
