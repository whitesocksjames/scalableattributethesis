#!/usr/bin/env python3
"""Aggregate the fixed-4 Direct-D611 screen and select at most two checkpoints."""

import argparse
import csv
import os


SEQUENCES = ("longdress", "loot", "redandblack", "soldier")
METRICS = ("physical_bpp", "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_611")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full28-root")
    parser.add_argument("--reference-full28")
    return parser.parse_args()


def read_screen(root):
    grouped = {}
    for sequence in SEQUENCES:
        path = os.path.join(root, sequence, "physical_rd.csv")
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = row["endpoint"]
                if not (label.startswith("Direct_s") or label == "TwoStage_s3525"):
                    continue
                grouped.setdefault(label, []).append(row)
    if any(len(rows) != len(SEQUENCES) for rows in grouped.values()):
        raise RuntimeError("Every trajectory endpoint must have four sequence rows")
    result = []
    for label, rows in grouped.items():
        item = {"checkpoint": label}
        for metric in METRICS:
            item[metric] = sum(float(row[metric]) for row in rows) / len(rows)
        result.append(item)
    return sorted(result, key=lambda item: item["physical_bpp"])


def pareto(items):
    result = []
    for item in items:
        dominated = any(
            other["physical_bpp"] <= item["physical_bpp"] and
            other["yuv_psnr_611"] >= item["yuv_psnr_611"] and
            (other["physical_bpp"] < item["physical_bpp"] or
             other["yuv_psnr_611"] > item["yuv_psnr_611"])
            for other in items)
        if not dominated:
            result.append(item)
    return result


def choose(items):
    direct = [item for item in items if item["checkpoint"].startswith("Direct_s")]
    frontier = pareto(direct)
    if not frontier:
        raise RuntimeError("Direct-D611 Pareto frontier is empty")
    best_quality = max(frontier, key=lambda item: item["yuv_psnr_611"])
    near_best = [item for item in frontier
                 if item["yuv_psnr_611"] >= best_quality["yuv_psnr_611"] - 0.05]
    primary = min(near_best, key=lambda item: item["physical_bpp"])
    selected = [primary]
    if primary is not best_quality and (
            primary["physical_bpp"] <= best_quality["physical_bpp"] - 0.005):
        selected.append(best_quality)
    return frontier, selected


def write_csv(path, items, frontier, selected):
    frontier_names = {item["checkpoint"] for item in frontier}
    selected_names = {item["checkpoint"] for item in selected}
    fields = ("checkpoint",) + METRICS + ("pareto", "shortlisted")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({**item,
                             "pareto": item["checkpoint"] in frontier_names,
                             "shortlisted": item["checkpoint"] in selected_names})


def full28_rows(root, selected):
    result = []
    if not root:
        return result
    for item in selected:
        label = item["checkpoint"]
        path = os.path.join(root, label, "endpoint_summary.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            row = next(row for row in csv.DictReader(handle)
                       if row["endpoint"] == "Full")
        result.append({
            "checkpoint": label,
            "bpp": float(row["mean_model_bpp"]),
            "y": float(row["mean_model_y_psnr"]),
            "yuv611": float(row["mean_model_yuv_psnr_611"]),
        })
    return result


def reference_full28(path):
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle)
                   if row["endpoint"] == "Full")
    return {
        "checkpoint": "TwoStage_s3525",
        "bpp": float(row["mean_model_bpp"]),
        "y": float(row["mean_model_y_psnr"]),
        "yuv611": float(row["mean_model_yuv_psnr_611"]),
    }


def write_report(path, items, frontier, selected, full28, reference):
    frontier_names = {item["checkpoint"] for item in frontier}
    selected_names = {item["checkpoint"] for item in selected}
    lines = [
        "# Direct-D611 matched-budget trajectory",
        "",
        "## 8i fixed-4 screening",
        "",
        "| Checkpoint | bpp | Y | U | V | YUV611 | Pareto | Full28 shortlist |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in items:
        lines.append(
            "| {checkpoint} | {physical_bpp:.6f} | {y_psnr:.4f} | "
            "{u_psnr:.4f} | {v_psnr:.4f} | {yuv_psnr_611:.4f} | {pareto} | "
            "{selected} |".format(
                **item,
                pareto="YES" if item["checkpoint"] in frontier_names else "NO",
                selected="YES" if item["checkpoint"] in selected_names else "NO"))
    lines.extend([
        "",
        "Shortlist rule: Direct-only Pareto frontier; select the lowest-rate point "
        "within 0.05 dB of the best Direct quality, and retain the best-quality "
        "point as a second candidate only when the near-best point saves at least "
        "0.005 bpp.",
        "",
        "Selected: " + ", ".join(item["checkpoint"] for item in selected),
    ])
    if full28 or reference:
        lines.extend([
            "",
            "## RWTT Full28",
            "",
            "| Checkpoint | bpp | Y | YUV611 |",
            "|---|---:|---:|---:|",
        ])
        for item in ([reference] if reference else []) + full28:
            lines.append("| {checkpoint} | {bpp:.6f} | {y:.4f} | {yuv611:.4f} |".format(
                **item))
        lines.extend([
            "",
            "This is a single 32K operating-point/trajectory comparison, not a "
            "BD-rate comparison.",
        ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_plot(path, items):
    direct = sorted(
        (item for item in items if item["checkpoint"].startswith("Direct_s")),
        key=lambda item: int(item["checkpoint"].split("_s", 1)[1]))
    reference = next(
        (item for item in items if item["checkpoint"] == "TwoStage_s3525"), None)
    all_items = direct + ([reference] if reference else [])
    xs = [item["physical_bpp"] for item in all_items]
    ys = [item["yuv_psnr_611"] for item in all_items]
    x_pad = max(max(xs) - min(xs), 0.01) * 0.12
    y_pad = max(max(ys) - min(ys), 0.05) * 0.15
    x_min, x_max = min(xs) - x_pad, max(xs) + x_pad
    y_min, y_max = min(ys) - y_pad, max(ys) + y_pad

    def point(item):
        x = 70 + 620 * (item["physical_bpp"] - x_min) / (x_max - x_min)
        y = 430 - 360 * (item["yuv_psnr_611"] - y_min) / (y_max - y_min)
        return x, y

    direct_points = " ".join(
        "{:.1f},{:.1f}".format(*point(item)) for item in direct)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="500">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<line x1="70" y1="430" x2="700" y2="430" stroke="black"/>',
        '<line x1="70" y1="430" x2="70" y2="60" stroke="black"/>',
        '<text x="385" y="485" text-anchor="middle" font-size="14">Physical bpp</text>',
        '<text x="18" y="245" transform="rotate(-90 18 245)" '
        'text-anchor="middle" font-size="14">pc_error YUV-PSNR 6:1:1 (dB)</text>',
        '<polyline points="{}" fill="none" stroke="#1f77b4" stroke-width="2"/>'.format(
            direct_points),
    ]
    for item in direct:
        x, y = point(item)
        step = item["checkpoint"].split("_s", 1)[1]
        parts.extend([
            '<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="#1f77b4"/>'.format(x, y),
            '<text x="{:.1f}" y="{:.1f}" font-size="10">{}</text>'.format(
                x + 5, y - 5, step),
        ])
    if reference:
        x, y = point(reference)
        parts.extend([
            '<circle cx="{:.1f}" cy="{:.1f}" r="7" fill="#d62728"/>'.format(x, y),
            '<text x="{:.1f}" y="{:.1f}" font-size="10">two-stage 3525</text>'.format(
                x + 8, y + 4),
        ])
    parts.extend([
        '<text x="70" y="452" font-size="10">{:.4f}</text>'.format(x_min),
        '<text x="665" y="452" font-size="10">{:.4f}</text>'.format(x_max),
        '<text x="35" y="433" font-size="10">{:.3f}</text>'.format(y_min),
        '<text x="35" y="65" font-size="10">{:.3f}</text>'.format(y_max),
        '</svg>',
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    items = read_screen(args.screen_root)
    frontier, selected = choose(items)
    write_csv(os.path.join(args.output_dir, "DIRECT_D611_TRAJECTORY.csv"),
              items, frontier, selected)
    write_plot(os.path.join(args.output_dir, "DIRECT_D611_TRAJECTORY.svg"), items)
    with open(os.path.join(args.output_dir, "shortlist.txt"), "w",
              encoding="utf-8") as handle:
        for item in selected:
            step = item["checkpoint"].split("_s", 1)[1]
            checkpoint = (args.initial_checkpoint if step == "1762" else
                          os.path.join(
                              args.checkpoint_root, "step_{}.pth".format(step)))
            handle.write("{}={}\n".format(
                item["checkpoint"], checkpoint))
    full28 = full28_rows(args.full28_root, selected)
    reference = reference_full28(args.reference_full28)
    write_report(
        os.path.join(args.output_dir, "DIRECT_D611_MATCHED_BUDGET_ANALYSIS.md"),
        items, frontier, selected, full28, reference)
    write_report(
        os.path.join(args.output_dir, "DIRECT_D611_8I_SCREEN.md"),
        items, frontier, selected, [], None)
    print("\n".join(item["checkpoint"] for item in selected))


if __name__ == "__main__":
    main()
