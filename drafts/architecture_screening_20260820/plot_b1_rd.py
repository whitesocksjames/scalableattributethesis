#!/usr/bin/env python3
"""Dependency-free SVG plot of B1 Full28 endpoints and official RD curve."""
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "rd_plot_data"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scale(value, low, high, start, length, invert=False):
    result = start + (value - low) / (high - low) * length
    return start + length - (result - start) if invert else result


def circle(x, y, color="#222", radius=5):
    return '<circle cx="{:.1f}" cy="{:.1f}" r="{}" fill="{}"/>'.format(
        x, y, radius, color)


def diamond(x, y, color):
    points = "{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}".format(
        x, y - 7, x + 7, y, x, y + 7, x - 7, y)
    return '<polygon points="{}" fill="{}" stroke="white"/>'.format(points, color)


def panel(items, b1, x0, y0, width, height, xlim, ylim, title, zoom=False):
    left, top = x0 + 58, y0 + 38
    right, bottom = x0 + width - 18, y0 + height - 48
    pw, ph = right - left, bottom - top
    sx = lambda value: scale(value, xlim[0], xlim[1], left, pw)
    sy = lambda value: scale(value, ylim[0], ylim[1], top, ph, invert=True)
    output = [
        '<rect x="{}" y="{}" width="{}" height="{}" fill="white" stroke="#aaa"/>'.format(left, top, pw, ph),
        '<text x="{}" y="{}" text-anchor="middle" font-size="15" font-weight="bold">{}</text>'.format(x0 + width / 2, y0 + 20, title),
    ]
    for index in range(6):
        xv = xlim[0] + index * (xlim[1] - xlim[0]) / 5
        xp = sx(xv)
        output.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#ddd"/>'.format(xp, top, bottom))
        output.append('<text x="{:.1f}" y="{}" text-anchor="middle" font-size="10">{:.2f}</text>'.format(xp, bottom + 17, xv))
    for index in range(6):
        yv = ylim[0] + index * (ylim[1] - ylim[0]) / 5
        yp = sy(yv)
        output.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#ddd"/>'.format(left, yp, right, yp))
        output.append('<text x="{}" y="{:.1f}" text-anchor="end" font-size="10">{:.1f}</text>'.format(left - 7, yp + 4, yv))
    visible = [(x, y, label) for x, y, label in items if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]]
    if len(visible) > 1:
        output.append('<polyline points="{}" fill="none" stroke="#222" stroke-width="2"/>'.format(" ".join("{:.1f},{:.1f}".format(sx(x), sy(y)) for x, y, _ in visible)))
    for x, y, label in visible:
        output.append(circle(sx(x), sy(y)))
        output.append('<text x="{:.1f}" y="{:.1f}" font-size="10">{}</text>'.format(sx(x) + 5, sy(y) - 7, label))
    colors = {4: "#d62728", 8: "#1f77b4"}
    for row in b1:
        step = int(row["step"])
        bx, by = float(row["mean_model_base_bpp"]), float(row["mean_model_base_yuv_psnr_611"])
        fx, fy = float(row["mean_model_full_bpp"]), float(row["mean_model_full_yuv_psnr_611"])
        color = colors[step]
        if xlim[0] <= bx <= xlim[1] and ylim[0] <= by <= ylim[1]:
            output.append(diamond(sx(bx), sy(by), color))
            label = "step{} Base ({:.3f}, {:.3f})".format(step, bx, by)
            dy = -12 if step == 4 else 18
            output.append('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="11">{}</text>'.format(sx(bx) + 9, sy(by) + dy, color, label))
        if not zoom and xlim[0] <= fx <= xlim[1] and ylim[0] <= fy <= ylim[1]:
            output.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="1.5"/>'.format(sx(bx), sy(by), sx(fx), sy(fy), color))
            output.append(circle(sx(fx), sy(fy), color, 8))
            output.append(circle(sx(fx), sy(fy), "white", 2))
            output.append('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="10">step{} Full</text>'.format(sx(fx) + 8, sy(fy) + (14 if step == 4 else -9), color, step))
    output.append('<text x="{}" y="{}" text-anchor="middle" font-size="12">Physical bitrate (bpp)</text>'.format((left + right) / 2, y0 + height - 8))
    output.append('<text x="{}" y="{}" transform="rotate(-90 {} {})" text-anchor="middle" font-size="12">YUV-PSNR 6:1:1 (dB)</text>'.format(x0 + 13, (top + bottom) / 2, x0 + 13, (top + bottom) / 2))
    return output


def main():
    official_rows = read_csv(DATA / "official_full28_curve.csv")
    b1 = read_csv(DATA / "b1_full28_endpoints.csv")
    official = [(float(row["mean_model_bpp"]), float(row["mean_model_yuv_psnr_611"]), row["rate_id"]) for row in official_rows]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="570" viewBox="0 0 1320 570">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="660" y="24" text-anchor="middle" font-size="16" font-weight="bold">B1 Full28 RD vs official Unicorn R01-R09</text>',
        '<text x="660" y="44" text-anchor="middle" font-size="11">B1 is derived from R02 (32k8k, lambda=16384); step4/8 are quantizer steps, not lambdas</text>',
    ]
    svg.extend(panel(official, b1, 10, 55, 790, 500, (0.03, 1.68), (29.8, 43.0), "Full curve and B1 Base-to-Full paths"))
    svg.extend(panel(official, b1, 810, 55, 500, 500, (0.38, 0.62), (34.8, 37.0), "Zoom: B1 Base in R04-R03 gap", zoom=True))
    svg.append('</svg>')
    output = HERE / "B1_FULL28_RD_VS_OFFICIAL.svg"
    output.write_text("\n".join(svg), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
