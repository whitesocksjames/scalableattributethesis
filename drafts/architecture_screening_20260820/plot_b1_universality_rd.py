#!/usr/bin/env python3
"""Dependency-free Full28 RD visualization for R01/R02/R04-derived B1."""
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "rd_plot_data"


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    official = read(DATA / "official_full28_curve.csv")
    sources = []
    for source, filename in (
            ("R01", "b1_r01_full28_endpoints.csv"),
            ("R02", "b1_full28_endpoints.csv"),
            ("R04", "b1_r04_full28_endpoints.csv")):
        for row in read(DATA / filename):
            sources.append((
                source, int(row["step"]),
                float(row.get("base_bpp", row.get("mean_model_base_bpp"))),
                float(row.get("base_yuv_psnr_611", row.get("mean_model_base_yuv_psnr_611"))),
                float(row.get("full_bpp", row.get("mean_model_full_bpp"))),
                float(row.get("full_yuv_psnr_611", row.get("mean_model_full_yuv_psnr_611"))),
            ))
    width, height = 1220, 690
    left, top, right, bottom = 85, 70, 1180, 610
    xmin, xmax, ymin, ymax = 0.03, 1.68, 29.8, 43.0
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    sy = lambda y: bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(width, height),
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="610" y="27" text-anchor="middle" font-size="18" font-weight="bold">B1 Full28 universality evidence vs official Unicorn</text>',
        '<text x="610" y="48" text-anchor="middle" font-size="11">All B1 points use fixed step4/8; step is not lambda. Lines show Base to exact source Full.</text>',
        '<rect x="{}" y="{}" width="{}" height="{}" fill="white" stroke="#999"/>'.format(left, top, right-left, bottom-top),
    ]
    for i in range(6):
        x = xmin + i * (xmax-xmin)/5
        xp = sx(x)
        svg += ['<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#ddd"/>'.format(xp, top, bottom),
                '<text x="{:.1f}" y="630" text-anchor="middle" font-size="11">{:.2f}</text>'.format(xp, x)]
    for i in range(7):
        y = ymin + i * (ymax-ymin)/6
        yp = sy(y)
        svg += ['<line x1="{1}" y1="{0:.1f}" x2="{2}" y2="{0:.1f}" stroke="#ddd"/>'.format(yp, left, right),
                '<text x="75" y="{:.1f}" text-anchor="end" font-size="11">{:.1f}</text>'.format(yp+4, y)]
    points = [(float(r["mean_model_bpp"]), float(r["mean_model_yuv_psnr_611"]), r["rate_id"]) for r in official]
    svg.append('<polyline points="{}" fill="none" stroke="#222" stroke-width="2.2"/>'.format(" ".join("{:.1f},{:.1f}".format(sx(x), sy(y)) for x,y,_ in points)))
    for x,y,label in points:
        svg += ['<circle cx="{:.1f}" cy="{:.1f}" r="5" fill="#222"/>'.format(sx(x),sy(y)),
                '<text x="{:.1f}" y="{:.1f}" font-size="10">{}</text>'.format(sx(x)+6,sy(y)-7,label)]
    colors = {"R01":"#d62728", "R02":"#2ca02c", "R04":"#1f77b4"}
    offsets = {("R01",4):-12, ("R01",8):17, ("R02",4):-12, ("R02",8):17,
               ("R04",4):-12, ("R04",8):17}
    for source,step,bx,by,fx,fy in sources:
        color=colors[source]
        svg.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="1.4" opacity="0.7"/>'.format(sx(bx),sy(by),sx(fx),sy(fy),color))
        if step == 4:
            svg.append('<polygon points="{0:.1f},{1:.1f} {2:.1f},{3:.1f} {4:.1f},{5:.1f} {6:.1f},{7:.1f}" fill="{8}" stroke="white"/>'.format(sx(bx),sy(by)-7,sx(bx)+7,sy(by),sx(bx),sy(by)+7,sx(bx)-7,sy(by),color))
        else:
            svg.append('<rect x="{:.1f}" y="{:.1f}" width="13" height="13" fill="{}" stroke="white"/>'.format(sx(bx)-6.5,sy(by)-6.5,color))
        svg.append('<circle cx="{:.1f}" cy="{:.1f}" r="7" fill="{}" stroke="white"/>'.format(sx(fx),sy(fy),color))
        svg.append('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="11">{} s{} Base</text>'.format(sx(bx)+8,sy(by)+offsets[(source,step)],color,source,step))
    svg += [
        '<text x="{}" y="666" text-anchor="middle" font-size="13">Physical bitrate (bpp)</text>'.format((left+right)/2),
        '<text x="20" y="{}" transform="rotate(-90 20 {})" text-anchor="middle" font-size="13">pc_error YUV-PSNR 6:1:1 (dB)</text>'.format((top+bottom)/2,(top+bottom)/2),
        '<text x="900" y="655" font-size="11" fill="#d62728">R01-derived</text>',
        '<text x="990" y="655" font-size="11" fill="#2ca02c">R02-derived</text>',
        '<text x="1080" y="655" font-size="11" fill="#1f77b4">R04-derived</text>',
        '</svg>'
    ]
    output = HERE / "B1_R01_R02_R04_FULL28_RD.svg"
    output.write_text("\n".join(svg), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
