#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
with (root / "trajectory.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
for axis, point in zip(axes, ("2K", "4K")):
    selected = [row for row in rows if row["point"] == point]
    steps = [int(row["step"]) for row in selected]
    axis.plot(steps, [float(row["rwtt_delta_from_step500_db"]) for row in selected],
              "o-", label="RWTT-28Lite")
    axis.plot(steps, [float(row["longdress_delta_from_step500_db"]) for row in selected],
              "s--", label="Longdress1300")
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    axis.set_title(point)
    axis.set_xlabel("Training step")
    axis.grid(True, alpha=0.25)
axes[0].set_ylabel("YUV611 gain from step500 (dB)")
axes[0].legend()
figure.suptitle("Low-rate Canonical Base training trajectory")
figure.tight_layout()
figure.savefig(root / "lowrate_longdress_training_diagnosis.png", dpi=180)
figure.savefig(root / "lowrate_longdress_training_diagnosis.svg")
