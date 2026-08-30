"""Small shared helpers for H5-to-original-model RD aggregation."""

import math
from pathlib import PurePosixPath
import re


_MODEL = re.compile(r"RWT[0-9]+")
_PARTITION = re.compile(r".+_P([0-9]+)\.h5")
_MVUB_SUBJECT = re.compile(r"[A-Za-z]+10")
_MVUB_SAMPLE = re.compile(
    r"(?P<subject>[A-Za-z]+10)_(?P<frame>frame[0-9]{4})_P(?P<part>[0-9]+)\.h5")


def sample_identity(manifest_entry):
    path = PurePosixPath(manifest_entry)
    if len(path.parts) == 2 and _MODEL.fullmatch(path.parts[0]):
        match = _PARTITION.fullmatch(path.name)
        if match:
            return path.parts[0], int(match.group(1))
    if len(path.parts) == 2 and _MVUB_SUBJECT.fullmatch(path.parts[0]):
        match = _MVUB_SAMPLE.fullmatch(path.name)
        if match and match.group("subject") == path.parts[0]:
            return "{}/{}".format(
                path.parts[0], match.group("frame")), int(match.group("part"))
    raise ValueError(
        "Expected RWTT or MVUB partition manifest entry: " + manifest_entry)


def _model_sort_key(value):
    match = _MODEL.fullmatch(value)
    if match:
        return (0, int(value[3:]))
    return (1, value)


def psnr(mse, peak=1.0):
    """PSNR for pc_error color MSE, which is normalized to the [0, 1] range."""
    if mse < 0:
        raise ValueError("MSE cannot be negative")
    return float("inf") if mse == 0 else 10.0 * math.log10(peak * peak / mse)


def aggregate_models(rows, bits_field="base_bits", metric_prefix=""):
    grouped = {}
    for row in rows:
        current = grouped.setdefault(row["model_id"], {
            "rate_id": row["rate_id"],
            "checkpoint_profile": row["checkpoint_profile"],
            "base_lambda": int(row["base_lambda"]),
            "model_id": row["model_id"],
            "num_h5": 0,
            "total_points": 0,
            "total_bits": 0,
            "y_sse": 0.0,
            "u_sse": 0.0,
            "v_sse": 0.0,
        })
        points = int(row["points"])
        current["num_h5"] += 1
        current["total_points"] += points
        current["total_bits"] += int(row[bits_field])
        for channel in "yuv":
            key = metric_prefix + channel + "_mse"
            current[channel + "_sse"] += float(row[key]) * points

    output = []
    for model_id in sorted(grouped, key=_model_sort_key):
        row = grouped[model_id]
        points = row["total_points"]
        result = {key: row[key] for key in (
            "rate_id", "checkpoint_profile", "base_lambda", "model_id",
            "num_h5", "total_points", "total_bits")}
        result["bpp"] = row["total_bits"] / points
        channel_psnr = {}
        for channel in "yuv":
            mse = row[channel + "_sse"] / points
            result[channel + "_mse"] = mse
            result[channel + "_psnr"] = psnr(mse)
            channel_psnr[channel] = result[channel + "_psnr"]
        result["yuv_psnr_611"] = (
            6.0 * channel_psnr["y"] + channel_psnr["u"]
            + channel_psnr["v"]) / 8.0
        output.append(result)
    return output


def average_models(rows):
    if not rows:
        raise ValueError("No per-model rows")
    first = rows[0]
    return {
        "rate_id": first["rate_id"],
        "checkpoint_profile": first["checkpoint_profile"],
        "base_lambda": int(first["base_lambda"]),
        "num_models": len(rows),
        "num_h5": sum(int(row["num_h5"]) for row in rows),
        "total_points": sum(int(row["total_points"]) for row in rows),
        "mean_model_bpp": sum(float(row["bpp"]) for row in rows) / len(rows),
        "mean_model_y_psnr": (
            sum(float(row["y_psnr"]) for row in rows) / len(rows)),
        "mean_model_yuv_psnr_611": (
            sum(float(row["yuv_psnr_611"]) for row in rows) / len(rows)),
    }
