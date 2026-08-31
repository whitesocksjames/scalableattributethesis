"""Small, shared resolver for canonical multi-rate operating points."""

from copy import deepcopy
import json
import os
from pathlib import Path

from scalable_attribute.reference_points import reference_point


DEFAULT_CONFIG = str(
    Path(__file__).resolve().parents[2]
    / "configs" / "scalable_attribute" / "canonical_operating_points.json")


def load_operating_points(path=DEFAULT_CONFIG):
    path = os.path.abspath(os.path.expandvars(path))
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported operating-point config schema")
    points = config.get("points")
    if not isinstance(points, dict) or not points:
        raise ValueError("Operating-point config has no points")

    for point_id, point in points.items():
        required = ("rate_id", "conditioning_lambda", "released_profile")
        missing = [name for name in required if name not in point]
        if missing:
            raise ValueError(
                "{} is missing {}".format(point_id, ", ".join(missing)))
        official_id, official_profile, official_lambda = reference_point(
            point["rate_id"])
        actual = (
            point["rate_id"], point["released_profile"],
            int(point["conditioning_lambda"]))
        if actual != (official_id, official_profile, official_lambda):
            raise ValueError(
                "{} conflicts with official Unicorn mapping: {}".format(
                    point_id, actual))
    return config, path


def resolve_operating_point(point_id, released_root, config_path=DEFAULT_CONFIG):
    """Resolve one point without putting a cluster-specific path in config."""
    config, source = load_operating_points(config_path)
    key = point_id.lower()
    if key not in config["points"]:
        raise ValueError(
            "Unknown point {}; choose from {}".format(
                point_id, ", ".join(config["points"])))
    if not released_root:
        raise ValueError("--released-root is required when --point is used")
    point = deepcopy(config["points"][key])
    point.update({
        "id": key,
        "config_path": source,
        "released_root": os.path.abspath(os.path.expandvars(released_root)),
        "base": deepcopy(config["canonical_base"]),
        "enhancement": deepcopy(config.get("enhancement", {})),
    })
    point["released_checkpoint"] = os.path.join(
        point["released_root"], point["released_profile"], "epoch_last.pth")
    return point


def point_for_lambda(conditioning_lambda, profile=None, config_path=DEFAULT_CONFIG):
    """Return the configured point for a lambda/profile pair."""
    config, _ = load_operating_points(config_path)
    matches = []
    for point_id, point in config["points"].items():
        if int(point["conditioning_lambda"]) != int(conditioning_lambda):
            continue
        if profile is not None and point["released_profile"] != profile:
            continue
        matches.append((point_id, point))
    if len(matches) != 1:
        raise ValueError(
            "Expected one configured point for lambda={} profile={}, got {}"
            .format(conditioning_lambda, profile, len(matches)))
    return matches[0]
