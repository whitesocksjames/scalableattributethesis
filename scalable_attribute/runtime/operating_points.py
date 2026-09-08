"""Small, shared resolver for canonical multi-rate operating points."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Optional

from scalable_attribute.reference_points import reference_point


DEFAULT_CONFIG = str(
    Path(__file__).resolve().parents[2]
    / "configs" / "scalable_attribute" / "canonical_operating_points.json")


@dataclass(frozen=True)
class OperatingPointConfig:
    """Resolved canonical point with cluster paths supplied at runtime."""

    id: str
    rate_id: str
    conditioning_lambda: int
    released_profile: str
    role: str
    released_root: str
    released_checkpoint: str
    canonical_experiment_root: str
    selected_base_checkpoint: Optional[str]
    config_path: str
    base: dict
    enhancement: dict

    @classmethod
    def resolve(cls, point_id, released_root, canonical_experiment_root,
                config_path=DEFAULT_CONFIG, require_selected_base=True):
        config, source = load_operating_points(config_path)
        key = point_id.lower()
        if key not in config["points"]:
            raise ValueError(
                "Unknown point {}; choose from {}".format(
                    point_id, ", ".join(config["points"])))
        if not released_root:
            raise ValueError("released_root is required to resolve a point")
        if not canonical_experiment_root:
            raise ValueError(
                "canonical_experiment_root is required to resolve selected Base")
        point = config["points"][key]
        released_root = os.path.abspath(os.path.expandvars(released_root))
        experiment_root = os.path.abspath(os.path.expandvars(
            canonical_experiment_root))
        selected_base = point["selected_base_checkpoint"]
        resolved = cls(
            id=key,
            rate_id=point["rate_id"],
            conditioning_lambda=int(point["conditioning_lambda"]),
            released_profile=point["released_profile"],
            role=point["role"],
            released_root=released_root,
            released_checkpoint=os.path.join(
                released_root, point["released_profile"], "epoch_last.pth"),
            canonical_experiment_root=experiment_root,
            selected_base_checkpoint=(
                None if selected_base is None else
                os.path.join(experiment_root, selected_base)),
            config_path=source,
            base=deepcopy(config["canonical_base"]),
            enhancement=deepcopy(config["enhancement"]),
        )
        if require_selected_base:
            resolved.require_selected_base()
        return resolved

    def require_selected_base(self):
        """Reject points that have not passed canonical Base selection."""
        if self.selected_base_checkpoint is None:
            raise ValueError(
                "Operating point {} ({}) is diagnostic and has no selected "
                "Base checkpoint".format(self.id, self.rate_id))
        return self.selected_base_checkpoint

    def to_dict(self):
        return asdict(self)


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
        required = (
            "rate_id", "conditioning_lambda", "released_profile",
            "role", "selected_base_checkpoint")
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
        role = point["role"]
        if role not in ("canonical_selected", "diagnostic"):
            raise ValueError(
                "{} has unsupported role: {}".format(point_id, role))
        selected_base = point["selected_base_checkpoint"]
        if role == "canonical_selected" and not selected_base:
            raise ValueError(
                "{} canonical_selected point requires a selected Base "
                "checkpoint".format(point_id))
        if role == "diagnostic" and selected_base is not None:
            raise ValueError(
                "{} diagnostic point must not claim a selected Base "
                "checkpoint".format(point_id))
        if (selected_base is not None and
                (not isinstance(selected_base, str) or
                 os.path.isabs(selected_base) or
                 ".." in Path(selected_base).parts)):
            raise ValueError(
                "{} selected Base checkpoint must be relative to the canonical "
                "experiment root".format(point_id))

    enhancement = config.get("enhancement", {})
    if enhancement.get("recipe_status") != "provisional_stage_gated":
        raise ValueError(
            "Canonical Enhancement recipe must be provisional_stage_gated")
    for stage_name in ("stage1", "stage2"):
        stage = enhancement.get(stage_name)
        if not isinstance(stage, dict):
            raise ValueError("Enhancement config is missing " + stage_name)
    if enhancement["stage2"].get("manager_trigger_required") is not True:
        raise ValueError(
            "Enhancement Stage 2 must require an explicit manager trigger")
    return config, path


def resolve_operating_point(point_id, released_root, config_path=DEFAULT_CONFIG):
    """Resolve one point without putting a cluster-specific path in config."""
    # Existing Base/evaluation callers do not need a selected Base path. Use a
    # neutral root while retaining the historical dictionary return type.
    point = OperatingPointConfig.resolve(
        point_id, released_root, canonical_experiment_root=os.curdir,
        config_path=config_path, require_selected_base=False).to_dict()
    point.pop("canonical_experiment_root")
    point.pop("selected_base_checkpoint")
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
