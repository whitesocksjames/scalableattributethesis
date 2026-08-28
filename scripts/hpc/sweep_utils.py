import itertools
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess

from resource_profiles import get_resource_profile


META_KEYS = {
    "name", "base_profile", "base_checkpoint", "train_experiment",
    "checkpoint_tag",
}


def grid(common=None, **axes):
    """Return a small Cartesian sweep as ordinary flat experiment dicts."""
    common = dict(common or {})
    names = list(axes)
    values = [value if isinstance(value, (list, tuple)) else [value]
              for value in axes.values()]
    return [dict(common, **dict(zip(names, combination)))
            for combination in itertools.product(*values)]


def load_spec(path, kind):
    namespace = runpy.run_path(path)
    key = kind.upper() + "_EXPERIMENTS"
    experiments = namespace.get(key, namespace.get("EXPERIMENTS"))
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("{} must define a non-empty list".format(key))
    common = namespace.get(kind.upper() + "_COMMON", {})
    if not isinstance(common, dict):
        raise ValueError("{}_COMMON must be a dict".format(kind.upper()))
    resolved = []
    for item in experiments:
        if not isinstance(item, dict):
            raise ValueError("Each experiment must be a dict")
        merged = dict(common)
        merged.update(item)
        if "name" not in merged and kind == "eval" and merged.get("train_experiment"):
            tag = merged.get("checkpoint_tag", "checkpoint")
            merged["name"] = "{}_{}".format(merged["train_experiment"], tag)
        if "name" not in merged:
            merged["name"] = auto_name(
                merged, namespace.get("EXPERIMENT_NAME_FIELDS", ()))
        resolved.append(merged)
    names = [item["name"] for item in resolved]
    if len(names) != len(set(names)):
        raise ValueError(
            "Experiment names are not unique; add an explicit name for runs that "
            "share the auto-name fields")
    return namespace, resolved


def auto_name(experiment, name_fields):
    parts = []
    for key, prefix in name_fields:
        if key in experiment:
            parts.append(prefix + str(experiment[key]).replace(".", "p"))
    if not parts:
        raise ValueError(
            "Experiment needs a name or EXPERIMENT_NAME_FIELDS in the spec")
    return "_".join(parts)


def experiment_cli(experiment, excluded=()):
    cli = []
    for key, value in experiment.items():
        if key in META_KEYS or key in excluded:
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(option)
        elif isinstance(value, (list, tuple)):
            cli.append(option)
            cli.extend(str(item) for item in value)
        elif value is not None:
            cli.extend([option, str(value)])
    return cli


def resolve_base_checkpoint(experiment, namespace, checkpoints_root):
    profiles = namespace.get("BASE_CHECKPOINT_PROFILES", {})
    profile_name = experiment.get(
        "base_profile", namespace.get("DEFAULT_BASE_PROFILE"))
    explicit = experiment.get("base_checkpoint")
    if not profile_name and explicit:
        return os.path.expanduser(os.path.expandvars(explicit))
    if not profile_name or profile_name not in profiles:
        raise ValueError(
            "Set base_checkpoint or select a known base_profile; got {!r}".format(
                profile_name))
    profile = profiles[profile_name]
    if isinstance(profile, dict) and "validated_lambda" in profile:
        actual_lambda = experiment.get("base_lambda")
        if actual_lambda != profile["validated_lambda"]:
            raise ValueError(
                "Base profile {} requires base_lambda={}, got {}".format(
                    profile_name, profile["validated_lambda"], actual_lambda))
    if explicit:
        return os.path.expanduser(os.path.expandvars(explicit))
    relative = profile["checkpoint"] if isinstance(profile, dict) else profile
    return os.path.join(checkpoints_root, relative)


def checkpoint_step(path):
    name = Path(path).stem
    if name.startswith("step_") and name[5:].isdigit():
        return int(name[5:])
    return -1


def resolve_checkpoint(checkpoint_dir, tag, require_exists=True):
    checkpoint_dir = Path(checkpoint_dir)
    if tag == "latest":
        numbered = sorted(checkpoint_dir.glob("step_*.pth"), key=checkpoint_step)
        selected = numbered[-1] if numbered else None
    elif tag.startswith("step") and tag[4:].isdigit():
        selected = checkpoint_dir / "step_{}.pth".format(tag[4:])
    elif tag.startswith("step_") and tag[5:].isdigit():
        selected = checkpoint_dir / (tag + ".pth")
    else:
        raise ValueError("Checkpoint tag must be latest, step100, or step_100")
    if selected is None:
        raise FileNotFoundError(
            "No numbered checkpoint under {}; latest must resolve to step_N.pth".format(
                checkpoint_dir))
    if require_exists and not selected.exists():
        raise FileNotFoundError(str(selected))
    return str(selected), selected.stem


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_manifest(experiment_root):
    path = Path(experiment_root) / "manifest.json"
    if not path.exists():
        return {"version": 1, "experiment": Path(experiment_root).name,
                "output_dir": str(experiment_root), "train": {"attempts": []},
                "evaluations": {}, "references": {}}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def submit_job(command, job_name, experiment_root, run_dir, repo, python,
               profile_name, kind, dry_run=False, verbose=False,
               dependency=None, checkpoint_tag=None, walltime=None,
               retry=False):
    profile = get_resource_profile(profile_name)
    experiment_root = Path(experiment_root)
    run_dir = Path(run_dir)
    slurm_dir = experiment_root / "slurm"
    manifest = read_manifest(experiment_root)
    if kind == "train":
        attempts = manifest["train"].setdefault("attempts", [])
    elif kind == "eval":
        evaluation = manifest["evaluations"].setdefault(
            checkpoint_tag,
            {"checkpoint": command[command.index("--enhancement-checkpoint") + 1],
             "output_dir": str(run_dir), "attempts": []})
        attempts = evaluation["attempts"]
    elif kind == "reference":
        reference = manifest.setdefault("references", {}).setdefault(
            checkpoint_tag, {"output_dir": str(run_dir), "attempts": []})
        attempts = reference["attempts"]
    else:
        raise ValueError("Unknown job kind: " + kind)
    if any(item.get("job_id") for item in attempts) and not retry and not dry_run:
        raise RuntimeError(
            "{} already has a recorded submission; use explicit retry or a new "
            "experiment name".format(run_dir))
    attempt = len(attempts)
    command_path = run_dir / "command_attempt{}.txt".format(attempt)
    stdout = str(slurm_dir / "{}_attempt{}_{}.out".format(kind, attempt, "%j"))
    stderr = str(slurm_dir / "{}_attempt{}_{}.err".format(kind, attempt, "%j"))
    wrapped = "module load cuda/11.8.0; export PATH={}:$PATH; export PYTHONPATH={}; {}".format(
        shlex.quote(os.path.dirname(python)), shlex.quote(repo), shlex.join(command))
    sbatch = [
        profile["submit"], "--parsable",
        "--partition=" + profile["partition"],
        "--gres=" + profile["gres"],
        "--nodes=1", "--ntasks=1",
        "--cpus-per-task=" + str(profile["cpus"]),
        "--time=" + (walltime or profile["walltime"]),
        "--job-name=" + job_name[:80],
        "--output=" + stdout,
        "--error=" + stderr,
    ]
    if dependency:
        sbatch.append("--dependency=afterok:" + str(dependency))
    sbatch.append("--wrap=" + wrapped)
    print("{}: {} -> {} [{}]".format(kind, job_name, run_dir, profile_name))
    if verbose:
        print(shlex.join(sbatch))
    if dry_run:
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    try:
        result = subprocess.run(sbatch, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        attempts.append({
            "submission_error": (error.stderr or str(error)).strip(),
            "profile": profile_name,
            "command": str(command_path),
            "stdout": stdout,
            "stderr": stderr,
        })
        write_json(experiment_root / "manifest.json", manifest)
        raise
    job_id = result.stdout.strip().split(";")[0]
    record = {
        "job_id": job_id,
        "profile": profile_name,
        "command": str(command_path),
        "stdout": stdout.replace("%j", job_id),
        "stderr": stderr.replace("%j", job_id),
    }
    if dependency:
        record["afterok"] = str(dependency)
    attempts.append(record)
    write_json(experiment_root / "manifest.json", manifest)
    print("submitted job {}".format(job_id))
    print("SUBMITTED {} {}".format(job_name, job_id))
    return job_id
