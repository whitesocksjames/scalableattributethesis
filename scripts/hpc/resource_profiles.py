RESOURCE_PROFILES = {
    "train_v100": {
        "submit": "sbatch.tinygpu",
        "partition": "v100",
        "gres": "gpu:v100:1",
        "cpus": 8,
        "walltime": "1-00:00:00",
    },
    "train_v100_6h": {
        "submit": "sbatch.tinygpu",
        "partition": "v100",
        "gres": "gpu:v100:1",
        "cpus": 8,
        "walltime": "06:00:00",
    },
    "smoke_v100": {
        "submit": "sbatch.tinygpu",
        "partition": "v100",
        "gres": "gpu:v100:1",
        "cpus": 8,
        "walltime": "00:30:00",
    },
    "eval_rtx3080": {
        "submit": "sbatch.tinygpu",
        "partition": "work,rtx3080",
        "gres": "gpu:rtx3080:1",
        "cpus": 8,
        "walltime": "04:00:00",
    },
    "smoke_rtx3080": {
        "submit": "sbatch.tinygpu",
        "partition": "work,rtx3080",
        "gres": "gpu:rtx3080:1",
        "cpus": 8,
        "walltime": "00:30:00",
    },
    "eval_work_any": {
        "submit": "sbatch.tinygpu",
        "partition": "work,rtx3080",
        "gres": "gpu:1",
        "cpus": 8,
        "walltime": "04:00:00",
    },
    "smoke_work_any": {
        "submit": "sbatch.tinygpu",
        "partition": "work,rtx3080",
        "gres": "gpu:1",
        "cpus": 8,
        "walltime": "00:30:00",
    },
    "eval_all_gpu": {
        "submit": "sbatch.tinygpu",
        "partition": "work,rtx3080,v100,a100",
        "gres": "gpu:1",
        "cpus": 8,
        "walltime": "02:00:00",
    },
    "smoke_all_gpu": {
        "submit": "sbatch.tinygpu",
        "partition": "work,rtx3080,v100,a100",
        "gres": "gpu:1",
        "cpus": 8,
        "walltime": "00:30:00",
    },
}

DEFAULT_RESOURCE_PROFILES = {
    "train": "train_v100",
    "eval": "eval_all_gpu",
    "smoke": "smoke_all_gpu",
}


def get_resource_profile(name):
    try:
        return dict(RESOURCE_PROFILES[name])
    except KeyError:
        raise ValueError("Unknown resource profile {!r}; choose from {}".format(
            name, ", ".join(sorted(RESOURCE_PROFILES))))
