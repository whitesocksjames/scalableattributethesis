#!/usr/bin/env python3
"""Resolve the reviewed 8-way config into auditable per-arm command files."""

import argparse
import json
from pathlib import Path
import shlex


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--source-root",required=True,type=Path)
    p.add_argument("--command-dir",required=True,type=Path)
    p.add_argument("--expected-source-commit",required=True)
    p.add_argument("--smoke-output-root",type=Path)
    a=p.parse_args(); cfg=json.loads(a.config.read_text()); common=cfg["common"]
    if len(cfg["arms"])!=8 or len({x["name"] for x in cfg["arms"]})!=8:
        raise ValueError("Expected eight unique rescue arms")
    a.command_dir.mkdir(parents=True,exist_ok=True)
    python="/data/run01/scz0ade/Tanzeyu/envs/unicorn-me-py38/bin/python"
    script=a.source_root/"scripts/scalable_attribute/canonical/train_base_rescue.py"
    for arm in cfg["arms"]:
        released=Path(common["released_root"])/arm["profile"]/"epoch_last.pth"
        smoke = a.smoke_output_root is not None
        out=(a.smoke_output_root/arm["name"] if smoke else
             Path(common["output_root"])/arm["name"]/"train")
        cmd=[python,str(script),"--experiment-name",arm["name"],"--data-root",common["data_root"],
             "--train-file-list",common["train_file_list"],"--content-statistics",common["content_statistics"],
             "--released-checkpoint",str(released),"--checkpoint-profile",arm["profile"],
             "--conditioning-lambda",str(arm["lambda"]),"--trainable-scope",arm["scope"],
             "--sampling",arm["sampling"],"--prefix-lr",str(arm.get("prefix_lr",1e-5)),
             "--base-synthesis-lr",str(arm["base_synthesis_lr"]),"--batch-size",str(common["batch_size"]),
             "--max-steps",str(arm["max_steps"]),"--save-steps",*[str(x) for x in arm["save_steps"]],
             "--seed",str(common["seed"]),"--num-workers","0","--output-dir",str(out),
             "--expected-source-commit",a.expected_source_commit]
        if arm["initial_base"]: cmd.extend(["--initial-base-checkpoint",arm["initial_base"]])
        if smoke: cmd.append("--smoke-only")
        path=a.command_dir/(arm["name"]+".sh")
        path.write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            "export PYTHONDONTWRITEBYTECODE=1\n"
            "export PYTHONPATH={}\n"
            "cd {}\nexec {}\n".format(
                shlex.quote(str(a.source_root)),
                shlex.quote(str(a.source_root)), shlex.join(cmd)))
        path.chmod(0o700)
    print(json.dumps({"status":"PASS","arms":[x["name"] for x in cfg["arms"]]},indent=2))


if __name__=="__main__": main()
