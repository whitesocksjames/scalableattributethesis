#!/usr/bin/env python3
"""Physical hard evaluation of one complete Base-rescue checkpoint."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5, read_ply_ascii, write_ply_ascii
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import aggregate_models, average_models, sample_identity
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import (
    metric, reconstruction_rgb, sparse_max_difference)


def parse_args():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--file-list")
    source.add_argument("--input-ply")
    p.add_argument("--data-root")
    p.add_argument("--sequence")
    p.add_argument("--frame")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--released-checkpoint",required=True)
    p.add_argument("--checkpoint-profile",required=True)
    p.add_argument("--conditioning-lambda",type=int,required=True)
    p.add_argument("--gpcc-binary",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--max-samples",type=int,default=0)
    return p.parse_args()


def write_csv(path,rows):
    if not rows: raise ValueError("No rows")
    with open(path,"w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def inputs(args):
    if args.file_list:
        if not args.data_root: raise ValueError("--data-root required with --file-list")
        with open(args.file_list,encoding="utf-8") as h: entries=[x.strip() for x in h if x.strip()]
        files=h5_files(args.data_root,args.file_list)
        result=[]
        for entry,path in zip(entries,files):
            model,part=sample_identity(entry); result.append((entry,path,model,part,"h5"))
    else:
        sequence=args.sequence or Path(args.input_ply).stem
        result=[("{}:{}".format(sequence,args.frame or "single"),args.input_ply,sequence,0,"ply")]
    return result[:args.max_samples] if args.max_samples else result


def main():
    args=parse_args()
    for name in ("checkpoint","released_checkpoint","gpcc_binary","output_dir"):
        setattr(args,name,os.path.abspath(os.path.expandvars(getattr(args,name))))
    for name in ("data_root","file_list","input_ply"):
        value=getattr(args,name)
        if value: setattr(args,name,os.path.abspath(os.path.expandvars(value)))
    if os.path.basename(os.path.dirname(args.released_checkpoint)) != args.checkpoint_profile:
        raise ValueError("checkpoint-profile/path mismatch")
    state=torch.load(args.checkpoint,map_location="cpu")
    if state.get("architecture")!="canonical_base_rescue_v1": raise ValueError("Base rescue architecture mismatch")
    if int(state.get("conditioning_lambda",-1))!=args.conditioning_lambda: raise ValueError("lambda mismatch")
    if state.get("checkpoint_profile")!=args.checkpoint_profile: raise ValueError("profile mismatch")
    protected=("per_h5.csv","per_model.csv","endpoint_summary.csv")
    if any(os.path.exists(os.path.join(args.output_dir,x)) for x in protected): raise FileExistsError("Refusing overwrite")
    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,"resolved_args.json"),"w",encoding="utf-8") as h: json.dump(vars(args),h,indent=2)
    with open(os.path.join(args.output_dir,"command.txt"),"w",encoding="utf-8") as h: h.write(shlex.join([sys.executable]+sys.argv)+"\n")
    link=os.path.join(args.output_dir,"tmc3_v21")
    if not os.path.exists(link): os.symlink(args.gpcc_binary,link)
    os.chdir(args.output_dir)
    model=CanonicalBaseModel(args.released_checkpoint,BaseSynthesisConfig(**state["config"])).cuda().eval()
    model.load_state_dict(state["base_model"],strict=True); model.requires_grad_(False)
    rows=[]; metric_dir=os.path.join(args.output_dir,"metric_tmp"); os.makedirs(metric_dir,exist_ok=True)
    for index,(entry,path,model_id,partition_id,kind) in enumerate(inputs(args)):
        started=time.perf_counter(); coords,rgb=read_h5(path) if kind=="h5" else read_ply_ascii(path)
        yuv=rgb2yuv(rgb.astype("float32"),out_range=1).astype("float32")
        bc,bf=ME.utils.sparse_collate([coords],[yuv]); attribute=ME.SparseTensor(features=bf,coordinates=bc,tensor_stride=1,device="cuda")
        soft=model.prefix(attribute,args.conditioning_lambda)
        hard,rate=model.prefix.hard_forward(attribute,args.conditioning_lambda,return_details=True)
        if rate["num_residual_streams"]!=4 or len(rate["residual_bits"])!=4: raise RuntimeError("Base must code r1-r4 only")
        if rate["base_bits"]!=rate["bits_xlow"]+sum(rate["residual_bits"]): raise RuntimeError("Base bit identity failed")
        soft_base=model.reconstruct_from_state(soft)["Base"]; hard_base=model.reconstruct_from_state(hard)["Base"]
        difference=sparse_max_difference(soft_base,hard_base,"Base soft/hard diagnostic")
        gt=os.path.join(metric_dir,"gt.ply"); rec=os.path.join(metric_dir,"base.ply")
        write_ply_ascii(gt,coords,rgb); write_ply_ascii(rec,hard_base.C[:,1:].cpu().numpy(),reconstruction_rgb(hard_base))
        quality=metric(gt,rec); rb=rate["residual_bits"]
        rows.append({"candidate":Path(args.checkpoint).stem,"checkpoint_step":int(state["step"]),
            "rate_id":state["resolved_args"]["experiment_name"],"checkpoint_profile":args.checkpoint_profile,
            "base_lambda":args.conditioning_lambda,"model_id":model_id,"partition_id":partition_id,
            "sample":entry,"points":len(attribute),"x_low_bits":rate["bits_xlow"],
            "r1_bits":rb[0],"r2_bits":rb[1],"r3_bits":rb[2],"r4_bits":rb[3],
            "num_base_residual_streams":4,"num_native_r5_streams":0,"enhancement_bits":0,
            "base_bits":rate["base_bits"],"physical_bits":rate["base_bits"],
            "base_bpp":rate["base_bits"]/len(attribute),**{"base_"+k:v for k,v in quality.items()},
            "soft_hard_max_abs_difference":difference,"hard_prefix_decode_used":True,
            "bit_identity_pass":True,"seconds":time.perf_counter()-started})
        write_csv(os.path.join(args.output_dir,"per_h5.csv"),rows)
        print("[{}/{}] {}".format(index+1,len(inputs(args)),entry),flush=True)
    prepared=[{**r,"rate_id":r["rate_id"]} for r in rows]
    per_model=aggregate_models(prepared,bits_field="base_bits",metric_prefix="base_")
    summary=average_models(per_model)
    summary.update({"status":"PASS","hard_prefix_decode_used":True,"bit_identity_pass":True,
                    "num_base_residual_streams":4,"num_native_r5_streams":0,"enhancement_bits":0,
                    "soft_hard_max_abs_difference":max(r["soft_hard_max_abs_difference"] for r in rows)})
    write_csv(os.path.join(args.output_dir,"per_model.csv"),per_model)
    write_csv(os.path.join(args.output_dir,"endpoint_summary.csv"),[summary])
    with open(os.path.join(args.output_dir,"endpoint_summary.json"),"w",encoding="utf-8") as h: json.dump(summary,h,indent=2)


if __name__=="__main__": main()
