#!/usr/bin/env python3
"""Merge, validate, aggregate and plot the fixed-model internal trace runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PEAK = 1.0


def psnr(mse):
    mse = np.asarray(mse, dtype=float)
    return -10.0 * np.log10(np.maximum(mse, np.finfo(float).tiny))


def weighted_endpoint(g):
    n = g.points.to_numpy(float)
    out = {
        "num_blocks": len(g), "points": int(n.sum()),
        "physical_bits": int(g.physical_bits.sum()),
    }
    out["physical_bpp"] = out["physical_bits"] / out["points"]
    for c in "yuv":
        out[f"{c}_mse"] = np.average(g[f"{c}_mse"], weights=n)
        out[f"{c}_psnr"] = float(psnr(out[f"{c}_mse"]))
    out["yuv611"] = (6*out["y_psnr"] + out["u_psnr"] + out["v_psnr"]) / 8
    for c in ["x_low_bits", "r1_bits", "r2_bits", "r3_bits", "r4_bits", "r5_bits"]:
        out[c] = int(g[c].sum())
    return pd.Series(out)


def weighted_stage(g):
    n = g.points.to_numpy(float)
    out = {"num_blocks": len(g), "points": int(n.sum()),
           "stage_bits": int(g.stage_bits.fillna(0).sum()),
           "cumulative_bits": int(g.cumulative_bits.fillna(0).sum())}
    out["stage_bpp"] = out["stage_bits"] / out["points"]
    out["cumulative_bpp"] = out["cumulative_bits"] / out["points"]
    for c in "YUV":
        out[f"mse_{c}"] = np.average(g[f"mse_{c}"], weights=n)
        out[f"psnr_{c}"] = float(psnr(out[f"mse_{c}"]))
    out["yuv611"] = (6*out["psnr_Y"] + out["psnr_U"] + out["psnr_V"]) / 8
    for c in ["x_rms", "f_rms", "d_rms", "f4_rms", "f5p_rms", "cB_rms", "FB_rms"]:
        if c in g and g[c].notna().any():
            z = g[c].notna()
            out[c] = float(np.sqrt(np.average(g.loc[z, c] ** 2, weights=n[z])))
    return pd.Series(out)


def load_run(root, name):
    d = root / name
    a = pd.read_csv(d / "internal_trace_blocks.csv")
    b = pd.read_csv(d / "full_resolution_blocks.csv")
    a["run"] = b["run"] = name
    return a, b, json.loads((d / "run_info.json").read_text())


def aggregate_frame(frame, keys, function):
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(values, tuple): values = (values,)
        row = dict(zip(keys, values)); row.update(function(group).to_dict()); rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--content-csv", type=Path, required=True)
    p.add_argument("--matched-8i", type=Path, required=True)
    p.add_argument("--matched-owlii", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)

    core_i, core_f, core_info = load_run(a.raw_root, "full_88blocks_v2")
    fam_i, fam_f, fam_info = load_run(a.raw_root, "external_family_probes_88blocks")
    rw_i, rw_f, rw_info = load_run(a.raw_root, "rwtt_reference_88blocks")

    # Strong completeness and overlap validation.
    expected_ext = {"Longdress":16,"Loot":16,"Redandblack":8,"Soldier":16,
                    "Basketball":8,"Dancer":8,"Exercise":8,"Model":8}
    expected_rw = {"train_low":20,"train_typical":20,"train_high":20,"val28":28}
    checks = []
    for label, x, expected, ncfg in [("core",core_f,expected_ext,5),("family",fam_f,expected_ext,4),
                                      ("rwtt",rw_f,expected_rw,5)]:
        assert x.configuration.nunique() == ncfg
        for cfg, z in x.groupby("configuration"):
            got = z[["sequence","source","block"]].drop_duplicates().groupby("sequence").size().to_dict()
            assert got == expected, (label,cfg,got)
        assert not x.duplicated(["configuration","source","block","endpoint"]).any()
        checks.append({"run":label,"configs":ncfg,"blocks_per_config":sum(expected.values()),"status":"PASS"})
    for x in [core_i,fam_i,rw_i]:
        assert not x.duplicated(["configuration","source","block","stage"]).any()
    for cfg1,cfg2 in [("R04_8k256_L4096","OFFICIAL_R04_8k256_L4096"),
                      ("R05_8k256_L2048","OFFICIAL_R05_8k256_L2048")]:
        l=core_f[core_f.configuration==cfg1].sort_values(["source","endpoint"]).reset_index(drop=True)
        r=fam_f[fam_f.configuration==cfg2].sort_values(["source","endpoint"]).reset_index(drop=True)
        l=l[l.endpoint.isin(r.endpoint.unique())].sort_values(["source","block","endpoint"]).reset_index(drop=True)
        r=r.sort_values(["source","block","endpoint"]).reset_index(drop=True)
        cols=[c for c in l if c not in ["configuration","run","base_sha256"]]
        assert l[cols].equals(r[cols]), f"overlap mismatch {cfg1}"
    checks.append({"run":"official overlap R04/R05","configs":2,"blocks_per_config":88,"status":"EXACT"})
    pd.DataFrame(checks).to_csv(a.output_dir/"validation_checks.csv",index=False)

    # Canonical merged raw tables (family run contributes diagnostic configurations only).
    fi=fam_i[fam_i.configuration.str.startswith("DIAGNOSTIC")]
    ff=fam_f[fam_f.configuration.str.startswith("DIAGNOSTIC")]
    external_i=pd.concat([core_i,fi],ignore_index=True)
    external_f=pd.concat([core_f,ff],ignore_index=True)
    all_i=pd.concat([external_i,rw_i],ignore_index=True)
    all_f=pd.concat([external_f,rw_f],ignore_index=True)
    all_i.to_csv(a.output_dir/"internal_trace_blocks_merged.csv",index=False)
    all_f.to_csv(a.output_dir/"full_resolution_blocks_merged.csv",index=False)

    keys=["run","configuration","point","profile","lambda","dataset","sequence","stage","stage_index"]
    st=aggregate_frame(all_i[all_i.stage!="base_features"],keys,weighted_stage)
    epkeys=["run","configuration","point","profile","lambda","dataset","sequence","endpoint"]
    ep=aggregate_frame(all_f,epkeys,weighted_endpoint)
    st.to_csv(a.output_dir/"internal_trace_sequences.csv",index=False)
    ep.to_csv(a.output_dir/"full_resolution_sequences.csv",index=False)

    # Group-level aggregation treats external sequences equally; RWTT strata separately.
    sg=st.groupby(["configuration","dataset","stage"],dropna=False).mean(numeric_only=True).reset_index()
    eg=ep.groupby(["configuration","dataset","endpoint"],dropna=False).mean(numeric_only=True).reset_index()
    sg.to_csv(a.output_dir/"internal_trace_groups.csv",index=False)
    eg.to_csv(a.output_dir/"full_resolution_groups.csv",index=False)

    # Full-resolution decomposition and r5 dependency.
    piv=ep.pivot_table(index=["configuration","point","profile","lambda","dataset","sequence"],
                       columns="endpoint",values=["physical_bpp","yuv611"]).reset_index()
    piv.columns=["_".join(str(v) for v in c if str(v)) if isinstance(c,tuple) else c for c in piv.columns]
    for lhs,rhs,name in [("yuv611_B_native","yuv611_B_unpool","gain_native_db"),
                         ("yuv611_Canonical_Base","yuv611_B_native","gain_synthesis_db"),
                         ("yuv611_Official_Full","yuv611_B_native","native_r5_gap_db"),
                         ("yuv611_Official_Full","yuv611_Canonical_Base","canonical_to_full_gap_db")]:
        if lhs in piv and rhs in piv: piv[name]=piv[lhs]-piv[rhs]
    piv["r5_delta_bpp"]=piv.get("physical_bpp_Official_Full")-piv.get("physical_bpp_B_native")
    piv.to_csv(a.output_dir/"reconstruction_decomposition_sequences.csv",index=False)

    # Same-lambda and alternative-family controls.
    def compare(cfg_a,cfg_b,label):
        z=ep[ep.configuration.isin([cfg_a,cfg_b]) & ep.endpoint.isin(["B_native","Official_Full"])]
        q=z.pivot_table(index=["dataset","sequence","endpoint"],columns="configuration",values=["physical_bpp","yuv611"]).reset_index()
        q.columns=["_".join(str(v) for v in c if str(v)) if isinstance(c,tuple) else c for c in q.columns]
        q["comparison"]=label
        q["delta_bpp_diagnostic_minus_official"]=q[f"physical_bpp_{cfg_b}"]-q[f"physical_bpp_{cfg_a}"]
        q["delta_yuv611_diagnostic_minus_official"]=q[f"yuv611_{cfg_b}"]-q[f"yuv611_{cfg_a}"]
        return q.dropna(subset=["delta_bpp_diagnostic_minus_official","delta_yuv611_diagnostic_minus_official"])
    controls=pd.concat([
      compare("R03_32k8k_L8192","DIAGNOSTIC_8k256_L8192","8192: 8k256 - 32k8k"),
      compare("R04_8k256_L4096","DIAGNOSTIC_32k8k_L4096_OUT_OF_RANGE_UNKNOWN","4096: 32k8k - 8k256"),
      compare("R05_8k256_L2048","DIAGNOSTIC_2k128_L2048_RANGE_UNKNOWN","2048: 2k128 - 8k256")
    ],ignore_index=True)
    controls.to_csv(a.output_dir/"family_controls_sequences.csv",index=False)

    # Align prior content metrics and matched-rate degradation.
    content=pd.read_csv(a.content_csv)
    m8=pd.read_csv(a.matched_8i); m8.sequence=m8.sequence.str.title(); m8.point=m8.point.str.upper()
    mo=pd.read_csv(a.matched_owlii); mo=mo[mo.endpoint.str.lower()=="base"].copy()
    mo.sequence=mo.sequence.replace({"basketball_player":"Basketball","dancer":"Dancer","exercise":"Exercise","model":"Model"})
    matched=pd.concat([m8[m8.point.isin(["4K","2K"])][["sequence","point","delta_matched_db"]].assign(dataset="8i"),
                       mo[mo.point.isin(["4K","2K"])][["sequence","point","delta_matched_db"]].assign(dataset="Owlii")])
    aligned=content.merge(matched,on=["dataset","sequence"],how="inner")
    # Add selected internal quantities from canonical configurations.
    cmap={"4K":"R04_8k256_L4096","2K":"R05_8k256_L2048"}
    rows=[]
    for _,r in aligned.iterrows():
        d=piv[(piv.configuration==cmap[r.point])&(piv.dataset==r.dataset)&(piv.sequence==r.sequence)]
        if len(d):
            z=r.to_dict(); z.update({k:d.iloc[0].get(k,np.nan) for k in ["gain_native_db","gain_synthesis_db","native_r5_gap_db","r5_delta_bpp"]}); rows.append(z)
    aligned=pd.DataFrame(rows); aligned.to_csv(a.output_dir/"content_internal_alignment.csv",index=False)
    metrics=["r2_E_D111","tv6_D111","var_V","native_r5_gap_db","r5_delta_bpp","gain_synthesis_db"]
    cr=[]
    for point,z in aligned.groupby("point"):
        for m in metrics:
            rho,pv=spearmanr(z[m],z.delta_matched_db,nan_policy="omit")
            cr.append({"point":point,"metric":m,"spearman_rho_vs_delta_matched":rho,"p_descriptive_only":pv,"n":z[[m,"delta_matched_db"]].dropna().shape[0]})
    pd.DataFrame(cr).to_csv(a.output_dir/"internal_degradation_correlations.csv",index=False)

    # Figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    order=["R03_32k8k_L8192","R04_8k256_L4096","R05_8k256_L2048","R06_2k128_L1024"]
    stages=["r1","r2","r3","r4","r5"]
    fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=False)
    for ax,((ds,seq),z) in zip(axs.flat,st[st.dataset.isin(["8i","Owlii"])].groupby(["dataset","sequence"],sort=False)):
        for cfg in order:
            q=z[(z.configuration==cfg)&z.stage.isin(stages)].sort_values("stage_index")
            ax.plot(q.cumulative_bpp,q.yuv611,"o-",label=cfg.split("_")[0])
        ax.set_title(f"{ds}/{seq}"); ax.set_xlabel("cumulative bpp"); ax.set_ylabel("stage-local YUV611")
    axs[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(a.output_dir/"figure1_native_stage_local_trace.png",dpi=180); plt.close(fig)

    q=piv[piv.configuration.isin(order)&piv.dataset.isin(["8i","Owlii"])]
    fig,ax=plt.subplots(figsize=(12,5))
    for i,cfg in enumerate(order):
        z=q[q.configuration==cfg]; ax.scatter(np.arange(len(z))+i*.08,z.native_r5_gap_db,label=cfg.split("_")[0],s=25)
    ax.set_xticks(range(len(z))); ax.set_xticklabels(z.sequence,rotation=45,ha="right"); ax.set_ylabel("Official Full - B_native (dB)"); ax.legend(); fig.tight_layout(); fig.savefig(a.output_dir/"figure2_native_r5_quality_gap.png",dpi=180); plt.close(fig)

    fig,axs=plt.subplots(2,4,figsize=(16,8),sharey=False)
    for ax,((ds,seq),z) in zip(axs.flat,q.groupby(["dataset","sequence"],sort=False)):
        z=z.set_index("configuration").reindex(order)
        ax.plot(range(4),z.yuv611_B_unpool,"o-",label="B_unpool"); ax.plot(range(4),z.yuv611_B_native,"o-",label="B_native")
        ax.plot(range(4),z.yuv611_Canonical_Base,"o-",label="Canonical Base"); ax.plot(range(4),z.yuv611_Official_Full,"o-",label="Official Full")
        ax.set_xticks(range(4)); ax.set_xticklabels(["8K","4K","2K","1K"]); ax.set_title(f"{ds}/{seq}")
    axs[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(a.output_dir/"figure3_full_resolution_decomposition.png",dpi=180); plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,(name,z) in zip(axs,controls.groupby("comparison",sort=False)):
        for epn,qq in z.groupby("endpoint"):
            ax.scatter(qq.delta_bpp_diagnostic_minus_official,qq.delta_yuv611_diagnostic_minus_official,label=epn)
        ax.axhline(0,color="k",lw=.7); ax.axvline(0,color="k",lw=.7); ax.set_title(name); ax.set_xlabel("diagnostic-official bpp"); ax.set_ylabel("diagnostic-official dB"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(a.output_dir/"figure4_same_lambda_and_family_controls.png",dpi=180); plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,(point,z) in zip(axs,aligned.groupby("point",sort=False)):
        ax.scatter(z.r2_E_D111,z.delta_matched_db,c=z.native_r5_gap_db,cmap="viridis",s=55)
        for _,r in z.iterrows(): ax.annotate(r.sequence,(r.r2_E_D111,r.delta_matched_db),fontsize=7)
        ax.set_title(point); ax.set_xlabel("GT r2 residual energy"); ax.set_ylabel("Base matched-rate delta (dB)")
    fig.tight_layout(); fig.savefig(a.output_dir/"figure5_content_internal_vs_degradation.png",dpi=180); plt.close(fig)

    provenance={"run_info":{"core":core_info,"family":fam_info,"rwtt":rw_info},"status":"PASS"}
    (a.output_dir/"analysis_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")

if __name__ == "__main__": main()
