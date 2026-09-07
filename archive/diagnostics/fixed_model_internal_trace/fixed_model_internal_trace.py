#!/usr/bin/env python3
"""Fixed-model native stage and canonical Base trace on external blocks."""

import argparse, csv, hashlib, json, os, shlex, sys, time
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5, read_ply_ascii, write_ply_ascii
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.prefix import FrozenUnicornPrefix
from scalable_attribute.canonical.scalable_model import load_frozen_base
from scripts.scalable_attribute.canonical.evaluate_scalable_formal import metric, reconstruction_rgb, sparse_max_difference


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--external-manifest',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--gpcc-binary',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-blocks',type=int,default=0)
    return p.parse_args()


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def stats(prefix, sparse, row):
    f=sparse.F.float()
    row[prefix+'_mean']=f.mean().item(); row[prefix+'_std']=f.std(unbiased=False).item()
    row[prefix+'_rms']=f.square().mean().sqrt().item()
    row[prefix+'_l2']=f.norm().item(); row[prefix+'_zero_fraction']=(f==0).float().mean().item()


def aligned_quality(rec, gt):
    if list(rec.tensor_stride)!=list(gt.tensor_stride) or not torch.equal(rec.C,gt.C):
        raise RuntimeError('stage-local support/stride mismatch')
    mse=(rec.F-gt.F).square().mean(0)
    psnr=-10*torch.log10(mse)
    return {**{'mse_'+c:mse[i].item() for i,c in enumerate('YUV')},
            **{'psnr_'+c:psnr[i].item() for i,c in enumerate('YUV')},
            'yuv611':((6*psnr[0]+psnr[1]+psnr[2])/8).item()}


def rows_write(path, rows):
    if not rows:return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys:keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def inputs(path):
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        fields=line.split('\t')
        dataset,sequence,ply=fields[:3]
        kind=fields[3] if len(fields)>3 else 'ply'
        if kind=='h5':
            coords,rgb=read_h5(ply)
            c=coords.astype('int32');c-=c.min(0)
            yield dataset,sequence,0,ply,c,rgb.astype('uint8')
            continue
        coords,rgb=read_ply_ascii(ply)
        parts=kdtree_partition(np.column_stack((coords,rgb)),max_num=100000)
        for i,part in enumerate(parts):
            c=part[:,:3].astype('int32'); c-=c.min(0)
            yield dataset,sequence,i,ply,c,part[:,3:6].astype('uint8')


def kdtree_partition(points, max_num):
    """Exact logic of data_utils.attribute.partition without optional imports."""
    parts=[]
    def split(data):
        if len(data)<=max_num:
            parts.append(data);return
        variances=(np.var(data[:,0]),np.var(data[:,1]),np.var(data[:,2]))
        axis=variances.index(max(variances))
        ordered=data[np.lexsort(data.T[axis,None])]
        middle=len(data)//2
        split(ordered[:middle]);split(ordered[middle:])
    split(points)
    return parts


def main():
    a=args_parser(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False)
    cfg=json.load(open(a.config)); configs=cfg['configurations']
    json.dump({**vars(a),'configurations':configs},open(out/'resolved_args.json','w'),indent=2)
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    os.symlink(os.path.abspath(a.gpcc_binary),out/'tmc3_v21'); os.chdir(out)
    all_inputs=list(inputs(a.external_manifest))
    if a.max_blocks:all_inputs=all_inputs[:a.max_blocks]
    trace=[]; endpoints=[]
    for conf in configs:
        released=conf['released_checkpoint']; base_ckpt=conf.get('base_checkpoint')
        if sha(released)!=conf['released_sha256']:
            raise RuntimeError('released checkpoint hash mismatch: '+conf['id'])
        if base_ckpt and sha(base_ckpt)!=conf['base_sha256']:
            raise RuntimeError('Base checkpoint hash mismatch: '+conf['id'])
        if base_ckpt:
            state=torch.load(base_ckpt,map_location='cpu')
            wrapper=CanonicalBaseModel(released,BaseSynthesisConfig(**state['config'])).cuda().eval()
            load_frozen_base(wrapper,base_ckpt,released,conf['lambda'])
            prefix=wrapper.prefix
        else:
            wrapper=None; prefix=FrozenUnicornPrefix(released).cuda().eval()
        model=prefix.model; emb=model.embedder(conf['lambda'],device='cuda')
        for dataset,sequence,block,source,coords,rgb in all_inputs:
            yuv=rgb2yuv(rgb.astype('float32'),out_range=1).astype('float32')
            bc,bf=ME.utils.sparse_collate([coords],[yuv])
            attr=ME.SparseTensor(features=bf,coordinates=bc,tensor_stride=1,device='cuda')
            gt=[attr]
            for pooling in model.pooling_list:gt.append(pooling(gt[-1]))
            encoded,xlow,xlow_bits=model(attr,training=False,lmb=conf['lambda'],encode=True)
            bits=[len(x['strings'])*8 for x in encoded]
            x0=ME.SparseTensor(features=torch.zeros_like(attr.F),coordinate_map_key=attr.coordinate_map_key,coordinate_manager=attr.coordinate_manager,device=attr.device)
            common={'configuration':conf['id'],'point':conf['point'],'profile':conf['profile'],'lambda':conf['lambda'],'dataset':dataset,'sequence':sequence,'block':block,'source':source,'points':len(attr),'released_sha256':conf['released_sha256'],'base_sha256':conf.get('base_sha256','')}
            q=aligned_quality(xlow,gt[5]); row={**common,'stage':'x_low','stage_index':0,'stage_bits':int(xlow_bits),'cumulative_bits':int(xlow_bits),**q}; stats('x',xlow,row); trace.append(row)
            states=[]
            for n in range(1,6):
                x,f,d=model.decode(x0,xlow,encoded,lmb=conf['lambda'],max_residual_stages=n,return_state=True)
                states.append((x,f,d)); q=aligned_quality(x,gt[5-n])
                row={**common,'stage':'r'+str(n),'stage_index':n,'stage_bits':bits[n-1],'cumulative_bits':int(xlow_bits+sum(bits[:n])),**q}
                stats('x',x,row);stats('f',f,row);stats('d',d,row);trace.append(row)
            full=states[4][0]
            if sparse_max_difference(encoded[-1]['x_out'],full,'full hard')!=0:raise RuntimeError('hard Full mismatch')
            x4,f4,d4=states[3]; pstate=prefix._complete_state(x4,f4,d4)
            zero=type(pstate.f5p)(features=torch.zeros_like(pstate.f5p.F),coordinate_map_key=pstate.f5p.coordinate_map_key,coordinate_manager=pstate.f5p.coordinate_manager)
            _,delta=prefix.synthesize(pstate.f5p,zero); native=pstate.x5p+delta
            candidates=[('B_unpool',pstate.x5p),('B_native',native),('Official_Full',full)]
            if wrapper:
                result=wrapper.reconstruct_from_state(pstate); candidates.insert(2,('Canonical_Base',result['Base']))
                corr={**common,'stage':'base_features'};stats('f4',pstate.f4,corr);stats('f5p',pstate.f5p,corr);stats('cB',result['c_B'],corr);stats('FB',result['F_B'],corr);trace.append(corr)
            tmp=out/'metric_tmp'/conf['id']/sequence/str(block);tmp.mkdir(parents=True,exist_ok=True)
            gtpath=tmp/'gt.ply';write_ply_ascii(str(gtpath),coords,rgb)
            for name,rec in candidates:
                recpath=tmp/(name+'.ply');write_ply_ascii(str(recpath),rec.C[:,1:].cpu().numpy(),reconstruction_rgb(rec))
                qq=metric(str(gtpath),str(recpath)); endpoints.append({**common,'endpoint':name,'physical_bits':int(xlow_bits+sum(bits[:4]) if name!='Official_Full' else xlow_bits+sum(bits)),'physical_bpp':(xlow_bits+sum(bits[:4]) if name!='Official_Full' else xlow_bits+sum(bits))/len(attr),'x_low_bits':int(xlow_bits),**{'r%d_bits'%(i+1):bits[i] for i in range(5)},**qq})
            rows_write(out/'internal_trace_blocks.csv',trace);rows_write(out/'full_resolution_blocks.csv',endpoints)
            print(conf['id'],sequence,block,flush=True)
    json.dump({'status':'PASS','released':{c['id']:{'path':c['released_checkpoint'],'sha256':sha(c['released_checkpoint'])} for c in configs},'bases':{c['id']:{'path':c.get('base_checkpoint'),'sha256':sha(c['base_checkpoint']) if c.get('base_checkpoint') else None} for c in configs}},open(out/'run_info.json','w'),indent=2)


if __name__=='__main__':main()
