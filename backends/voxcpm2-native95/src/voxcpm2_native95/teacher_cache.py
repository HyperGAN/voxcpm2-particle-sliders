"""Persist original native teachers/conditions once; exact restore never rerenders."""
from pathlib import Path
import os
import torch
from .contracts import SOURCE_PROFILE,architecture,file_hash
from .data import read_manifest,native_row,conditions
from .waveform import Prefix,WaveRow,PreparedWave,validate_prefix

def _prefix(p):return {k:getattr(p,k).detach().cpu().clone()for k in('tokens','feats','text_mask','audio_mask')}
def _row(p):
    r=p.row
    return dict(neutral_prefix=_prefix(r.neutral),positive_prefix=_prefix(r.positive),noise=r.noise.detach().cpu().clone(),
        steps=r.steps,guidance=r.guidance,sway=r.sway,cfg_zero_star=r.cfg_zero_star,scale=r.scale,
        target=p.target.detach().cpu().clone(),neutral=p.neutral.detach().cpu().clone())
def _restore(value,model):
    device=next(model.parameters()).device
    def prefix(k):
        p=Prefix(**{n:t.to(device)for n,t in value[k].items()});validate_prefix(model,p);return p
    r=WaveRow(prefix('neutral_prefix'),prefix('positive_prefix'),value['noise'].to(device),
        **{k:value[k]for k in('steps','guidance','sway','cfg_zero_star','scale')})
    p=PreparedWave(r,value['target'].to(device),value['neutral'].to(device))
    if p.target.shape!=p.neutral.shape or p.target.ndim!=2 or p.target.shape[0]!=1 or not torch.isfinite(p.target).all()or not torch.isfinite(p.neutral).all():raise ValueError('invalid saved native teacher')
    return p
def prepare(native,options,*,resume):
    path=Path(options['teacher_cache']);document=read_manifest(options['training_file'])
    # Authenticate reference/data bytes, not merely record IDs or filenames.
    records=document['rows'];identity=dict(source_profile=SOURCE_PROFILE,model=native.model_identity,
        architecture=architecture(options),manifest_sha256=file_hash(options['training_file']),
        sampler_sources={name:file_hash(Path(__file__).with_name(name))for name in('waveform.py','data.py','artifacts.py','teacher_cache.py')},
        references={p:file_hash(p)for r in records for p in(r.get('neutral_reference'),r.get('positive_reference'))if p})
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('schema')!='voxcpm2-native95-frozen-teachers-v1'or saved.get('identity')!=identity:raise ValueError('original teacher cache identity differs; never retag/rebuild')
        rows=[_restore(p,native.model)for p in saved['rows']];cond=saved['conditions'].to(next(native.network.parameters()).device)
        if len(rows)<2 or cond.ndim!=2 or cond.shape[0]!=len(rows)or not torch.isfinite(cond).all():raise ValueError('frozen cache rows/conditions differ')
        return rows,cond
    if resume:raise ValueError('exact resume requires its original durable native teacher cache; no repeated forwards')
    rows=[native.prepare(native_row(native.model,r,frames=options.get('frames',4),steps=options.get('euler_steps',10)))for r in records if r['split']=='train']
    if len(rows)<2:raise ValueError('at least two matched native train rows required')
    cond=conditions(native.model,rows)
    saved=dict(schema='voxcpm2-native95-frozen-teachers-v1',identity=identity,rows=[_row(p)for p in rows],conditions=cond.detach().cpu().clone())
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive partial plus atomic no-overwrite publication. A failed partial is
    # retained and refuses a blind retry; it is never treated as a complete cache.
    partial=path.with_name(path.name+'.partial')
    with partial.open('xb')as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
    os.link(partial,path);partial.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return rows,cond
