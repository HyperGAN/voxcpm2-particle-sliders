"""NEW integration checks; Root executes only after actual CPU admission.

Uses the real pinned VoxCPM2/AudioVAE classes at tiny random sizes. No pretrained
loader/network/GPU is needed; this is not speech quality or native fit evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from particle_sliders.lifecycle import SessionConfig,CheckpointError
from particle_sliders.inference import open_inference
from voxcpm2_native95 import factory as current
from voxcpm2_native95.contracts import BACKEND,SOURCE_PROFILE,INITIALIZATION_POLICY,validate_profile,rate_overrides,samples
from voxcpm2_native95.adapter import ParticleAdapter
from voxcpm2_native95.waveform import VoxCPM2WaveBackend,WaveRow,make_prefix,fixed_noise

def config(tmp_path):
    (tmp_path/'model').mkdir();(tmp_path/'model/config.json').write_text(json.dumps(dict(architecture='voxcpm2',patch_size=2,audio_vae_config=dict(decoder_rates=[3,2,2]))))
    (tmp_path/'rows.json').write_text('matched tiny native manifest')
    return SessionConfig(BACKEND,'new-Vox095-tiny-native-integration',model_dir=str(tmp_path/'model'),revision='a'*40,
        training_file=str(tmp_path/'rows.json'),teacher_cache=str(tmp_path/'native-teachers.pt'),device='cpu',frames=8,euler_steps=3,
        steps=3,batch=2,seed=7,ema_decay=None,lr=.012,d_lr_mult=.5,prior_lr_mult=.75,
        source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous')

def install_tiny(monkeypatch,model_factory):
    def load(options):
        with torch.random.fork_rng(devices=[]):torch.default_generator.manual_seed(219);model=model_factory()
        slider=ParticleAdapter(model,seed=options.get('seed',7))
        return VoxCPM2WaveBackend(model,slider,model_identity={'qualification':'real official tiny random VoxCPM2; no pretrained','revision':'a'*40})
    monkeypatch.setattr(current,'load_native',load)
    import voxcpm2_native95.teacher_cache as cache
    monkeypatch.setattr(cache,'read_manifest',lambda path:dict(rows=[dict(id='one',split='train'),dict(id='two',split='train')]))
    def row(model,record,*,frames,steps):
        seed=11 if record['id']=='one'else 22
        return WaveRow(make_prefix(model,'one two'),make_prefix(model,'one two three'),fixed_noise(model,frames,seed),steps=steps)
    monkeypatch.setattr(cache,'native_row',row)
    return load

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)is type(b)and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def test_profile_rates_and_EMA_refuse_before_allocation(tmp_path,monkeypatch):
    c=config(tmp_path)
    monkeypatch.setattr(current,'load_native',lambda *a,**k:pytest.fail('loader before refusal'))
    for value in (True,0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):rate_overrides({'lr':value})
    assert rate_overrides({})=={} and rate_overrides(dict(lr=.001,d_lr_mult=18.,prior_lr_mult=30.))==dict(lr=.001,d_lr_mult=18.,prior_lr_mult=30.)
    for change in (dict(ema_decay=.9),dict(penalty_input_domain='auto'),dict(source_profile='legacy')):
        with pytest.raises(ValueError):validate_profile(dict(c.options,**change))
    with pytest.raises(ValueError):current.open_training(SessionConfig('historical','old',**c.options))

def test_unadmitted_cuda_refuses_without_probe(monkeypatch):
    from voxcpm2_native95.runtime import require_owner
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:pytest.fail('unowned CUDA probe'))
    with pytest.raises(ValueError,match='OwnedCUDAAdmission'):require_owner('cuda:0')
    assert require_owner('cpu')is None

def test_scoped_initialization_preserves_zero_up_backbone_and_alias(tmp_path,monkeypatch,model_factory):
    c=config(tmp_path);load=install_tiny(monkeypatch,model_factory);native=load(c.options)
    from voxcpm2_native95.hndl_binding import WaveCritic
    from voxcpm2_native95.initialization import construction_for,initialize_owned,policies
    from hndl.operators.linear import Linear as HNDLLinear
    from hndl.operators.conv import Conv2d as HNDLConv2d
    critic=WaveCritic(samples(c.options),22);before=deepcopy(native.model.state_dict());bank=native.slider.particles;ids=[id(p)for p in native.slider.parameters()]
    rng=torch.get_rng_state().clone()
    try:
        # These are the real verified HNDL classes, not plain Linear substitutes.
        leaves={prefix+'.'+name:leaf for prefix,root in (('adapter',native.slider),('critic',critic))
                for name,leaf in root.named_modules()
                if any(p.requires_grad for p in leaf.parameters(recurse=False)) and leaf is not native.slider}
        assert {type(leaf) for leaf in leaves.values()}=={HNDLLinear,HNDLConv2d}
        scopes=policies(native.slider,critic,7)
        for name,leaf in leaves.items():
            assert type(scopes[name].module) is torch.nn.Module
            assert dict(scopes[name].module.named_parameters())==dict(leaf.named_parameters(recurse=False))
            assert all(scopes[name].module._parameters[n] is p for n,p in leaf.named_parameters(recurse=False))
        # A genuinely undeclared subclass refuses before the official helper can
        # mutate any owned value or RNG; the strict shared whitelist stays intact.
        class ForeignLinear(torch.nn.Linear):pass
        native.slider.foreign=ForeignLinear(2,2)
        refused_state=deepcopy(native.slider.state_dict());critic_state=deepcopy(critic.state_dict());refused_rng=torch.get_rng_state().clone()
        with pytest.raises(ValueError,match='undeclared VoxCPM2 affine'):
            initialize_owned(native.slider,critic,construction_for(c,None),c.options)
        equal(refused_state,native.slider.state_dict());equal(critic_state,critic.state_dict())
        assert torch.equal(refused_rng,torch.get_rng_state())
        del native.slider.foreign
        torch.set_rng_state(rng)
        receipt=initialize_owned(native.slider,critic,construction_for(c,None),c.options).as_dict()
        zero_up={id(b.lora_up) for b in native.slider.branches}
        for name,leaf in leaves.items():
            fan_in=leaf.in_features if type(leaf) is HNDLLinear else leaf.weight[0].numel()
            bound=1/fan_in**.5
            for n,p in leaf.named_parameters(recurse=False):
                entry=receipt['components'][name]['parameters'][n]
                law=dict(kind='KEEP') if n=='weight' and id(leaf) in zero_up else dict(kind='Uniform',low=-bound,high=bound)
                assert entry['host_declaration']==law
                if law['kind']=='KEEP':assert entry['protected'] and entry['before_sha256']==entry['after_sha256']
        assert receipt['components']['adapter.bank']['parameters']['particles']['host_declaration']==dict(kind='R2Normal',mean=0.,std=1.)
        assert native.slider.particles is bank and [id(p)for p in native.slider.parameters()]==ids
        for b in native.slider.branches:
            assert torch.count_nonzero(b.lora_up.weight)==0 and torch.count_nonzero(b.lora_down.weight)>0
            assert b._particles is bank
        assert all(torch.equal(v,before[k])for k,v in native.model.state_dict().items())and torch.equal(rng,torch.get_rng_state())
        assert all(not p.requires_grad for p in native.model.parameters())
        # New holders cannot bypass same-original-Parameter repeat protection.
        initialized=deepcopy(native.slider.state_dict());critic_initialized=deepcopy(critic.state_dict())
        from dataclasses import replace
        wrong=deepcopy(receipt)
        wrong['components']['adapter.bank']['parameters']['particles']['host_declaration']['std']=.1
        resume=replace(construction_for(c,None),mode='resume',initialization_json=json.dumps(wrong))
        with pytest.raises(ValueError,match='saved initialization parameter identity'):
            initialize_owned(native.slider,critic,resume,c.options)
        equal(initialized,native.slider.state_dict());equal(critic_initialized,critic.state_dict())
        with pytest.raises(ValueError,match='already initialized parameter'):
            initialize_owned(native.slider,critic,construction_for(c,None),c.options)
        equal(initialized,native.slider.state_dict());equal(critic_initialized,critic.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
    finally:native.slider.close()

def test_full_native_width_exact_resume_cache_and_common_rates(tmp_path,monkeypatch,model_factory):
    c=config(tmp_path);install_tiny(monkeypatch,model_factory);calls=[];original=VoxCPM2WaveBackend.prepare
    def prepare(self,row):calls.append('teacher_pair');return original(self,row)
    monkeypatch.setattr(VoxCPM2WaveBackend,'prepare',prepare)
    with current.open_training(c)as full:
        assert tuple(full.game.response_shape)==(1,192) and full.game.signature['penalty_input_domain']=='continuous'
        assert full.ema.values is None
        assert [g['lr']for g in full.game.opt_g.param_groups]==[.012,.012*.75]
        assert all(g['lr']==.012*.5 for g in full.game.opt_d.param_groups)
        full.train();expected=full.checkpoint()
    assert len(calls)==2 and Path(c.options['teacher_cache']).is_file()
    with current.open_training(c)as first:first.train(stop_after=2);saved=first.checkpoint()
    import voxcpm2_native95.initialization as init
    monkeypatch.setattr(init,'initialize_fresh_components',lambda *a,**k:pytest.fail('fresh init on exact restore'))
    with current.open_training(c,resume=saved)as resumed:
        assert resumed.game.completed_steps==2;resumed.train();equal(expected,resumed.checkpoint())
        artifact=resumed.export_inference(selection='raw')
        with pytest.raises(ValueError):resumed.export_inference(selection='ema')
    assert len(calls)==2  # No teacher-pair replay in fresh cache reuse or exact restore.
    with pytest.raises(CheckpointError):current.open_training(c,resume=artifact)
    assert not torch.cuda.is_initialized()

def test_RAW_native_render_without_teacher_or_initializer(tmp_path,monkeypatch,model_factory):
    c=config(tmp_path);install_tiny(monkeypatch,model_factory)
    with current.open_training(c)as session:artifact=session.export_inference(selection='raw')
    import voxcpm2_native95.initialization as init
    monkeypatch.setattr(init,'initialize_fresh_components',lambda *a,**k:pytest.fail('init in RAW inference'))
    monkeypatch.setattr(VoxCPM2WaveBackend,'prepare',lambda *a,**k:pytest.fail('teacher in native inference'))
    options={k:c.options[k]for k in('model_dir','revision','device')};request=dict(text='one two',frames=8,euler_steps=3)
    rng=torch.get_rng_state().clone()
    with open_inference(artifact,options=options,backend=current.INFERENCE,native_request=request)as session:
        off=session.render(strength=0,seed=11,**request);zero=session.render(strength=1,seed=11,**request)
        assert off.sample_rate==48000 and off.waveform.shape==(1,192)and torch.equal(off.waveform,zero.waveform)
        assert zero.native['selection']=='raw'and zero.native['weight_EMA']is None
        out=tmp_path/'actual-tiny.wav';zero.save(out)
        import soundfile as sf
        info=sf.info(out);assert info.frames==192 and info.samplerate==48000 and info.subtype=='FLOAT'
        with pytest.raises(ValueError):zero.save(out)
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()

def test_missing_resume_cache_refuses_no_teacher_replay(tmp_path,monkeypatch,model_factory):
    c=config(tmp_path);install_tiny(monkeypatch,model_factory)
    with current.open_training(c)as session:saved=session.checkpoint()
    Path(c.options['teacher_cache']).unlink()
    monkeypatch.setattr(VoxCPM2WaveBackend,'prepare',lambda *a,**k:pytest.fail('missing resume cache must not rerender'))
    with pytest.raises(ValueError,match='original durable native teacher cache'):current.open_training(c,resume=saved)
