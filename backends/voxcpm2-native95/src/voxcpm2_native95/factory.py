"""Actual full-waveform response lifecycle; core owns all GAN/optimizer math."""
from copy import deepcopy
from pathlib import Path
import torch
from particle_sliders.inference import InferenceBackend,InferenceBinding,native_identity
from particle_sliders.lifecycle import Binding,Capabilities,read_packet
from particle_sliders.pg095 import ConditionalPG095,engine_options
from particle_sliders.response import ResponseBackend,ResponseCapabilities
from .contracts import (BACKEND,SOURCE_PROFILE,INITIALIZATION_POLICY,PROTOCOL,WEIGHT_POLICY,DEFAULTS,
    architecture,digest,file_hash,samples,rate_overrides,validate_training,validate_installed_core)
from .initialization import Construction,construction_for,initialize_owned,validate_receipt
from .runtime import load_native,require_owner
from .hndl_binding import WaveCritic
from .teacher_cache import prepare as prepare_teachers

class WaveResponse:
    family='voxcpm2_native95_full_waveform'
    def __init__(self,native):
        self.runtime=native;self.model,self.slider,self.network=native.model,native.slider,native.network
        self.model_identity=deepcopy(native.model_identity)
        self.model_identity['current_product_sources']={p.name:file_hash(p)for p in Path(__file__).parent.iterdir()if p.suffix in('.py','.hndl','.json')}
        self.fresh_initialization=None
    def prepare(self,row):return self.runtime.prepare(row)
    def generated(self,row):return self.runtime.generated(row)
    def sample(self,row,*,scale=1.):return self.runtime._run(row.row,row.row.neutral,scale).detach().clone()
    def metadata(self):return dict(self.runtime.metadata(),source_profile=SOURCE_PROFILE,weight_policy=WEIGHT_POLICY,
        fresh_initialization=self.fresh_initialization,penalty_input_domain='continuous')
    def close(self):return self.slider.close()
    def response(self,*,strength=1.,seed=0,**request):
        from .finished_audio import generate
        return generate(self.runtime,strength=strength,seed=seed,**request).waveform

def capabilities(options):
    return ResponseCapabilities('routed_particle_gan','audio','native_mono_waveform','continuous_vector',
        (1,samples(options)),'0.9.5','release-0.9.5',penalty_input_domain='continuous',base_precisions=('float32','bfloat16'),
        qualification='New full native waveform Source; CPU/pretrained/quality qualification pending')

def validate(options):validate_training(options);validate_installed_core()
def preflight(options,request,staged):
    c=staged.get('construction')
    if (set(staged)!={'construction'}or not isinstance(c,Construction)or c.options_sha256!=digest(options)
        or c.mode not in('fresh','resume')):raise ValueError('explicit fresh/resume intent before native/optimizer construction required')
    if request['native']:raise ValueError('training uses matched native manifest only')

def prepare_training(options,*,construction=None):
    validate(options)
    if not isinstance(construction,Construction)or construction.options_sha256!=digest(options):raise ValueError('use open_training with explicit fresh/resume intent')
    native=load_native(options)
    try:
        current=WaveResponse(native);rows,cond=prepare_teachers(native,options,resume=construction.mode=='resume')
        if any(tuple(p.target.shape)!=(1,samples(options))for p in rows):raise ValueError('actual full native waveform differs from declared geometry')
        device=next(current.network.parameters()).device;require_owner(device)
        devices=[]if device.type=='cpu'else[device.index]
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get('seed',7),devices)
            critic=WaveCritic(samples(options),cond.shape[1]).to(device=device,dtype=torch.float32)
            receipt=initialize_owned(native.slider,critic,construction,options)
            current.fresh_initialization=receipt.as_dict()
        game=ConditionalPG095(current,rows,critic,conditions=cond,particles=current.slider.particles,
            response_domain='continuous_vector',total_steps=options.get('steps',2),batch_size=options.get('batch',2),seed=options.get('seed',7),
            recipe_overrides=rate_overrides(options),**engine_options(options))
        spec=INFERENCE.specification(current,model_identity=current.model_identity,protocol=PROTOCOL,
            configuration=dict(architecture=architecture(options),source_profile=SOURCE_PROFILE,initialization_policy=INITIALIZATION_POLICY,
                weight_policy=WEIGHT_POLICY,initialization=receipt.as_dict()))
        return Binding(game,current.close,inference=spec)
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([native.slider.close],primary=primary);raise

def validate_inference(options,spec):
    validate_installed_core()
    if set(options)-{'model_dir','revision','device'}:raise ValueError('RAW inference accepts local model/revision/device only')
    c=spec.configuration
    if (set(c)!={'architecture','source_profile','initialization_policy','weight_policy','initialization'}
        or c['source_profile']!=SOURCE_PROFILE or c['initialization_policy']!=INITIALIZATION_POLICY
        or c['weight_policy']!=WEIGHT_POLICY or spec.identity['protocol']!=PROTOCOL):raise ValueError('exact new Vox095 RAW artifact required; no historical packet migration')
    validate_receipt(c['initialization'],cohort=c['initialization'].get('cohort_id'));require_owner(options.get('device','cpu'))
def preflight_inference(options,request,spec):
    from .finished_audio import validate_request
    validate_request(strength=request['strength'],output_type=request['output_type'],**request['native'])
def prepare_inference(options,spec):
    validate_inference(options,spec);native=load_native(dict(options,**spec.configuration['architecture']))
    try:
        current=WaveResponse(native);current.fresh_initialization=spec.configuration['initialization']
        def render(**request):
            from .finished_audio import generate
            return generate(native,**request)
        return InferenceBinding(current.network,native_identity(current,model_identity=current.model_identity,protocol=PROTOCOL),
            current.close,response=current.response,render=render)
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([native.slider.close],primary=primary);raise

# Configuration replaces the descriptor with exact native decoder geometry before
# allocation; the initial positive placeholder never supplies training responses.
TRAINING=ResponseBackend(BACKEND,ResponseCapabilities('routed_particle_gan','audio','native_mono_waveform','continuous_vector',
    (1,1),'0.9.5','release-0.9.5',penalty_input_domain='continuous'),prepare_training,validate,
    configure_capabilities=capabilities,preflight=preflight)
INFERENCE=InferenceBackend(BACKEND,Capabilities('routed_particle_gan',render=True,render_outputs=('audio',),
    operations=('response','render'),qualification='Actual native AudioVAE waveform Source; qualification pending'),
    prepare_inference,validate_inference,preflight=preflight_inference)

def open_training(config,*,resume=None,**request):
    validate(config.options)
    packet=deepcopy(resume)if isinstance(resume,dict)else read_packet(resume)if resume is not None else None
    construction=construction_for(config,packet)
    if construction.mode=='resume'and not Path(config.options['teacher_cache']).is_file():
        raise ValueError('exact resume requires its original durable native teacher cache; no repeated forwards')
    session=TRAINING.open(config,operation='resume'if packet is not None else'train',construction=construction,**request)
    try:
        if packet is not None:session.restore(packet)
        return session
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([session.close],primary=primary);raise
