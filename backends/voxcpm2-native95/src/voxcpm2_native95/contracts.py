"""New explicit PG095 waveform cohort; historical packets are never migrated."""
import hashlib,json,math
from pathlib import Path
from math import prod

CORE_REVISION='41f6066f2d2644902670d5eb736de052e1a70553'
BACKEND='voxcpm2_native95'
SOURCE_PROFILE='voxcpm2-core0141-pg095-full-waveform-hndl-v1'
INITIALIZATION_POLICY='voxcpm2-host-RMS-QR-zero-up-bank-R2-std1-v1'
PROTOCOL='voxcpm2-native48k-fixed-noise-full-waveform-RAW-v1'
WEIGHT_POLICY='raw-no-weight-EMA'
DEFAULTS=dict(hosts=['dit'],rank=8,alpha=8.,particles=128,frames=4,euler_steps=10)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rate_overrides(options):
    result={k:options[k]for k in('lr','d_lr_mult','prior_lr_mult')if k in options}
    if any(type(v)not in(int,float)or not math.isfinite(v)or v<=0 for v in result.values()):raise ValueError('finite positive non-Boolean common rates required')
    return result
def architecture(options):return {k:options.get(k,v)for k,v in DEFAULTS.items()}
def samples(options):
    config=json.loads((Path(options['model_dir'])/'config.json').read_text())
    patch=config.get('patch_size',4);rates=(config.get('audio_vae_config')or{}).get('decoder_rates',[8,6,5,2,2,2])
    if type(patch)is not int or patch<1 or not rates or any(type(x)is not int or x<1 for x in rates):raise ValueError('native decoder geometry required')
    return architecture(options)['frames']*patch*prod(rates)
def validate_profile(options):
    # Refuse BEFORE filesystem, model, native code or optimizer allocation.
    if options.get('ema_decay')is not None or options.get('ema')is not None:raise ValueError('Vox095 refuses weight EMA')
    if (options.get('source_profile')!=SOURCE_PROFILE or options.get('initialization_policy')!=INITIALIZATION_POLICY
        or options.get('pg_version')!='0.9.5'or options.get('pg_source_profile')!='release-0.9.5'
        or options.get('penalty_input_domain')!='continuous'):raise ValueError('explicit new PG095/full-waveform/continuous profile required')
    rate_overrides(options)
def validate_installed_core():
    from importlib import metadata
    lock=json.loads(Path(__file__).with_name('current_core.json').read_text());d=metadata.distribution('particle-sliders-core')
    if d.version!=lock['version']or lock['commit']!=CORE_REVISION:raise ValueError('exact current core source required')
    for name,sha in lock['checked_api_files'].items():
        if file_hash(d.locate_file('particle_sliders/'+name))!=sha:raise ValueError('current core Source differs:'+name)
def validate_training(options):
    validate_profile(options)
    allowed=set(DEFAULTS)|{'model_dir','revision','training_file','teacher_cache','device','steps','batch','seed','ema_decay',
        'source_profile','initialization_policy','pg_version','pg_source_profile','penalty_input_domain','lr','d_lr_mult','prior_lr_mult'}
    if set(options)-allowed:raise ValueError('unknown Vox095 option; use literal common rates, no nested recipe')
    a=architecture(options)
    if a['rank']!=8 or type(a['rank'])is not int or a['particles']!=128 or type(a['particles'])is not int or a['alpha']!=8.:raise ValueError('unchanged HNDL rank8/alpha8/128x4 required')
    if not isinstance(a['hosts'],list)or not a['hosts']or len(set(a['hosts']))!=len(a['hosts'])or set(a['hosts'])-{'dit','lm'}:raise ValueError('explicit dit/lm hosts required')
    for k,default,low,high in(('frames',4,1,64),('euler_steps',10,2,32),('steps',2,1,2**31),('batch',2,2,512),('seed',7,0,2**63-101)):
        v=options.get(k,default)
        if type(v)is not int or not low<=v<=high:raise ValueError('invalid '+k)
    revision=options.get('revision')
    if not isinstance(revision,str)or len(revision)!=40 or any(c not in'0123456789abcdef'for c in revision):raise ValueError('resolved original model revision required')
    if not isinstance(options.get('model_dir'),str)or not Path(options['model_dir']).is_dir():raise ValueError('offline model_dir required')
    config=json.loads((Path(options['model_dir'])/'config.json').read_text())
    if config.get('architecture','').lower()!='voxcpm2':raise ValueError('original VoxCPM2 architecture required')
    if config.get('dtype','bfloat16')not in('float32','bfloat16'):raise ValueError('unquantized native FP32/BF16 configuration required')
    if not isinstance(options.get('training_file'),str)or not Path(options['training_file']).is_file():raise ValueError('matched local training manifest required')
    if not isinstance(options.get('teacher_cache'),str)or not Path(options['teacher_cache']).is_absolute():raise ValueError('explicit owned durable teacher_cache path required')
    from .runtime import require_owner
    require_owner(options.get('device','cpu'))
    samples(options)
