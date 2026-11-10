"""Actual native 48k mono waveform output; no denoising-loss or decoder proxy."""
from dataclasses import dataclass
from pathlib import Path
import json,math
import torch
from .waveform import waveform,make_prefix,fixed_noise
from .runtime import require_owner

def validate_request(*,text,reference_wav=None,frames=4,euler_steps=10,strength=1.,seed=0,output_type='audio',images=None):
    if images or output_type not in('audio',None):raise ValueError('native mono audio only')
    if not isinstance(text,str)or not text.strip():raise ValueError('nonempty spoken text required')
    if reference_wav is not None and(not isinstance(reference_wav,str)or not Path(reference_wav).is_file()):raise ValueError('local native reference waveform required')
    if type(frames)is not int or not 1<=frames<=64 or type(euler_steps)is not int or not 2<=euler_steps<=32:raise ValueError('declared native frames/Euler budget required')
    if type(seed)is not int or not 0<=seed<2**63-100:raise ValueError('explicit valid seed required')
    if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('native unipolar strength in[0,1] required')

@dataclass(frozen=True)
class AudioOutput:
    waveform:object
    native:dict
    sample_rate:int=48000
    def save(self,path):
        path=Path(path);meta=Path(str(path)+'.native.json')
        if path.exists()or meta.exists():raise ValueError('fresh native audio/receipt paths required')
        # Floating WAV stores the actual samples, without clipping/normalization.
        import soundfile as sf
        with path.open('xb')as stream:sf.write(stream,self.waveform[0].detach().cpu().numpy(),self.sample_rate,format='WAV',subtype='FLOAT')
        with meta.open('x')as stream:json.dump(self.native,stream,indent=2,allow_nan=False);stream.write('\n')

@torch.no_grad()
def generate(native,**request):
    validate_request(**request);device=next(native.model.parameters()).device;require_owner(device)
    devices=[]if device.type=='cpu'else[device.index]
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(request.get('seed',0))
        if devices:torch.cuda.default_generators[devices[0]].manual_seed(request.get('seed',0))
        prefix=make_prefix(native.model,request['text'],reference_wav=request.get('reference_wav'))
        noise=fixed_noise(native.model,request.get('frames',4),request.get('seed',0))
        with native.network.scaled(request.get('strength',1.)):
            audio=waveform(native.model,prefix,noise,steps=request.get('euler_steps',10))
    return AudioOutput(audio.detach().cpu(),dict(schema='voxcpm2-native95-finished-waveform-v1',selection='raw',weight_EMA=None,
        sample_rate=48000,samples=audio.shape[-1],frames=request.get('frames',4),euler_steps=request.get('euler_steps',10),
        strength=request.get('strength',1.),seed=request.get('seed',0),model_identity=native.model_identity,
        actual_resolved_base_dtype=native.model.config.dtype,native_AudioVAE_dtype='float32',adapter_dtype='float32',
        protocol='official full-prefix + Euler + native AudioVAE; fixed duration',pretrained_quality_qualified=False))
