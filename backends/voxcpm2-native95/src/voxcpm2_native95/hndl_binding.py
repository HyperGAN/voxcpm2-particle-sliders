"""The existing Vox singleton waveform graph on the exact current shared builder."""
from pathlib import Path
from torch import nn
from .contracts import validate_installed_core,file_hash
PROFILE='voxcpm2-current0141-HNDL-singleton-waveform-v1'

def verified_profile(profile=PROFILE):
    if profile!=PROFILE:raise ValueError('new exact Vox HNDL profile required; no legacy fallback')
    validate_installed_core()
    from particle_sliders import fibo_hndl
    builder,identity=fibo_hndl.verified_builder()
    return fibo_hndl.HNDLParticleProjection,builder,dict(profile=PROFILE,shared_runtime=identity,
        graph_sha256=file_hash(Path(__file__).with_name('wave_critic_singleton.hndl')),
        binding_sha256=file_hash(__file__),training_pg_version='0.9.5',penalty_input_domain='continuous')

class WaveCritic(nn.Module):
    score_bound=0
    def __init__(self,samples,condition_width):
        super().__init__();_,builder,identity=verified_profile()
        self.samples,self.condition_width=samples,condition_width;self._identity=identity
        self.graph=builder(Path(__file__).with_name('wave_critic_singleton.hndl').read_text(),
            input_shape={'response':['B',samples],'condition':['B',condition_width]},output_shape=['B',1],parameters={'samples':samples})
    def forward(self,response,condition):
        if response.ndim!=2 or response.shape[1]!=self.samples or condition.shape!=(response.shape[0],self.condition_width):raise ValueError('native waveform/condition geometry differs')
        return self.graph(response=response,condition=condition).squeeze(-1)
