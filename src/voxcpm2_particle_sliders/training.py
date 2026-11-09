"""No local GAN fork: waveform admission precedes the shared ConditionalPG09 game."""
import torch
from torch import nn
from particle_sliders import ConditionalPG09
from particle_sliders.pg09 import validate_pg09_domain


class WaveCritic(nn.Module):
    """Raw conditional score on native mono waveform coordinates."""
    def __init__(self, condition_width):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(1, 16, 15, stride=4, padding=7), nn.SiLU(),
                                   nn.Conv1d(16, 16, 15, stride=4, padding=7), nn.SiLU())
        self.condition = nn.Linear(condition_width, 32)
        self.score = nn.Linear(16, 1)

    def forward(self, response, condition):
        gain, bias = self.condition(condition).chunk(2, dim=-1)
        h = self.audio(response[:, None]).mean(-1)
        return self.score(torch.nn.functional.silu(h * (1 + gain) + bias)).squeeze(-1)


def make_game(backend, rows, *, conditions, total_steps, batch_size=2, seed=7):
    from .hndl_binding import PROFILE, HNDLWaveCritic
    from .singleton_binding import PROFILE as SINGLETON_PROFILE, HNDLSingletonWaveCritic, engine_options
    modern = backend.slider.network_profile in (PROFILE, SINGLETON_PROFILE)
    options = engine_options(backend.slider.network_profile) if modern else {}
    # Genuine 48kHz waveform vector; do not reshape into image domains or shorten
    # to width<170. Current released core rejects every useful waveform here.
    validate_pg09_domain(tuple(rows[0].target.shape), response_domain="continuous_vector", **options)
    # Import HyperGAN's current public recipe schema as a compatibility witness;
    # ConditionalPG09 below owns the actual released game and optimizer state.
    from hypergan.particlegan_integration import particlegan_recipe_fields
    hypergan_fields = particlegan_recipe_fields()
    if not hypergan_fields:
        raise RuntimeError("HyperGAN develop has no public ParticleGAN recipe schema")
    if backend.slider.network_profile == SINGLETON_PROFILE:
        critic = HNDLSingletonWaveCritic(rows[0].target.shape[1], conditions.shape[1])
    elif backend.slider.network_profile == PROFILE:
        critic = HNDLWaveCritic(rows[0].target.shape[1], conditions.shape[1])
    else:
        critic = WaveCritic(conditions.shape[1])
    critic = critic.to(backend.network.weights[0].device)
    return ConditionalPG09(backend, rows, critic, conditions=conditions,
                           particles=backend.slider.particles, response_domain="continuous_vector",
                           total_steps=total_steps, batch_size=batch_size, seed=seed, **options)
