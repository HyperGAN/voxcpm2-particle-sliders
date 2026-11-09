"""Vox-local shape/identity bindings over the genuine shared HyperGAN/HNDL graph.

No routed network or GAN equations are implemented here. The parent profile's
verified_builder owns the exact HNDL/HyperGAN runtime source closure.
"""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

PROFILE = "voxcpm2-hypergan-hndl-v1"
LEGACY_PROFILE = "legacy"
ROOT = Path(__file__).parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_core_closure(root, expected):
    """Bind every installed core source/data file, excluding Python caches."""
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and "__pycache__" not in path.relative_to(root).parts}
    if actual != set(expected):
        raise ValueError("Vox installed shared core source inventory differs")
    for name, digest in expected.items():
        if _sha(root / name) != digest:
            raise ValueError("Vox installed shared core source differs: " + name)
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def verified_runtime():
    import particle_sliders
    from particle_sliders import fibo_hndl
    required = json.loads((ROOT / "hndl_profile.json").read_text())
    if particle_sliders.__version__ != required["core_version"]:
        raise ValueError("Vox modern HNDL profile requires the separately reviewed core0.12 candidate; legacy pins are not a migration")
    source = Path(inspect.getsourcefile(fibo_hndl))
    if _sha(source) != required["shared_binding_sha256"]:
        raise ValueError("Vox shared HNDL binding source differs from its explicit candidate profile")
    closure_sha = _verify_core_closure(Path(particle_sliders.__file__).parent,
                                       required["shared_core_sources_sha256"])
    if closure_sha != required["shared_core_source_closure_sha256"]:
        raise ValueError("Vox shared core closure manifest differs")
    builder, shared_identity = fibo_hndl.verified_builder()
    if (shared_identity["profile"] != required["shared_network_profile"]
            or shared_identity["projection_sha256"] != required["shared_projection_sha256"]
            or shared_identity["runtime_sources_sha256"] != required["shared_runtime_manifest_sha256"]
            or _sha(ROOT / "wave_critic.hndl") != required["waveform_critic_sha256"]):
        raise ValueError("Vox shared network profile differs")
    identity = dict(profile=PROFILE, shared_runtime=shared_identity,
                    shared_binding_sha256=_sha(source),
                    vox_binding_sha256=_sha(__file__),
                    waveform_critic_sha256=_sha(ROOT / "wave_critic.hndl"),
                    candidate_core_commit=required["candidate_core_commit"],
                    shared_core_source_closure_sha256=closure_sha,
                    declared_pg_profile={k:required[k] for k in ("pg_version", "pg_source_profile", "penalty_input_domain")},
                    qualification="CPU graph software only; no wide-waveform/pretrained admission")
    return fibo_hndl.HNDLParticleProjection, builder, identity


def engine_options():
    """Pass only the exact reviewed profile through the owner's public validator.

    The current candidate declares release0.9.4/auto and still refuses wide
    waveforms. Changing a JSON label cannot invent an upstream capability.
    """
    verified_runtime()
    required = json.loads((ROOT / "hndl_profile.json").read_text())
    from particle_sliders.pg094 import engine_options as shared_engine_options
    return shared_engine_options({key:required[key]
        for key in ("pg_version", "pg_source_profile", "penalty_input_domain")})


class HNDLWaveCritic(nn.Module):
    """Same native 1D Conv/SiLU/mean/FiLM/raw-score graph as WaveCritic.

    `[B,samples] -> [B,1,samples]` is a local Conv1d channel view inside the
    critic. ConditionalPG09 still receives the genuine waveform vector.
    """
    score_bound = 0

    def __init__(self, samples, condition_width):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (samples, condition_width)):
            raise ValueError("positive native sample and condition dimensions required")
        _, builder, identity = verified_runtime()
        self.samples, self.condition_width = samples, condition_width
        self.graph = builder((ROOT / "wave_critic.hndl").read_text(),
            input_shape={"response": ["B", samples], "condition": ["B", condition_width]},
            output_shape=["B", 1], parameters={"samples": samples})
        self._identity = identity

    def extra_repr(self):
        return json.dumps(dict(samples=self.samples, condition_width=self.condition_width,
                               **self._identity), sort_keys=True)

    def forward(self, response, condition):
        if (response.ndim != 2 or response.shape[1] != self.samples or condition.ndim != 2
                or condition.shape != (response.shape[0], self.condition_width)):
            raise ValueError("native waveform/condition dimensions differ")
        return self.graph(response=response, condition=condition).squeeze(-1)
