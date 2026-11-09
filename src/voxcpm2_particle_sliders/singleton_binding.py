"""Fresh singleton-spatial critic cohort over the unchanged verified runtime.

This binds the reviewed 6872683 CPU representation proof, not waveform admission.
No existing locked HNDL plan, old weights or optimizer checkpoint is converted.
"""
import hashlib
import json
from pathlib import Path

from torch import nn
from . import hndl_binding as original

PROFILE = "voxcpm2-hypergan-hndl-singleton-v1"
PROOF_COMMIT = "6872683b15cfc267e725acdce59c5772ebf9f233"
GRAPH_SHA256 = "6e42b438f5bde89c435331841ad1454045dff9c540f0a56b096d6ebc4ef3dc68"
ROOT = Path(__file__).parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_runtime():
    required = json.loads((ROOT / "singleton_profile.json").read_text())
    if (required["profile"] != PROFILE or required["proof_commit"] != PROOF_COMMIT
            or required["base_vox_profile"] != original.PROFILE
            or required["base_binding_sha256"] != _sha(original.__file__)
            or required["base_manifest_sha256"] != _sha(ROOT / "hndl_profile.json")):
        raise ValueError("singleton candidate base/proof source identity differs")
    graph_sha = _sha(ROOT / "wave_critic_singleton.hndl")
    proof_graph = required["proof_sources"]["analysis/voxcpm2_waveform_audit/wave_critic_singleton.hndl"]["sha256"]
    if graph_sha != GRAPH_SHA256 or graph_sha != required["graph_sha256"] or graph_sha != proof_graph:
        raise ValueError("singleton candidate graph differs from reviewed proof")
    projection, builder, base = original.verified_runtime()
    identity = dict(profile=PROFILE, proof_commit=PROOF_COMMIT,
        base_runtime=base, binding_sha256=_sha(__file__),
        manifest_sha256=_sha(ROOT / "singleton_profile.json"),
        waveform_critic_sha256=graph_sha,
        kernel_mapping="W2[out,in,0,k]=W1[out,in,k]",
        game_boundary="rank2 native waveform unchanged", production_admitted=False,
        qualification="CPU singleton representation only; CUDA/native/307200/cadence unqualified")
    return projection, builder, identity


def verified_profile(profile):
    if profile == PROFILE:
        return verified_runtime()
    if profile == original.PROFILE:
        return original.verified_runtime()
    raise ValueError("unknown modern Vox network profile")


def engine_options(profile):
    verified_profile(profile)
    return original.engine_options()  # Still release0.9.4/auto; useful waveforms refuse.


class HNDLSingletonWaveCritic(nn.Module):
    """Fresh real Conv2d graph; only critic-internal height is singleton."""
    score_bound = 0

    def __init__(self, samples, condition_width):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (samples, condition_width)):
            raise ValueError("positive native sample and condition dimensions required")
        _, builder, identity = verified_runtime()
        self.samples, self.condition_width = samples, condition_width
        self.graph = builder((ROOT / "wave_critic_singleton.hndl").read_text(),
            input_shape={"response": ["B", samples], "condition": ["B", condition_width]},
            output_shape=["B", 1], parameters={"samples": samples})
        for name, inputs in (("audio0", 1), ("audio2", 16)):
            layer = self.graph[name]
            if (not isinstance(layer, nn.Conv2d) or layer.weight.ndim != 4
                    or layer.in_channels != inputs or layer.out_channels != 16
                    or layer.kernel_size != (1, 15) or layer.stride != (1, 4)
                    or layer.padding != (0, 7) or layer.dilation != (1, 1)
                    or layer.groups != 1 or layer.padding_mode != "zeros"):
                raise ValueError("resolved singleton convolution geometry differs")
        self._identity = identity

    def extra_repr(self):
        return json.dumps(dict(samples=self.samples, condition_width=self.condition_width,
                               **self._identity), sort_keys=True)

    def forward(self, response, condition):
        if (response.ndim != 2 or response.shape[1] != self.samples or condition.ndim != 2
                or condition.shape != (response.shape[0], self.condition_width)):
            raise ValueError("native waveform/condition dimensions differ")
        return self.graph(response=response, condition=condition).squeeze(-1)
