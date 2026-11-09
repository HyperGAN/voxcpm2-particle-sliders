# Vox HyperGAN/HNDL binding: local implementation, final profile pending

The historical prototype remains available at commit
`2047e35b98d221c8a6f59c729f1dad6e17fac4ff`, and the default `legacy` adapter,
checkpoint metadata and dependency pins remain explicit. A new opt-in
`voxcpm2-hypergan-hndl-v1` source-bound cohort is now implemented. It is not a
rename of the prototype and does not claim pretrained or wide-waveform admission.

## Implemented Vox-local pieces

1. `ParticleAdapter(network_profile="voxcpm2-hypergan-hndl-v1")` adopts the actual
   shared `particle_sliders.fibo_hndl.HNDLParticleProjection`. Rank8/alpha8 and one
   128x4 bank match the shared graph's contract. No routed graph or GAN equations
   are copied. The actual HNDL graph executes through the verified HyperGAN
   builder, and metadata records the distinct Vox/shared graph/runtime identities.
2. `HNDLWaveCritic` implements this product's existing Conv1d/SiLU/mean/FiLM/raw
   score network as `wave_critic.hndl`, built by that same verified HyperGAN
   builder. The game still receives genuine `[B,samples]` mono vectors. The local
   critic channel view `[B,1,samples]` is not an image-domain admission workaround.
3. Retained-up distillation needs **no new capture API**. Inspection and executed
   tests established that the shared projection's public `lora_up` module runs
   ordinary PyTorch pre-hooks. Capture now pairs native host inputs with actual
   pre-up activations from this hook, for both legacy and HNDL execution. It
   does not copy routing or read private graph endpoints. Missing hooks fail
   explicitly, and temporary handles unwind on failure.

The owner candidate inspected read-only is core0.12 commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, PG0.9.4, HNDL0.10.0, HyperGAN
develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, shared profile
`hypergan-hndl-v2`. `hndl_profile.json` pins shared binding/graph/runtime manifest
hashes, all 50 installed core source/data/license files and the local waveform graph. This code refuses a different source
closure; it never falls back to the legacy network while retaining a modern label.

## Concrete shared interface request

No mandatory shared API addition blocks the Vox-local implementation now.
The pre-up request is withdrawn because the existing public module hook works.
For public naming, the smallest optional owner change is a generic re-export:

```python
# particle_sliders/routed_hndl.py -- proposed owner-only addition, not applied here
from .fibo_hndl import HNDLParticleProjection, verified_builder

__all__ = ["HNDLParticleProjection", "verified_builder"]
```

This introduces no new network or objective. Root should ask the core owner to
confirm that Vox may adopt the existing generic projection contract under the
new Vox cohort, or supply this alias in the final candidate. Product code already
uses the actual existing public class and does not depend on that optional alias.

## Remaining external qualification

The optional modern profile now pins the core owner-selected a794 source closure.
The core passed its own actual-source review and bounded synthetic CUDA checks;
full actual-source review of the Vox implementation remains pending. This core
still refuses wide waveforms under release0.9.4/auto. A separately reviewed
explicit-continuous profile is still required; no domain guard was changed. Default package metadata
continues to install the historical core0.11 cohort. Modern tests use the owner's
exact a794 installed candidate in a separate dependency view. The default core0.11
distribution pin is unchanged; no resolver-impossible old/new-core extra is added.

Historical CPU evidence included 36 tests in the earlier modern view, including ten direct HNDL tests:
native waveform/bank-VJP parity, first/second critic derivatives, actual public
pre-up capture, native reload, stock distilled reload, cleanup and source refusal.
These are small random official Vox modules, not pretrained speech. Distillation
remains approximate and unqualified; poor held-out fixture fit is retained.

Real qualification still requires visible/idle/leased physical GPU1, staged and
hashed official model weights, and the 10 verified same-speaker reference files.
GPU0 studio ownership remains protected. The bounded pilot now admits up to200
updates and an explicit wall budget only after matching measured source/model/
data/duration/GPU qualification. It does not bypass any gate or start training.
No central worktree edits, downloads, new GPU probes, paid calls or uploads were
performed for this integration pass.
