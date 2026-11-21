# VoxCPM2 native waveform backend

This backend uses Particle Sliders core 0.15.2 at `b1f780de` with official
ParticleGAN 0.11.1. It preserves the native mono48k waveform response, RAW
exports, scoped initialization and exact FULL continuation. HNDL graphs and
builder checks are adapted from the MIT-licensed core source, with HNDL 0.10.0
and HyperGAN pinned directly.

To schedule RAW strength, include `frame_gains` in the native render request.
Provide one gain for each declared `frames` patch. That patch uses the same
effective strength for its adapted language and denoising forwards. Keep
`strength * gain` within `[0, 1]`. The WAV sidecar records the values in
`effective_frame_strengths`. The tiny native AudioVAE check passed. Finished
pretrained speech effect is pending a matched comparison.

## Gate the saved RAW adapter in native EOS generation

`eos_patch_gate` applies the shared CFP, CLABS, or matched random gate to one
patch at a time in the original eager cached sampler. It reads `mu[0]`, the
current concatenation of the base and residual LM states after their native
DiT projections. This row is available before the DiT adapter can affect the
patch. There is one eligible event per actual decoder call, including the last
patch before the stop head reports EOS. Prompt prefill, feedback, DiT Euler
substeps, and the stop check do not advance the gate separately. The original
stop head, model RNG, classifier-free guidance, cache and AudioVAE remain in
charge of generation. Keep bad-case retries disabled so one scope is one
trajectory. The adapter must target the DiT only.
[Original native loop](https://github.com/OpenBMB/VoxCPM/blob/19b6bf7590025418821a86dcb817504e0ad7e5df/src/voxcpm/model/voxcpm2.py#L972-L1013)

This variant keeps the saved routed RAW edit and varies its strength between
zero and the selected native strength. The direction is a readout for the gate.
The paper's residual addition uses its direction as the edit itself. Preserve
the original RAW loader, sampler, and CUDA ownership for any replay.
[Right Token, Right Time](https://arxiv.org/html/2611.18274v1#S3)

```python
from particle_sliders.adaptive_steering import ActivationGate
from voxcpm2_native.frame_strength import eos_patch_gate

# Use the original owner scope, exact RAW loader, and per-case native RNG scope.
gate = ActivationGate("cfp", calibrated_threshold)
with eos_patch_gate(native, strength=qualified_strength, gate=gate,
                    direction=fitted_mu_direction) as patch_trace:
    audio = native.model.generate(retry_badcase=False, **native_request)
# Store patch_trace in the WAV receipt with the gate and artifact provenance.
```

Use one gate per utterance. CLABS also takes an explicit gate RNG seed and
`CLABSParameters` with measured projection statistics and a recorded controller
configuration. The audio RNG remains independent. Store each patch's
projection, probability, firing decision and effective strength. Store the
exact RAW, VoxCPM2 weights and source, solver settings, direction artifact,
calibration inputs and configuration with the trace.

To reproduce the paper's probe derivation in `mu` coordinates, collect the fused
row at the same hook with verified target and neutral corpora under the same
model and sampler. Use an SAE trained on this exact fused representation.
Average SAE codes over patches within each utterance and supply the example
rows, decoder matrix and declared feature count to
`particle_sliders.adaptive_steering.contrastive_sae_direction`. Keep the corpus
and SAE identity, feature selection and direction checksum. The fitted vector
must match the full fused width and have unit norm.

Fit the CFP threshold or CLABS bias with the shared `calibrate_gate_budget`
using fresh gate state for each rollout. Reuse the same disjoint calibration
utterances and seeds for each candidate with RAW steering active. Count every
emitted patch through native EOS or its cap, and report held-out realized
fired/eligible counts even when runs end at different lengths. Match the
random gate at the same target rate and strength. Report summed effective
strength as well as firing fraction. Compare whole waveforms to matched native
OFF, selected constant strength and the retained fixed schedule, under the
original conditioning/Euler/CFG, reference and seeds. Fit the direction and
calibrate on separate cases from the four qualified Sleepy RAW64 utterances
(porch, dinner, park and shop), which supply prior matched comparators. Keep the
original frozen speech checks and rank by equal-case pacing ON minus OFF on
qualified output pairs. Record the speech verdict separately and leave
incomplete or uncertain pairs unmeasured.
[Qualified RAW64 comparison and frozen checks](../../docs/sleepy-raw64-shifted-six-native-eos.json)

The saved audio cases contain the fixed-schedule results. They provide no
qualified fused-row SAE/direction or active-rollout threshold or bias for the
new gate. The adaptive scope is opt-in pending that calibration and a native
finished-output comparison.

For the CPU fixtures, install matching CPU PyTorch and torchaudio wheels before
installing this subproject. For example, from this directory:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu' 'torchaudio==2.9.1+cpu'
python -m pip install '.[test]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Two32-update pretrained Sleepy arms, G0.012 and G0.001, have completed the matched
finished-output comparison. Both score0.0; every OFF and ON was rated3/4. This
flat provisional instrument establishes no successful Sleepy control, unique
optimum, default rate or automatic-use authority. The original failed core and
all first observations remain unchanged. See the [HOW/value guide](../../docs/voxcpm2-native95-learning-rates.md)
and [portable values](https://github.com/HyperGAN/voxcpm2-particle-sliders/blob/ffb4e74c8f8c9fbfb4c311abb56f2adefcfc2dc5/docs/voxcpm2-native95-learning-rates.json).

Six distinct native CPU software cases and the original pretrained work remain
evidence for their captured Source/runtime. The exact public Source pair passed
five offline packaging phases: core and native sdists, wheels from those sdists,
and one fresh owned-target installation. All70 core files, four HNDL leaves,
12 core-guard hashes and15 measured native files matched, along with dependency
and entry metadata. Runtime modules were not imported; full native execution
through this installed public package remains unverified. No public binary or
released adapter weights are verified. Completed checks and searches are reused;
publication does not repeat them.

The saved core41f validation applies to the earlier cohort. CUDA still requires the real controller's
`runtime.admitted(OwnedCUDAAdmission(...))` context with actual UUID, held lease
FDs, receipt and live checker. The bare console cannot supply that authority.
CPU support is for tiny software fixtures; use the admitted CUDA API for the
pretrained model. No automatic placement, CPU offload or permission bypass is
provided.
