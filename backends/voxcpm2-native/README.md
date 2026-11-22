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

The published Sleepy RAW64 study records the probe, active calibration and
held-out waveform comparisons for these opt-in gates.
[Native gate study](../../docs/sleepy-raw64-activation-gate-evidence.json)

### Compare timing over a shared native prefix

Set `eligible_patches` to compare timing within a prefix every arm emits.
In the pinned cached sampler, use `min_len + 2` when `max_len` permits it.
The native stop check runs after emission and requires the zero-based
patch index to exceed `min_len`. All arms continue to native EOS. The
gate counts only the prefix, and the trace records zero strength for all
later patches. A run that ends before the requested prefix raises an error.

Calibrate the gate separately on that prefix. For a held-out run whose
gate fired `k` times over `n` eligible patches at strength `a`, pass
`[a * k / n] * n + [0.0]` to `eos_patch_schedule` for its constant control.
Record the applied strength traces and verify their summed strength at
native precision. Score the entire finished waveforms and check speech
integrity independently. This comparison controls the applied strength
when the adaptive and constant utterances end after different numbers of
patches.

## Replay the fitted S-PSR intervention

`eos_patch_spsr` reads the fused `mu` row at each official EOS decoder call.
It applies the shared `PromptSteering` edit `mu + alpha * ReLU(mu @ w + b) * d`
in FP32, then passes the result to the decoder in its native dtype. The fitted
weights are `w`, `b` and `d`. The scope holds the DiT-only RAW adapter at
zero for this standalone study. Each trace entry is the applied FP32
coefficient `alpha * ReLU(mu @ w + b)`, including the final emitted patch.
[Steer Like the LLM, equations 7–8](https://arxiv.org/html/2605.03907v1#S3.SS4)

The English Sleepy study used BF16 `mu` at width 2048, VoxCPM2 model revision
`7936080b41163874f38610f46f438e2c0757f9fd`, and `alpha=1.0` from the study's
frozen evaluation configuration. The trained `FULL-00045.pt` checkpoint
contains the fit under `probe`. Its SHA-256 is the `fitted_module_sha256` in
the study. The fit used 115 native patches from three prompt-paired utterances
and kept the base model frozen. [Fit and calibration evidence](../../docs/text-only-spsr-heldout-evidence.json)

Use the existing guarded RAW loader to obtain `native` with its original
owner. Supply the trained checkpoint and the native request from the recorded
case. Keep the same case seed and disable sampling retries. In the owner's
open inference session:

```python
from hashlib import sha256
from pathlib import Path

import torch
from particle_sliders.prompt_steering import PromptSteering
from voxcpm2_native.frame_strength import eos_patch_spsr

native = session.binding.response.__self__.runtime
checkpoint = Path(fitted_checkpoint)
if sha256(checkpoint.read_bytes()).hexdigest() != study["fit"]["fitted_module_sha256"]:
    raise ValueError("unexpected fitted S-PSR checkpoint")
fit = torch.load(checkpoint, map_location="cpu", weights_only=True)
steering = PromptSteering(2048).eval().requires_grad_(False)
steering.load_state_dict(fit["probe"], strict=True)
steering.to(native.owner.device)
request = {**native_request, "retry_badcase": False, "trim_silence_vad": False}

with session._observation(seed, {"native_profile": "official-EOS", "request": request}):
    with torch.inference_mode(), eos_patch_spsr(
        native, steering=steering, alpha=study["calibration"]["frozen_alpha"]
    ) as applied_coefficients:
        waveform = native.model.generate(**request)
```

Exact replay of the measured WAVs requires the archived `voxcpm2_native95`
runtime and its original runner, native request, owner and RAW64 checkpoint.
The original RAW64 SHA-256 is
`2f43ce750c03503a90a01423aa850d9b3c97edd6ca5e8ae515170546d655903e`.
The current loader validates the current RAW cohort. The trained checkpoint
and WAVs are archived with the original run.

CPU replay reproduced the original edited BF16 values and FP32 coefficients
on all 115 saved fit rows. The held-out receipts retain 91 applied
coefficients and their EOS lengths, which were verified against the control
dose calculations. Recomputing the held-out coefficients would require the
pre-edit held-out `mu` rows.

Across four held-out utterances, S-PSR increased mean pacing by `+0.02823`
versus OFF on the study's log active-duration scale. The two cases with a
constant matched to the observed EOS dose at FP32 precision averaged
`+0.00124` versus that constant. The other two timing contrasts are
unmeasured after the available control search. Store the coefficient trace,
count and total alongside the WAV, source and checkpoint hashes. Match each
constant to the adaptive total at the constant's realized native EOS length.
[Per-case outcomes and control search](../../docs/text-only-spsr-heldout-evidence.json)

For a separate fixed-prefix study, pass `eligible_patches` to
`eos_patch_spsr`. Use the same guaranteed native prefix in the adaptive and
dose-matched runs. The hook passes through all later decoder inputs and
records a zero coefficient on each. With the fitted direction held fixed,
the constant control uses the nearest representable FP32 mean of the
applied adaptive coefficients across the prefix. A shared
`PromptSteering` module with zero probe weights and that mean as its probe
bias applies this coefficient at `alpha=1`.

The reference-conditioned fixed-prefix study used the fitted text-only
module at alpha 1 for four native patches. All 12 clips passed independent
speech integrity. Two of four cases had matching transcript word counts and
qualified for pacing. Across those two, S-PSR averaged `+0.00257` versus OFF
and `+0.01433` versus the dose-matched constant on the log active-duration
scale. The full-panel effect remains unmeasured.
[Fixed-prefix per-case evidence](../../docs/sleepy-raw64-spsr-fixed-prefix-heldout.json)

A fresh four-utterance confirmation used the same text-only fit, stock
reference conditioning, and four-patch comparison. All 12 clips passed
independent speech integrity, and each case met the matched word-count
rule. S-PSR averaged `−0.01698` versus OFF and `+0.00359` versus the
dose-matched constant on the log active-duration scale. The test failed
the positive-effect criterion versus OFF for this profile.
[Four-case confirmation](../../docs/sleepy-raw64-spsr-fixed-prefix-cached-confirmation.json)

#### Qualified text-only fixed-prefix replay

The original text-only stock path passed a separate four-case confirmation
without reference audio or a reference cache. Use the pinned 45-update
fit at its original alpha of 1 and pass `eligible_patches=4` to
`eos_patch_spsr`. The original minimum length guarantees those four patch
calls. After the prefix, the native decoder continues through EOS with zero
added coefficients. The companion constant uses the observed FP32 mean over
the same prefix for that case.

All 12 clips passed independent speech integrity and all four comparisons
qualified for pacing. S-PSR averaged `+0.03466` versus OFF and `+0.04956`
versus the dose-matched constant on the log active-duration scale. The
measured result applies to this text-only fit and native sampler.
[Text-only confirmation](../../docs/sleepy-raw64-spsr-fixed-prefix-text-only-confirmation.json)


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
