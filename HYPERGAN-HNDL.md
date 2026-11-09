# Genuine modern HyperGAN/HNDL candidate

The modern path executes the shared routed projection and Vox's native-waveform
critic as real HNDL graphs through HyperGAN's reviewed `build_network`. The native
VoxCPM2 backbone, FSQ, full-feedback solver and AudioVAE remain official model
operations. The shared game still owns all losses, penalties, regularizers and
optimizer updates. Legacy and modern source identities remain separate.

`src/voxcpm2_particle_sliders/hndl_profile.json` identifies the inspected owner
candidate. The default `pyproject.toml` still pins the historical core0.11
cohort; do not install that dependency set and claim it is the modern runtime.
For candidate CPU reproduction use the exact installed owner core0.12 wheel
(SHA256 recorded in that profile), HNDL0.10.0 and exact HyperGAN develop build.
No owner files were modified or dependencies installed into its environment.
The optional profile now explicitly selects core commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, wheel SHA256
`22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a`.
Its installed 50-file source/data/license closure is checked before attaching
adapters, in addition to the shared verified HNDL/HyperGAN builder closure.
The core owner's actual-source review and bounded synthetic CUDA checks do not
constitute final review or native qualification of this Vox implementation.
No old checkpoint is relabeled or implicitly migrated to this new source closure.

Historical core4abc reproduction command (not the current installed-artifact receipt):

```bash
PYTHONPATH=.cpu:../task-3/evidence/joint-runtime-dependencies:voxcpm2-particle-sliders/src:vendor/VoxCPM/src:.deps \
  ../task-3/.venv-joint/bin/python -m pytest voxcpm2-particle-sliders/tests -q
```

Those historical runs passed36 modern tests. The separate historical view passed26 tests and
skips the one opt-in HNDL module. Both retain wide-waveform admission refusal.
These views use matched CPU torch/torchaudio2.10.0 and tiny random official model
classes; they do not establish CUDA performance or real character fidelity.

The critic graph matches the old Conv1d critic's outputs, input VJPs and tested
double backward exactly after affine weight matching. Routed legacy/HNDL
waveform differences are below2e-7 and bank-VJP differences below1e-12 in the
fixtures. Actual shared `lora_up` pre-hooks supply the pre-up calibration target
for existing `fit_routed_down`; no alternate output-regression conversion is used.

Once Vox source review, waveform admission, GPU and reference gates are actually met, start
with a bounded real qualification. Example commands are **blocked today**, not
executed recipes or evidence of an admitted future profile:

```bash
# CUDA_VISIBLE_DEVICES must map the verified physical GPU1; UUID is checked.
voxcpm2-slider train --network-profile voxcpm2-hypergan-hndl-v1 \
  --phase qualification --updates 2 --frames 4 --steps 10 \
  --model /verified/local/VoxCPM2 --revision FULL_MODEL_COMMIT \
  --manifest /verified/sleepy.json --device cuda:0 --max-wall-seconds 900 \
  --output runs/sleepy-qualification-4patches

# Before a longer-duration pilot, qualify that exact native duration/data first.
voxcpm2-slider train --network-profile voxcpm2-hypergan-hndl-v1 \
  --phase pilot --updates 100 --frames 48 --steps 10 \
  --qualification-receipt runs/sleepy-qualification-48patches/qualification.json \
  --model /verified/local/VoxCPM2 --revision FULL_MODEL_COMMIT \
  --manifest /verified/sleepy.json --device cuda:0 --max-wall-seconds 900 \
  --output runs/sleepy-pilot
```

Qualification requires >=2 successful shared-game updates covering every
generator row, frozen-base and connected finite-gradient evidence, actual CUDA
peak memory and per-update timing. The receipt binds the same model, adapter,
PG profile/source, Torch runtime, reference/input/condition bytes, batch and
native response geometry. A four-patch receipt cannot admit a48-patch pilot.
All authoritative recipe fields except the newly declared total-step budget
must match. A new horizon is a fresh game/cohort, never automatic optimizer
checkpoint migration. An optional native warm start is separately hashed.

Pilots admit1..200 updates and1..1800 seconds; the proposed batch uses100 updates
and900 seconds. The wall budget is checked between updates, not an interrupt of
one in-flight native update. Shared failures propagate; no failed step is
promoted. Successful partial runs retain exact checkpoint and stop reason.
The operational memory check requires the measured peak plus an explicit
headroom budget (default4096 MiB) to fit; this margin is not a scientific quality
threshold or a promise of same-card concurrency.

All CUDA model operations (train, smoke, render, distill and export comparison)
are restricted to physical GPU1 and hold the same cooperative UUID-bound lease
for the entire operation. Direct CUDA loading without that lease is refused.
Logical `cuda:0` is accepted only when it maps to the approved GPU1 UUID. The
lease cooperates with the observed
`/tmp/anima-1.lock` protocol and checks live compute ownership. GPU0 requires an
actual reviewed owner release before a parallel launcher can use it. No driver,
security or device-permission changes occur. All reference, model-weight and
real-world fidelity gates remain open, and native success must precede the
separate approximate LoRA-distillation quality gate.

Source hashes are drift checks after Python import, not a malicious-code security
boundary. The 50-file closure binds the shared core. `verified_builder` checks
HNDL/HyperGAN sources; official PG sources are checked by the shared PG validator
and game against their release manifest. Qualification uses a fresh labeled
post-update native waveform VJP rather than optimizer-owned `.grad` buffers.
