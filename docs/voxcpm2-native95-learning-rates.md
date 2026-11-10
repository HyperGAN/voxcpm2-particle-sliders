# VoxCPM2 Sleepy: measured values and native API

G0.012 and G0.001 tie at0.0 after32 completed updates on two matched finished-audio
cases. Each OFF and ON was assigned3/4 sleepy intensity. This is a completed
scoped provisional comparison with a flat, unqualified instrument; it does not
show that either recipe makes speech sleepier or establish a unique optimum.
Neither rate is selected for use or installed as a new default.

| Arm | Generator LR | D multiplier | Actual D LR | Bank multiplier | Actual bank LR | Finished effect ON−OFF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0.012 | 0.012 | 1.5 | 0.018000000000000002 | 2.5 | 0.03 | 0.0 |
| G0.001 | 0.001 | 18.0 | 0.018000000000000002 | 30.0 | 0.03 | 0.0 |

The sole selector is the equal-case mean normalized intended Sleepy intensity
ON minus matched OFF, with weights0.5/0.5. Package-seed7 and garden-seed11 each
contribute0.0 for both arms. Teacher/student SSE is diagnostic only. No
preservation, beauty, confidence, completion-order or second-score tiebreak is
used. The four original G0.012/OFF firsts were retained; only the two previously
missing G0.001 ON forwards were added. No old score, LR search, training update
or native output was replayed. [Machine-readable values](voxcpm2-native95-learning-rates.json).

The G0.001 arm retains its original failed core/STOP and first judgments. A later
separately qualified whole-garden observation returned PASS and a narrow explicit
decision permitted just the two missing effect diagnostics. Neither that PASS
nor this numeric comparison resets the failed core, establishes full-bank health,
permits training/resume/promotion or transfers eligibility from an earlier FULL2
guard. Catastrophic failure rejection stays separate from effect ranking; weak
effect is not catastrophic failure and missing checks remain unmeasured.

## Exact Source and installation scope

The native15-file implementation is byte exact measured product
`ba6ec023a121610a6a83cb2a816bf33f696584b2`. Runtime pins are core0.14.1/41f6066f,
ParticleGAN0.9.5, VoxCPM revision19b6bf7590025418821a86dcb817504e0ad7e5df,
HyperGAN5f368a3770cf5bdae0448995c9f76e29af8049f7 and HNDL0.10.0. Captured
Torch/torchaudio were2.9.1, NumPy2.2.6, safetensors0.8.0 and soundfile0.14.0.
The pretrained VoxCPM2 snapshot was7936080b41163874f38610f46f438e2c0757f9fd.
Model/reference assets and released adapter weights are not bundled.

Core41f's original package boundary omitted four existing HNDL Source/data files
and did not declare the requested `fibo-hndl` extra. The explicit
`f7001a821c89df6c2e8473da4f6914db45516aeb` packaging derivative includes those
same files and the HNDL0.10.0 extra. Core version0.14.1 and every runtime
Python/profile body remain unchanged, so the native exact core-file guard stays
intact. Current shared-core main and all captured environments are unchanged.

In a NEW already prepared owned runtime with those exact dependencies and
setuptools installed, the Source-only install route is:

```sh
python -m pip install --no-deps --no-index --no-build-isolation --no-cache-dir --no-compile \
  'particle-sliders-core[fibo-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@f7001a821c89df6c2e8473da4f6914db45516aeb#subdirectory=packages/particle-sliders-core'
python -m pip install --no-deps --no-index --no-build-isolation --no-cache-dir --no-compile \
  'voxcpm2-native95 @ git+https://github.com/HyperGAN/voxcpm2-particle-sliders.git@b470e56b840f1b2dec3c3009e385104defe7fd65#subdirectory=backends/voxcpm2-native95'
```

These commands fetch pinned Source and install no dependencies. Separately, on
2026-11-10 the exact coref700/nativeb470 Source pair completed five offline
packaging phases: one sdist and wheel-from-sdist per package, then one fresh
owned-target install. Validation copied the exact pinned Source into owned build
trees; it did not execute these VCS install commands. All70 core runtime files,
four HNDL leaves,12 checked core API hashes and15 measured native runtime files
matched. Distribution versions, mandatory/optional dependencies, pinned VCS URLs,
markers and console/backend entry metadata also matched.

The worker genuinely exited0 and was reaped; its exact owned resource roles were
released. No runtime module, model, GPU, old native test/proof, training, render or
score ran in this packaging validation. No public binary/PyPI release or full
native execution through the installed public package is claimed. The six prior
native CPU cases retain their captured Source/runtime scope and were not repeated.

## Common training options

For a separately authorized fresh cohort, choose an explicit recipe from the
table. The API does not choose a tied arm. Record the literal `lr`, `d_lr_mult`
and `prior_lr_mult`, retain `steps:128` even when stopping at32, and explicitly
set `ema_decay:null`. The measured geometry used dit hosts, rank8/alpha8,
128×4 particles, frames4, Euler10, batch2 and seed7.

```python
from particle_sliders.lifecycle import SessionConfig
from voxcpm2_native95.factory import open_training
from voxcpm2_native95.runtime import admitted

options = {
    "model_dir": "/ACTUAL/LOCAL/VoxCPM2",
    "revision": "7936080b41163874f38610f46f438e2c0757f9fd",
    "training_file": "/ACTUAL/matched-rows.json",
    "teacher_cache": "/ACTUAL/NEW-OWNED-COHORT/native-teachers.pt",
    "device": "cuda:0", "hosts": ["dit"], "rank": 8, "alpha": 8.0,
    "particles": 128, "frames": 4, "euler_steps": 10,
    "steps": 128, "batch": 2, "seed": 7, "ema_decay": None,
    "lr": 0.001, "d_lr_mult": 18.0, "prior_lr_mult": 30.0,
    "source_profile": "voxcpm2-core0141-pg095-full-waveform-hndl-v1",
    "initialization_policy": "voxcpm2-host-RMS-QR-zero-up-bank-R2-std1-v1",
    "pg_version": "0.9.5", "pg_source_profile": "release-0.9.5",
    "penalty_input_domain": "continuous",
}
config = SessionConfig("voxcpm2_native95", "A-NEW-EXPLICIT-COHORT", **options)
# owner is supplied by the real admitted controller, with actual held FDs and
# live ownership/caps/floors/failure checks. This snippet does not create one.
with admitted(owner):
    with open_training(config) as session:
        session.train(stop_after=32)
        session.checkpoint("/ACTUAL/NEW-OUTPUT/FULL32.pt")
        session.export_inference("/ACTUAL/NEW-OUTPUT/RAW32.pt", selection="raw")
```

The G0.001 literals above demonstrate the common interface; they are not a
recommendation, current candidate authorization or request to repeat the saved
comparison. A valid separately authorized restore supplies the genuine FULL
packet via `resume=...` and the identical config/cache. RAW is for inference;
never relabel it FULL, reset initialization, regenerate saved teachers or replay
completed updates. A qualified own failure stops the affected candidate before
another update/resume/promotion. The already failed measured arm remains stopped.

## Native previews

The backend's public `finished_audio`/render API uses the original full-prefix,
Euler and native AudioVAE with an explicit fixed frame count. It is not an EOS
utterance API. The finished-output comparison used the separate execution-owned
official `model.generate` EOS caller, with two original seeds, target texts,
reference, Euler10/CFG2, native stop head and honest stock cap; no VAD,
normalization, retry or duration truncation. That execution caller has not been
packaged here, so the fixed-duration public render is not claimed equivalent to
the measured full utterances.

Saved RAW32 full native previews are available in the
[remote sampler gallery](http://100.90.104.57:8770/#gallery). G0.001 package and
garden ON clips are2.72s/2.56s. Preview availability does not mean released
weights, an audible Sleepy effect, catastrophic eligibility or ordinary Generate
readiness. Preserve those saved outputs; publication needs no new render.
