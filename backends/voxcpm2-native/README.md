# VoxCPM2 native waveform backend

This backend uses Particle Sliders core 0.15.2 at `ec1161f5` with official
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
