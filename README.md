# VoxCPM2 Particle Sliders

**Calculated memory:** Native default waveform training: **60 GiB GPU / 18 GiB host** for batch 2,4 frames,10 Euler steps,≤256 prefix positions. Matched fixed-waveform inference: **10 GiB GPU**. [Full memory ledger and limits](MEMORY.md). These are planning estimates, not measured peaks.

The explicit [native waveform backend](backends/voxcpm2-native/README.md)
now preserves the measured full-waveform implementation, scoped fresh
initialization, RAW exports and strict FULL restores. Its
[HOW/value guide](docs/voxcpm2-native95-learning-rates.md) and
[portable values](https://github.com/HyperGAN/voxcpm2-particle-sliders/blob/ffb4e74c8f8c9fbfb4c311abb56f2adefcfc2dc5/docs/voxcpm2-native95-learning-rates.json) record the completed
G0.012/G0.001 Sleepy comparison: both score0.0 at32 updates on two equal cases.
Every OFF and ON was rated3/4; no successful Sleepy effect, unique optimum,
default or automatic-use authority is established. Saved firsts and the original
failed core remain unchanged. No adapter-weight release is verified.

The current backend uses Particle Sliders core 0.15.2 and ParticleGAN 0.11.1
with pinned HNDL and HyperGAN. Fresh training uses a new waveform cohort.
The earlier core41f packaging record and measured runtimes retain their
original identities. CUDA still requires controller-owned admission. Public
fixed-duration rendering and the earlier native EOS previews use different
execution protocols.

The original scaffold context follows unchanged.

---

# VoxCPM2 Particle Sliders — review scaffold

Public source publication is authorized; final implementation review is pending.
The complete sanitized implementation is on [review/implementation](https://github.com/HyperGAN/voxcpm2-particle-sliders/tree/review/implementation)
and will be proposed in a draft pull request. It is not merged or release-ready.

The export includes no model/adapter weights, recordings, audio, judge maps or
keys, private run evidence, or training outputs. Only the approved sanitized
source export is used; the original private checkout/history is not published.

Focused CPU budget/hardening checks passed43 cases. Historical/small-random
native modules establish software behavior only. Wide-waveform training remains
blocked; pretrained speech, character fidelity, GPU fit and native quality are
unqualified. Public GPU UUID placeholders deliberately admit no physical device;
protected-GPU, lease, process, memory and source/admission guards remain.

See [PUBLICATION.md](PUBLICATION.md) for source lineage and validation scope.
Automated CI is not configured; no hosted check pass is claimed.

### Scheduling the original eager native-EOS path

An owner already running the original cached VoxCPM2 eager waveform path can
wrap a single native generation call with `voxcpm2_native.frame_strength.eos_patch_schedule`.
Pass effective strengths in `[0,1]` and use the resulting list to record the
patches that were actually emitted. The last strength repeats through native
EOS or the original cap. This scope uses the original stop head, cached
feedback and decoder. Use the fixed-patch `frame_gains` API for the separate
fixed-length waveform path.
