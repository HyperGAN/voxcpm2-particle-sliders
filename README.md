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

Three Sleepy RAW64 timing panels produced sixteen new 48 kHz waveforms
through native EOS. At nominal 0.75, using 0.80 for six patches and 0.70
for six before returning to 0.75 recovered the measured pacing loss from
constant strength. Both early-heavy outputs passed speech integrity and
matched OFF on pacing. The reversed library case lacked a comparable score.
At nominal 0.5, late-heavy showed positive pacing gain against OFF, with
the constant arm missing a comparable library score.
[Per-case evidence](docs/sleepy-raw64-native-eos-timing.json)

A follow-up kept the first six patches at 0.75 and moved the same two
six-patch changes later. For replay of the tested Sleepy RAW64 profile,
pass `[0.75] * 6 + [0.80] * 6 + [0.70] * 6 + [0.75]` to
`eos_patch_schedule`. Its mean pacing gain was +0.00024 versus −0.02025
at constant 0.75 across the two matched cases. The tram case improved.
Both scheduled waveforms passed the native speech check. The reversed
library case lacked a comparable pacing score, and its archived neutral
speech verdict remains unmeasured.

All eight waveforms in a fresh two-case panel reached native EOS. In the
window case, all three profiles passed the speech check and tied at
+0.02353 in pacing gain. The clock case fell below the 0.80 transcript
threshold in every arm, including OFF, so the full panel has no comparable
mean for that panel.

A separate four-utterance confirmation produced 16 native-EOS waveforms,
all of which passed the frozen speech check. Mean pacing gain was +0.02465
at constant 0.75, +0.03002 with shifted early emphasis and +0.02939 with
its reverse. Shifted early emphasis improved two cases over constant, tied
one and lost one. Use the shifted early profile for the tested Sleepy RAW64
setup.
[Follow-up evidence](docs/sleepy-raw64-shifted-six-native-eos.json)
