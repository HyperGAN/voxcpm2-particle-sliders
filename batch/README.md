# Four delivery sliders: preparation only

Nova's initial batch is **Sleepy, Medieval Town Herald, Auctioneer and Flat
Announcer**. Definitions are hypotheses, with no speech references or measured
character effects available yet. All targets use the same base speaker and the
same modern neutral transcripts for OFF/ON. Herald is proclamation cadence,
projection and phrase endings; archaic words or a different speaker do not count.
Sleepy remains projected and intelligible rather than whispery/intimate.

`nova-v1/batch-plan.json` records dependency, actual HyperGAN/HNDL binding, GPU,
reference, qualification, pilot and evaluation gates. No scheduler or trainer is
started by any preparation command. GPU0 keeps its studio reservation. One
bounded qualification belongs on verified idle GPU1, then actual peak memory and
time determine concurrency. Two jobs total is a proposal if both owners release
their cards; packing two per card is unqualified.

The trainer keeps <=10-update technical qualification separate from a bounded
1..200-update pilot (proposed100 updates/900 seconds). A pilot requires actual
matching model/source/data/native-duration/GPU qualification and explicit memory
headroom. These commands remain **blocked today** by admission, model, GPU and
reference gates; the original prototype cap is not treated as user policy. See
[the modern integration and pilot contract](../HYPERGAN-HNDL.md).

The reference registry contains **10 unresolved recordings**: shared neutral
train/dev references and positive train/dev references for each delivery. Paths,
hashes, speaker IDs, recording sessions, transcripts and origins are null until
verified from actual recordings. Train/dev text and reference sessions remain
disjoint. No generated audio or synthetic training data is claimed as present;
no synthetic-reference spending is authorized by these manifests.

Reproduce the public plan/templates in a fresh directory (blinding IDs and order vary per batch):

```bash
python -m voxcpm2_particle_sliders.batch --output /path/to/fresh-batch-directory
python -m voxcpm2_particle_sliders.audio_integrity \
  --registry nova-v1/reference-registry.template.json --output /tmp/reference-audit.json
```

Audit verifies file digests, finite/nonempty audio, source metadata consistency,
same base speaker, distinct held-out sessions and recording hashes. It records
sample rate/channels, duration, peak/RMS/DC offset, clipping, near-silence and
trailing silence. Duration/amplitude screens are provisional review flags, not
scientific character-quality thresholds. Flagged files require explicit review
notes; an integrity pass is not proof of speech, identity or delivery quality.
The existing model reference loader performs the native 16kHz encoding; the
audit never rewrites or resamples source recordings.

The four `*.manifest.template.json` files have unresolved reference paths and
`executable=false`. The generator can produce `evaluation-requests.json` containing **256 unrendered requests**
for the expanded grid: 4 targets × 4 held-out texts × 2 seeds × 4 strengths
(0/.5/1/2) × native/distilled. Begin with the **16 native first-pass requests**
(2 texts × 1 seed × OFF/ON × 4 targets) before expanding. Strength2 is an
extrapolation stress test; training strength is1.

The same held-out texts include banal modern prose, urgency, an unhurried
instruction and exciting news. These probe content/delivery mismatch without
changing words between targets. Keep model/checkpoint/reference hashes, seed,
fixed duration, CFG2, zero-star and sampler steps with every render. Actual
output paths/digests remain null until files exist.

The generator can produce `judge-pairs.blinded.json` providing 192 proposed comparisons using opaque asset
IDs and randomized presentation order, without target/strength/format labels.
Each batch uses a new 256-bit random salt kept only in the ignored private
mapping. Asset IDs, pair IDs and swap choices use independent HMAC domains;
public IDs cannot be enumerated from the known labels or used to infer ordering.
The generated label/evaluation map and salt mapping are exclusive mode0600
files; they must remain local and ignored. No existing maps or keys are included
in this source export. This is presentation blinding, not
a security mechanism. Nothing uploads audio or calls Gemini/NanoGPT. Later
authorized judges can assess character identification (including none),
transcription fidelity, naturalness/artifacts and speaker consistency. Local
integrity checks come first; ASR/human transcript and speaker measures remain
separate evidence. Low-pass prosody-only listening is supplementary, never
definitive semantic-disentanglement proof. Nova's 70% ID and4/5 clarity ideas are
uncalibrated suggestions, not automatic acceptance criteria.

Distilled requests require a successful native candidate. Ordinary LoRA remains
an approximate, separate gate with disjoint held-out activation error/R² and
stock/native waveform, intelligibility, speaker and expressivity comparisons.
No arbitrary R² cutoff replaces real character evaluation.
