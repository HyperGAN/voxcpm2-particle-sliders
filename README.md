# VoxCPM2 Particle Sliders

**Experimental source publication.** Historical/small-random official-module CPU checks only. Current wide-waveform shared training admission remains blocked. No pretrained speech, character fidelity, real-model export, GPU fit or model-quality claim is established.

See [PUBLICATION.md](PUBLICATION.md) for sanitation, lineage and current CPU/CI limits.


Experimental public source for character TTS adapters on `openbmb/VoxCPM2`.
Native routed particles are the source; ordinary LoRAs are approximate distills.
This export includes no character-training or pretrained-listening evidence.
Historical assertions below refer to their recorded source snapshots, not an
audit of subsequent private work.

This standalone source export has fresh Git history. Read [EXPORT-NOTICE.md](EXPORT-NOTICE.md)
for omitted private/generated files, packaging verification and current holds.
Claude previously reviewed excerpts and a hardening packet. Final review of the
complete actual implementation, including its HyperGAN/HNDL binding and
operational paths, is still pending. Earlier feedback is not final source
approval, execution evidence, or voice-quality qualification.

## Current technical status

Twenty renderer/export CPU tests pass against actual official VoxCPM2 classes configured as small
random fixtures. They establish sampler geometry, gradient connectivity,
frozen-base ownership, stock inference agreement, exact zero-strength bypass,
native serialization and stock distilled-LoRA loading. These fixtures do **not**
qualify pretrained speech generation, memory, intelligibility or speaker quality.

Training is blocked by the pinned shared core's genuine continuous-vector
admission guard: `released PG09 cannot admit flat continuous vectors with width
>=170`. A useful 48 kHz waveform exceeds this. The product does not reshape audio
into images, shorten the response, substitute hidden-state probes or bypass the
guard. The inventory task owns shared-core compatibility upgrades.

This execution environment cannot communicate with the NVIDIA driver. No base
weights were downloaded and no shared GPU/process was displaced. Historical receipts are intentionally omitted from this sanitized source export;
see `EXPORT-NOTICE.md` for the verification scope.

The included four-target [Nova batch plan](batch/README.md) is an unresolved
template; this public payload contains no reference recordings, model weights
or rendered samples. Historical dependency-view runs passed
**26 tests**, with the optional modern module skipped. Earlier core0.12/
HNDL0.10/HyperGAN candidate runs passed **36 tests**, including ten direct modern
graph tests. [The genuine HNDL path](HYPERGAN-HNDL.md) is implemented locally;
[remaining profile/admission work](HYPERGAN-HNDL-BINDING-GAP.md) stays explicit.
Default dependency pins remain the historical cohort until the core owner
selects the final reviewed source profile.

## Dependencies and formulation

`pyproject.toml` pins these official sources by full commit:

| Dependency | Pin |
|---|---|
| VoxCPM code tag 2.0.3 | `19b6bf7590025418821a86dcb817504e0ad7e5df` |
| particle-sliders core 0.11.0 | `3c25f785d14073e6f627c0fa9d2297a041c79913` |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7` |
| ParticleGAN, as required by that core | `0.9.3` |

ParticleGAN 0.9.4 is available, but this core verifies exactly 0.9.3 source
hashes. Installing 0.9.4 separately is not a compatible upgrade. Latest-PG
qualification remains open. HyperGAN's train extra already requires 0.9.4;
this product pins its lightweight package and uses its public recipe-schema
integration as a compatibility witness. It does **not** claim the HyperGAN
reference GAN loop is the waveform trainer. `ConditionalPG09` owns the actual
conditional paired-error game, losses, penalty, particle regularizer, optimizers,
sampling and checkpoint state; no GAN mathematics are copied here. A core pin
upgrade must preserve that public game rather than silently switch objectives.

The legacy adapter imports `TinyParticleProjection` and `RoutedMLP` from the shared
package. The opt-in `voxcpm2-hypergan-hndl-v1` adapter uses the actual shared
`HNDLParticleProjection` and a Vox waveform critic graph through the verified
HyperGAN builder; both graph/source identities are signed separately. DiT attention q/k/v/o projections are the initial host. `--hosts dit lm`
also adapts the base and residual LMs. One shared particle cloud and FP32 adapter
leaves sit on a frozen backbone. Zero strength takes the original forward branch.

## Differentiable waveform path

`backend.waveform` uses full-prefix native causal LM forwards instead of mutable
KV-cache writes, supplied fixed noise for each patch, the unchanged official
`UnifiedCFM.solve_euler`, and the native AudioVAE decoder. Its output is the real
mono waveform `[1,samples]`. Reference input is encoded at 16 kHz; output is 48 kHz.

Public `_inference` / `_generate` and `UnifiedCFM.forward` use inference mode.
The training path does not call them. Fixed patch count replaces discrete stop
decisions; this is a bounded duration protocol, not learned stopping. Full-prefix
recomputation has quadratic sequence cost and needs real-model memory profiling.
Only FSQ is put into its official training-rounding mode: rounded forward values
are exactly eval values, while backward uses the official straight-through
estimator. Backbone weights stay frozen and other modules stay in eval.

The native adapter affects both CFG estimator halves, matching how stock VoxCPM
applies ordinary LoRAs. Teacher and neutral use the exact same noise and duration;
the positive reference supplies character/style conditioning. No denoising,
reconstruction or feature matching objective is added to GAN training.

## Installation and commands

Use an isolated Python 3.11/3.12 environment with a matched PyTorch/torchaudio
build for the machine, then `python -m pip install -e '.[test]'`. Official VoxCPM's
full dependency set is substantial. Code/source pins do not download base weights.

```bash
voxcpm2-slider preflight --output receipts/preflight.json
python -m pytest -q

# Requires an already downloaded verified local model and resolved HF revision.
voxcpm2-slider smoke --model /path/to/VoxCPM2 --revision FULL_MODEL_COMMIT \
  --device cuda:0 --frames 4 --output receipts/real-smoke.json

# Currently refuses at the shared waveform-domain guard, before loading weights.
voxcpm2-slider train --model /path/to/VoxCPM2 --revision FULL_MODEL_COMMIT \
  --manifest pilot.json --updates 2 --frames 16 --output runs/technical-gate

voxcpm2-slider render --model /path/to/VoxCPM2 --revision FULL_MODEL_COMMIT \
  --teacher runs/technical-gate/native.safetensors --strength 1 \
  --frames 16 --output runs/native.wav

voxcpm2-slider distill --model /path/to/VoxCPM2 --revision FULL_MODEL_COMMIT \
  --teacher runs/technical-gate/native.safetensors --manifest pilot.json \
  --frames 16 --output runs/distilled

python scripts/compare_exports.py --model /path/to/VoxCPM2 \
  --revision FULL_MODEL_COMMIT --teacher runs/technical-gate/native.safetensors \
  --distill runs/distilled --manifest pilot.json --output runs/heldout-comparison
```

Model loading requires local config/tokenizer, model safetensors and official
AudioVAE safetensors or `audiovae.pth` (stock `weights_only=True` loading),
hashes them, checks architecture and records the supplied resolved revision.
Public UUIDs are non-hardware placeholders and must be bound by reviewed
deployment configuration before CUDA can be admitted. The lock protocol and
protected-device refusals remain unchanged. Every CUDA command holds the
approved physical GPU1 UUID lease; protected GPU0
is refused even when idle, and direct unleased CUDA loading is refused. Logical
`cuda:0` must map to GPU1. Loading also requires at least 24 GiB free; this is a
conservative admission budget, not a measured training requirement. The official
LoRA estimate is about 20 GB, but native routed waveform training is unmeasured.
Nothing kills or moves another job.

A pilot manifest uses explicit row IDs, split, shared spoken text, seed and
positive character reference; optional neutral reference supplies the off voice:

```json
{
  "provenance": {"character": "pilot", "reference_dataset": "local dataset identity"},
  "rows": [
    {"id": "train-01", "split": "train", "text": "We can start again.",
     "seed": 171, "positive_reference": "refs/character.wav"},
    {"id": "train-02", "split": "train", "text": "The door is still open.",
     "seed": 172, "positive_reference": "refs/character.wav"},
    {"id": "dev-01", "split": "dev", "text": "I knew you would return.",
     "seed": 173, "positive_reference": "refs/heldout-character.wav"}
  ]
}
```

## Native checkpoints and ordinary LoRA distillation

Native safetensors contain only adapter leaves with model hashes, projection
identity, architecture and data provenance. The shared game checkpoint retains
optimizer/critic/prior/RNG/row identity for exact resume; an inference export is
separate. No frozen base weights are exported.

The existing Anima/YuE workflow is reused: bounded full projection inputs are
captured with each routed bottleneck's pre-up output; shared `fit_routed_down`
performs the same dual ridge solve at fraction .01, retaining the trained up
matrix. Separate train/dev records are used. No alternate direct-LoRA training
is substituted. The native alpha/r factor is carried in exported B; stock
`LoRAConfig(alpha=rank)` supplies unit loader scaling.

Capture requires runtime strength 1 and verifies that shared `branch.scale`
remains fixed alpha/r. Export provenance records duration, Euler steps,
`cfg_value`, `cfg_zero_star`, sway and the both-CFG-branches convention. The stock
comparison uses those same calibration settings and rejects unsupported stock
zero-star/sway changes. This is separate from testing extrapolation at new CFG.

Ridge reports use held-out activations only. Exact held-out inputs already in
calibration, and repeated held-out inputs, are excluded before computing
bottleneck/projection relative MSE and centered R²; fewer than two disjoint rows
per projection is refused. Fit-set error is not presented as fidelity. In the
small random CPU fixture, held-out projection relative MSE is about 0.477 and
R² about -122: loading and finite audio do not imply a good approximation.

Exports use VoxCPM's exact `name.lora_A` / `name.lora_B` keys, plus a loader config
and teacher/data/calibration provenance. The CPU test invokes the actual stock
`VoxCPM2Model.load_lora_weights` and checks every key loaded with none skipped.
Stock use after base loading:

```python
import json
from voxcpm.model.voxcpm2 import VoxCPM2Model, LoRAConfig

cfg = LoRAConfig(**json.load(open("runs/distilled/lora_config.json")))
model = VoxCPM2Model.from_local("/path/to/VoxCPM2", optimize=False,
                               device="cuda:0", lora_config=cfg)
loaded, skipped = model.load_lora_weights("runs/distilled")
assert loaded and not skipped
audio = model.generate(target_text="We can start again.")
```

Distillation approximates nonlinear, input-dependent routing with one linear
map. Projection error alone cannot qualify a voice. The comparison script uses
stock generation with matched fixed length/noise at 0/.5/1/2, reports unaligned
waveform relative MSE/SNR/log-spectral error and keeps intelligibility, identity,
expressivity and strength-quality gates pending for independent transcripts,
speaker similarity and blind listening. Preserve the native teacher permanently.

The review tests additionally compare every patch's mu, Euler output and actual
fed-back encoder input/features against official KV-cache inference, for both
zero-shot and reference-prefix layouts at identical noise/CFG. FSQ backward is
checked against `out_proj(tanh(in_proj(hidden)))`, not identity. Separate probes
check decoder-to-latent gradients, final-patch-to-first-patch feedback gradients,
and zero-up initialization followed by one diagnostic up-only update producing
nonzero down gradients in DiT and optional LM hosts. These are CPU diagnostic
updates, not GAN training or pretrained character evidence.

## Proposed next pilot, after technical gates

First qualify one already authorized idle GPU with four patches and a bounded
real-model forward/backward/native-reload/stock-distill-reload check. At fresh
zero-up initialization only up weights have nonzero gradients; all other
intended leaves must be connected and finite, then nonzero after a genuine game
update or in a separately labeled restored diagnostic probe.

After shared-core waveform admission and real-model technical qualification, try one
short character pilot, with held-out text/reference separation. Compare neutral,
sleepy, excited and laughing readings, native vs distilled, at 0/.5/1/2. Record
WER/CER, speaker similarity, blind expressivity/identity ratings and failures.
Choose the character and exact training budget from those technical results;
no long training campaign, audio upload or weight publication is authorized by
this source publication.

Installed VoxCPM Python sources are matched against the pinned official commit
before its code revision is recorded in model identity. This import-then-hash
check detects drift; it is not a security sandbox. Full Vox actual-source review
and native pretrained/reference/GPU qualification remain pending.

Controlled teacher rows may include `positive_control`, a single-line description
without parentheses. Both `neutral_reference` and `positive_reference` must then
resolve to identical frozen audio bytes. The neutral prefix uses plain target
text; the positive prefix uses `(positive_control)` followed by that same text.
The frozen backbone generates both teacher responses at the same supplied noise
and native duration with the slider bypassed. This uses the official text-control
layout; it does not train against arbitrarily aligned stock WAV files. The shared
ParticleGAN objective, HNDL graph, native waveform coordinates, and admission gate
are unchanged. Existing rows without this field retain reference-based behavior.

Stock prompt samples are screening evidence only. Actual fixed-length teacher
responses still require content/delivery screening at the proposed training
duration. The current 64-patch bound is 10.24 seconds on the official model, so a
12-second full-text Sleepy sample cannot be silently truncated into a qualified
target. The initial proposed pilot uses two short, explicitly train-side excerpts;
full-text OFF/ON evaluation uses stock stopping with native particles attached.

Pilot runs require a fresh game at step zero. `--resume` and `--teacher` warm
starts are refused for the pilot phase; a qualification checkpoint is evidence,
not the pilot's optimizer or parameter starting state. The default operational
headroom is 8192 MiB, checked against current free plus owned reserved capacity
after matching qualification and before updates. Incomplete qualification writes
diagnostics only, without native/game checkpoints or a qualification receipt.

The public recipe retains `training_precision=bf16_channels_last`; this native
host does not enter GANTrainer autocast and records `host_training_precision`
as float32. Particle leaves and critic parameters are float32, while the frozen
backbone follows its actual model dtype and VAE inputs match the decoder dtype.
The recipe's UCD class fields are inert here: ConditionalPG09 calls score-based
adversarial losses, the public critic penalty and SBNP, with no UCD/CE consumer.
Controlled-teacher rows reject a leading parenthesized neutral control. The
train/dev duplicate signature deliberately ignores positive_control: changing
delivery wording does not make an otherwise identical example held out. Actual
prepared positive prefixes and manifest provenance still bind the control text.

An explicitly separate local candidate, `voxcpm2-hypergan-hndl-singleton-v1`,
uses the exact singleton-spatial HNDL graph reviewed in core proof
`6872683b15cfc267e725acdce59c5772ebf9f233`. It builds a fresh immutable plan
with real Conv2d `(1,15)` kernels; it does not edit the original Conv1d plan or
convert old checkpoints. The game still receives `[batch,samples]` and the
unchanged release0.9.4/auto engine. The original profile, graph and manifest
remain available. Native packets identify these as different cohorts.

This candidate addresses official DualNorm convolution metadata compatibility
only. CPU equivalence and small critic optimizer restore are not CUDA polar,
pretrained/native, waveform-admission, or cadence qualification. The receptive
field remains 71 samples and the 307200-sample pilot remains gated. No temporal
critic or alternate GAN objective is introduced by the representation change.

Qualification receipts and saved game packets carry `voxcpm2_cohort`, containing
the full network profile/source identity. Resume checks this marker and the
embedded adapter identity before calling the shared loader, then removes only
the product marker. `--allow-historical-packets` explicitly permits unlabeled
legacy/original-profile packets subject to unchanged shared source/recipe checks;
it never permits singleton rebinding. Fresh-pilot restrictions still apply.
