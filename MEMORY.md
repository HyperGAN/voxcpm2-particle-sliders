# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Full-waveform response training (native defaults) | 1 | 60 | 18 | one GPU; ≤ 512 cachedpairs; batch 2; 4 frames×4patch=30,720 samples(.64s48k); 10 Euler; text/referenceprefix≤ 256 positions; BF16 base/FP32 VAE+adapters; no checkpointing |
| Fixed native waveform inference / teachers | 1 | 10 | 18 | batch 1; same4 frames/10 Euler/≤ 256prefix; native 48k mono AudioVAE; original runtime uncompiled |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN 0.9.5 `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **openbmb/VoxCPM2**, revision **`7936080b41163874f38610f46f438e2c0757f9fd`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Full-waveform response training (native defaults) | BF16_base: 4.2655; AudioVAE_FP32_file_upper: 0.3511; preallocated_LM_residual_KV: 0.2812; trainable_transient: 0.0930; unrolled_DiT: 0.9375; unrolled_LM_residual: 46.1250; AudioVAE_saved_activations: 0.5312; cached_target_neutral_prefix_condition: 0.2539; workspace: 6.0000 |
| Fixed native waveform inference / teachers | BF16_base: 4.2655; VAE: 0.3511; preallocated_LM_residual_KV: 0.2812; adapter: 0.0058; one_Euler_call_buffers: 0.1172; prefix_prefill: 0.0279; VAE_activations: 0.2656; workspace: 4.0000 |

Detailed arithmetic and assumptions:

- **BF16 base bytes**: `4580009088`.
- **AudioVAE FP32 file upper**: `376951122`.
- **preallocated LM residual KV bytes**: `301989888`.
- **host model loading peak bytes**: `9160018176`.
- **DiT LoRA elements upper**: `1179648`.
- **adapted DiT projections**: `48`.
- **generator plus wave critic elements upper**: `1560769`.
- **DiT unroll activation formula**: `64*B*frames*Euler*12*4*1024*4`.
- **LM unroll formula**: `64*B*frames*36*prefix*2048*4 +12*B*frames*36*prefix*6144*2`.
- **decoder stage position channels**: `[131072, 393216, 983040, 983040, 983040, 983040]`.
- **decoder saved tensor allowance**: `16*B*4*sum(decoder_stage_position_channels)`.
- **host cache**: `≤512pairs: two30,720sample FP32 waves +two256position prefix tensors +4096condition upper; two CPU serialization copies; model CPU FP32construction or CPU BF16model+state (2F) +2VAEpayloads +8 GiB +exports`.

Scope and limits:

- The original from_local loader constructs full CPUFP32 model shapes beforecastingBF16, then holdsCPUmodel+checkpointstate untilGPUtransfer; host budgeting uses2F. It also preallocates8192-position BF16 LM/residualKV buffers (301,989,888 bytes), even for short fixed patches.
- The current product renderer is fixed-duration waveform inference, not the separate private native-EOS museum sampler. Bounds are4 frames/10 Euler by default; Source permits64 frames/32 Euler and8192 total prefix+frame positions, which require a much larger calculation. Do not apply the short-patch training estimate to the maximum unrolled graph.
- Only the DiT attentionQ/K/V/O projections are adapted by default. Optional LM hosts add trainables and activation retention. The frozen base includes LM/residualLM/featureencoder/DiT; its separate AudioVAE remains FP32.
- AudioVAE .pth file size is a conservativeFP32 tensor upper bound rather than an invented parameter count. This document does not claim FP4/CPU-offload support, intended-effect success, or that increasing audio duration is free.

Source facts (read as text; no model execution):

- [backends/voxcpm2-native95/src/voxcpm2_native95/contracts.py](backends/voxcpm2-native95/src/voxcpm2_native95/contracts.py)
- [backends/voxcpm2-native95/src/voxcpm2_native95/adapter.py](backends/voxcpm2-native95/src/voxcpm2_native95/adapter.py)
- [backends/voxcpm2-native95/src/voxcpm2_native95/runtime.py](backends/voxcpm2-native95/src/voxcpm2_native95/runtime.py)
- [backends/voxcpm2-native95/src/voxcpm2_native95/waveform.py](backends/voxcpm2-native95/src/voxcpm2_native95/waveform.py)
- [backends/voxcpm2-native95/src/voxcpm2_native95/finished_audio.py](backends/voxcpm2-native95/src/voxcpm2_native95/finished_audio.py)
- [backends/voxcpm2-native95/src/voxcpm2_native95/wave_critic_singleton.hndl](backends/voxcpm2-native95/src/voxcpm2_native95/wave_critic_singleton.hndl)
- [Pinned model metadata](https://huggingface.co/openbmb/VoxCPM2/tree/7936080b41163874f38610f46f438e2c0757f9fd) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.9.5 release `1b8355b184b21bb7d81104ee590813c262f6df0a`, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `pg09_audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
