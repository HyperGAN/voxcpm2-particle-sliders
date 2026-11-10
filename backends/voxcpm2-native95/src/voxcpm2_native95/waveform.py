"""Fixed-noise, fixed-length native VoxCPM2 waveform path.

The model's ordinary full-prefix causal forwards replace its mutable KV cache.
The unchanged official UnifiedCFM.solve_euler is called directly because its
public forward is inference-only. No diffusion/denoising loss is optimized here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy
import math

import torch
from particle_sliders.pg09_backends import AdapterParameters


@dataclass(frozen=True)
class Prefix:
    tokens: torch.Tensor       # [1,T], includes audio_start; reference tokens optional
    feats: torch.Tensor        # [1,T,P,D]
    text_mask: torch.Tensor    # [1,T]
    audio_mask: torch.Tensor   # [1,T]


@dataclass(frozen=True)
class WaveRow:
    neutral: Prefix
    positive: Prefix
    noise: torch.Tensor        # [frames,1,D,P], supplied and shared exactly
    steps: int = 10
    guidance: float = 2.
    sway: float = 1.
    cfg_zero_star: bool = True
    scale: float = 1.


@dataclass(frozen=True)
class PreparedWave:
    row: WaveRow
    target: torch.Tensor
    neutral: torch.Tensor


def _copy_prefix(prefix):
    return Prefix(*(v.detach().clone() for v in (
        prefix.tokens, prefix.feats, prefix.text_mask, prefix.audio_mask)))


def validate_prefix(model, prefix):
    if prefix.tokens.ndim != 2 or prefix.tokens.shape[0] != 1 or prefix.tokens.shape[1] < 1:
        raise ValueError("single nonempty native prefix required")
    t = prefix.tokens.shape[1]
    if prefix.tokens.dtype != torch.long or (prefix.tokens < 0).any():
        raise ValueError("nonnegative long token IDs required")
    if prefix.feats.shape != (1, t, model.patch_size, model.feat_dim):
        raise ValueError("native prefix patch geometry differs")
    if prefix.text_mask.shape != (1, t) or prefix.audio_mask.shape != (1, t):
        raise ValueError("prefix mask geometry differs")
    for value in (prefix.feats, prefix.text_mask, prefix.audio_mask):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite prefix")
    if (not ((prefix.text_mask == 0) | (prefix.text_mask == 1)).all()
            or not ((prefix.audio_mask == 0) | (prefix.audio_mask == 1)).all()
            or not (prefix.text_mask + prefix.audio_mask == 1).all()):
        raise ValueError("prefix masks must be complementary binary masks")


def make_prefix(model, text, *, reference_wav=None):
    """Official reference-isolation layout, local 16kHz reference encoding."""
    if not text.strip():
        raise ValueError("nonempty text required")
    device = next(model.parameters()).device
    from .runtime import require_active_gpu1_lease
    require_active_gpu1_lease(device)
    tokens = torch.tensor([list(model.text_tokenizer(text)) + [model.audio_start_token]],
                          dtype=torch.long, device=device)
    feats = torch.zeros(1, tokens.shape[1], model.patch_size, model.feat_dim,
                        device=device, dtype=model._dtype())
    text_mask = torch.ones_like(tokens)
    audio_mask = torch.zeros_like(tokens)
    if reference_wav:
        with torch.no_grad():
            ref = model._encode_wav(str(reference_wav)).to(device=device, dtype=feats.dtype)
        rt, rf, tm, am = model._make_ref_prefix(ref, device)
        tokens = torch.cat((rt[None].long(), tokens), dim=1)
        feats = torch.cat((rf[None].to(feats), feats), dim=1)
        text_mask = torch.cat((tm[None], text_mask), dim=1)
        audio_mask = torch.cat((am[None], audio_mask), dim=1)
    prefix = Prefix(tokens, feats, text_mask, audio_mask)
    validate_prefix(model, prefix)
    return prefix


def fixed_noise(model, frames, seed):
    if type(frames) is not int or not 1 <= frames <= 64:
        raise ValueError("bounded qualification supports 1..64 patches")
    from .runtime import require_active_gpu1_lease
    require_active_gpu1_lease(next(model.parameters()).device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randn(1, model.feat_dim, model.patch_size, generator=gen)
                        for _ in range(frames)]).to(
        device=next(model.parameters()).device, dtype=model._dtype())


@contextmanager
def fsq_training_rounding(model):
    """Official FSQ STE has identical forward values; keep the backbone in eval."""
    old = model.fsq_layer.training
    model.fsq_layer.training = True
    try:
        yield
    finally:
        model.fsq_layer.training = old


def waveform(model, prefix, noise, *, steps=10, guidance=2., sway=1., cfg_zero_star=True):
    """Autograd-enabled autoregression; no stock generation wrappers/cache writes.

    Fixed duration replaces discrete stop decisions. The CFG adapter affects both
    estimator branches, matching stock VoxCPM's global ordinary LoRA behavior.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("waveform path cannot run inside inference_mode")
    from .runtime import require_active_gpu1_lease
    require_active_gpu1_lease(next(model.parameters()).device)
    validate_prefix(model, prefix)
    if (noise.ndim != 4 or noise.shape[1:] != (1, model.feat_dim, model.patch_size)
            or not 1 <= len(noise) <= 64 or not torch.isfinite(noise).all()):
        raise ValueError("finite native [frames,1,D,P] noise required")
    if type(steps) is not int or not 2 <= steps <= 32:
        raise ValueError("use 2..32 Euler steps; zero-star's first step is intentionally zero")
    if not all(math.isfinite(v) for v in (guidance, sway)) or guidance < 0 or not 0 <= sway <= 1:
        raise ValueError("finite nonnegative CFG and sway in [0,1] required")
    if prefix.tokens.shape[1] + len(noise) > model.config.max_length:
        raise ValueError("fixed sequence exceeds native context budget")
    feats, tokens = prefix.feats, prefix.tokens
    text_mask, audio_mask = prefix.text_mask, prefix.audio_mask
    output = []
    t = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=noise.dtype)
    t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    # Local mode only: no global model.train(), no dropout, no changed weights.
    with fsq_training_rounding(model):
        for z in noise:
            feat_embed = model.enc_to_lm_proj(model.feat_encoder(feats))
            scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.
            text_embed = model.base_lm.embed_tokens(tokens) * scale_emb
            combined = text_mask[..., None] * text_embed + audio_mask[..., None] * feat_embed
            enc, _ = model.base_lm(inputs_embeds=combined, is_causal=True)
            enc = model.fsq_layer(enc) * audio_mask[..., None] + enc * text_mask[..., None]
            residual, _ = model.residual_lm(inputs_embeds=model.fusion_concat_proj(
                torch.cat((enc, audio_mask[..., None] * feat_embed), dim=-1)), is_causal=True)
            mu = torch.cat((model.lm_to_dit_proj(enc[:, -1]),
                            model.res_to_dit_proj(residual[:, -1])), dim=-1)
            patch = model.feat_decoder.solve_euler(
                x=z.clone(), t_span=t, mu=mu, cond=feats[:, -1].transpose(1, 2).contiguous(),
                cfg_value=guidance, use_cfg_zero_star=cfg_zero_star).transpose(1, 2)
            output.append(patch)
            feats = torch.cat((feats, patch[:, None]), dim=1)
            tokens = torch.cat((tokens, tokens.new_zeros(1, 1)), dim=1)
            text_mask = torch.cat((text_mask, text_mask.new_zeros(1, 1)), dim=1)
            audio_mask = torch.cat((audio_mask, audio_mask.new_ones(1, 1)), dim=1)
    # Continuous latents [B,D,frames*P], native mono decoder [B,1,samples].
    latent = torch.cat(output, dim=1).transpose(1, 2).contiguous()
    decoder_dtype = next(model.audio_vae.decoder.parameters()).dtype
    audio = model.audio_vae.decode(latent.to(dtype=decoder_dtype)).squeeze(1).float()
    if audio.ndim != 2 or audio.shape[0] != 1 or not torch.isfinite(audio).all():
        raise ValueError("invalid native mono waveform")
    return audio


class VoxCPM2WaveBackend:
    family = "voxcpm2_native_waveform"

    def __init__(self, model, slider, *, model_identity):
        if not model_identity:
            raise ValueError("verified model identity or explicit random-fixture identity required")
        self.model, self.slider = model, slider
        self.model_identity = deepcopy(model_identity)
        self.network = AdapterParameters(slider.named_parameters(), slider.scaled)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("backbone must be frozen")

    def _run(self, row, prefix, scale):
        with self.network.scaled(scale):
            return waveform(self.model, prefix, row.noise, steps=row.steps,
                            guidance=row.guidance, sway=row.sway, cfg_zero_star=row.cfg_zero_star)

    def prepare(self, row):
        if row.scale != 1.:
            raise ValueError("only positive unipolar target at +1 is trained")
        row = WaveRow(_copy_prefix(row.neutral), _copy_prefix(row.positive), row.noise.detach().clone(),
                      row.steps, row.guidance, row.sway, row.cfg_zero_star, row.scale)
        with torch.no_grad():
            target = self._run(row, row.positive, 0.).detach().clone()
            neutral = self._run(row, row.neutral, 0.).detach().clone()
        if target.shape != neutral.shape:
            raise ValueError("teacher and neutral native durations must match")
        return PreparedWave(row, target, neutral)

    def generated(self, prepared):
        result = self._run(prepared.row, prepared.row.neutral, prepared.row.scale)
        if result.shape != prepared.target.shape:
            raise ValueError("generated duration differs")
        return result

    def gradient_evidence(self, prepared):
        """Fresh diagnostic VJP, independent of optimizer-owned .grad buffers."""
        with torch.enable_grad():
            response = self.generated(prepared)
            named = tuple(self.slider.named_parameters())
            gradients = torch.autograd.grad(response.sum(), [p for _, p in named], allow_unused=True)
        rows = {name: dict(connected=g is not None,
                          finite=g is not None and bool(torch.isfinite(g).all()),
                          l1=float(g.detach().abs().sum()) if g is not None else None)
                for (name, _), g in zip(named, gradients)}
        return dict(method="fresh post-update native waveform VJP; no optimizer or gradient-buffer mutation",
                    parameters=rows, connected_finite=all(v["connected"] and v["finite"] for v in rows.values()))

    def metadata(self):
        return dict(protocol="voxcpm2-fixed-noise-waveform-v1", model_identity=self.model_identity,
                    adapter=self.slider.metadata(), response_domain="continuous_vector",
                    response_coordinates="native mono 48kHz samples; no projection/truncation",
                    sampler="official Euler + full-prefix causal autoregression; fixed duration",
                    fsq_backward="official training straight-through rounding; exact eval forward values",
                    cfg_adapter_branch="both conditional and unconditional; matches stock LoRA",
                    encoder_sample_rate=16000, decoder_sample_rate=48000,
                    trained_scales=[1.0], audio_render_qualified=False, quality_qualified=False)
