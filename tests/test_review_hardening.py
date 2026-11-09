"""Claude review gates: official random CPU modules, never pretrained qualification."""
import json

import pytest
import torch
from safetensors.torch import load_file

from voxcpm2_particle_sliders.adapter import ParticleAdapter
from voxcpm2_particle_sliders.artifacts import save_native
from voxcpm2_particle_sliders.backend import Prefix, make_prefix, fixed_noise, waveform, fsq_training_rounding
from voxcpm2_particle_sliders.distill import capture, matrices, export_distilled, calibration_profile
from voxcpm2_particle_sliders.evaluation import waveform_metrics


def reference_prefix(model, enabled):
    prefix = make_prefix(model, "one two")
    if not enabled:
        return prefix
    rt, rf, tm, am = model._make_ref_prefix(torch.randn(2, model.patch_size, model.feat_dim), "cpu")
    return Prefix(torch.cat((rt[None].long(), prefix.tokens), 1),
                  torch.cat((rf[None], prefix.feats), 1),
                  torch.cat((tm[None], prefix.text_mask), 1),
                  torch.cat((am[None], prefix.audio_mask), 1))


def test_stock_comparison_preserves_fixed_duration_for_short_text(model):
    from voxcpm2_particle_sliders.evaluation import stock_fixed_waveform
    # Official generate would independently cap one-token text at 16 patches
    # under its default ratio, even when retries are disabled.
    calibration = calibration_profile(frames=24, steps=3, cfg_value=2., cfg_zero_star=True)
    record = dict(text="one", seed=133)
    rng = torch.get_rng_state().clone()
    audio = stock_fixed_waveform(model, record, calibration)
    assert torch.equal(rng, torch.get_rng_state())
    assert audio.shape == (1, 24 * model.patch_size * model._decode_chunk_size)
    assert torch.isfinite(audio).all()
    with pytest.raises(ValueError, match="zero-star/sway"):
        stock_fixed_waveform(model, record, dict(calibration, cfg_zero_star=False))


@pytest.mark.parametrize("with_reference", [False, True])
def test_per_patch_full_prefix_kv_parity(model, monkeypatch, record_property, with_reference):
    prefix = reference_prefix(model, with_reference)
    noise = fixed_noise(model, 4, 1234)
    original_solver = model.feat_decoder.solve_euler
    traces, embeddings, inputs = [], [], []
    def traced_solver(**kwargs):
        output = original_solver(**kwargs)
        traces.append({k: v.detach().clone() if torch.is_tensor(v) else v
                       for k, v in dict(kwargs, patch=output).items()})
        return output
    monkeypatch.setattr(model.feat_decoder, "solve_euler", traced_solver)
    h1 = model.feat_encoder.register_forward_pre_hook(lambda m, args: inputs.append(args[0].detach().clone()))
    h2 = model.enc_to_lm_proj.register_forward_hook(lambda m, args, out: embeddings.append(out.detach().clone()))
    try:
        with torch.no_grad():
            waveform(model, prefix, noise, steps=3, guidance=2., cfg_zero_star=True)
        ours, our_embeddings, our_inputs = traces[:], embeddings[:], inputs[:]
        traces.clear(); embeddings.clear(); inputs.clear()
        with torch.random.fork_rng():
            torch.manual_seed(1234)
            model.inference(prefix.tokens, prefix.text_mask, prefix.feats, prefix.audio_mask,
                            min_len=999, max_len=4, inference_timesteps=3, cfg_value=2.)
        assert len(ours) == len(traces) == 4
        for i, (full, cached) in enumerate(zip(ours, traces)):
            for coordinate in ("mu", "patch", "cond", "x", "t_span"):
                torch.testing.assert_close(full[coordinate], cached[coordinate], atol=3e-6, rtol=3e-5)
                record_property(f"patch{i}_{coordinate}_max_abs", float(
                    (full[coordinate] - cached[coordinate]).abs().max()))
            assert full["cfg_value"] == cached["cfg_value"] == 2.
            assert full["use_cfg_zero_star"] is cached["use_cfg_zero_star"] is True
            assert torch.equal(full["x"], noise[i])
            if i:
                # Cached encoder consumes previous patch alone; full-prefix path
                # consumes all history. Compare the actual fed-back final row.
                torch.testing.assert_close(our_inputs[i][:, -1], inputs[i][:, -1], atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(our_embeddings[i][:, -1], embeddings[i][:, -1], atol=3e-6, rtol=3e-5)
                record_property(f"patch{i}_feedback_embedding_max_abs", float(
                    (our_embeddings[i][:, -1] - embeddings[i][:, -1]).abs().max()))
    finally:
        h1.remove(); h2.remove()


def test_fsq_correct_bounded_surrogate_derivative_and_modes(model, record_property):
    # STE replaces round only. The actual derivative still traverses in_proj,
    # tanh (bounded, nonidentity), then out_proj.
    hidden = (3 * torch.randn(2, 16)).requires_grad_()
    tangent = torch.randn(2, 16)
    model.fsq_layer.eval()
    rounded = model.fsq_layer(hidden)
    with fsq_training_rounding(model):
        actual = model.fsq_layer(hidden)
        actual_gradient, = torch.autograd.grad(actual, hidden, grad_outputs=tangent)
    surrogate = model.fsq_layer.out_proj(torch.tanh(model.fsq_layer.in_proj(hidden)))
    expected_gradient, = torch.autograd.grad(surrogate, hidden, grad_outputs=tangent)
    assert torch.equal(actual, rounded)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
    assert actual_gradient.abs().sum() > 0
    record_property("fsq_bounded_surrogate_gradient_max_abs_error",
                    float((actual_gradient - expected_gradient).abs().max()))
    before = [(id(module), module.training) for module in model.modules()]
    for original_mode in (False, True):
        model.fsq_layer.training = original_mode
        with pytest.raises(RuntimeError, match="restoration"):
            with fsq_training_rounding(model):
                assert model.fsq_layer.training is True
                raise RuntimeError("restoration")
        assert model.fsq_layer.training is original_mode
    model.fsq_layer.training = False
    assert before == [(id(module), module.training) for module in model.modules()]


@pytest.mark.parametrize("hosts", [("dit",), ("dit", "lm")])
def test_solver_decoder_feedback_paths_and_real_zero_up_update(model, monkeypatch, hosts, record_property):
    slider = ParticleAdapter(model, hosts=hosts, rank=4, count=8)
    prefix = make_prefix(model, "one two")
    noise = fixed_noise(model, 3, 41)
    patches, latents = [], []
    solver, decode = model.feat_decoder.solve_euler, model.audio_vae.decode
    def retain_solver(**kwargs):
        value = solver(**kwargs)
        if value.requires_grad: value.retain_grad()
        patches.append(value)
        return value
    def retain_decode(value, *args, **kwargs):
        if value.requires_grad: value.retain_grad()
        latents.append(value)
        return decode(value, *args, **kwargs)
    monkeypatch.setattr(model.feat_decoder, "solve_euler", retain_solver)
    monkeypatch.setattr(model.audio_vae, "decode", retain_decode)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        with slider.scaled(1.):
            audio = waveform(model, prefix, noise, steps=3)
            audio.square().mean().backward()
        for name, parameter in slider.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if "lora_up" not in name:
                assert torch.count_nonzero(parameter.grad) == 0, name
        up_norm = sum(float(b.lora_up.weight.grad.abs().sum()) for b in slider.branches)
        assert up_norm > 0
        record_property("production_zero_up_first_gradient_l1", up_norm)
        # One explicit CPU diagnostic SGD step on up only, not a GAN update.
        optimizer = torch.optim.SGD([b.lora_up.weight for b in slider.branches], lr=.1)
        optimizer.step()
        slider.zero_grad(set_to_none=True)
        patches.clear(); latents.clear()
        with slider.scaled(1.):
            audio = waveform(model, prefix, noise, steps=3)
            latent = latents[-1]
            # A cotangent on the final patch ONLY must reach the first patch.
            # This fails if the autoregressive feedback history is detached.
            feedback, = torch.autograd.grad(latent[..., -model.patch_size:].square().sum(),
                                            patches[0], retain_graph=True)
            assert torch.isfinite(feedback).all() and feedback.abs().sum() > 0
            decoder_gradient, = torch.autograd.grad(audio.square().mean(), latent, retain_graph=True)
            assert torch.isfinite(decoder_gradient).all() and decoder_gradient.abs().sum() > 0
            # Isolate the following assertions from retained intermediate grads
            # populated by the two explicit autograd.grad diagnostics above.
            latent.grad = None
            for patch in patches: patch.grad = None
            audio.square().mean().backward()
        assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
        for index, patch in enumerate(patches):
            assert patch.grad is not None and torch.isfinite(patch.grad).all() and patch.grad.abs().sum() > 0
            record_property(f"solver_patch{index}_through_decoder_gradient_l1", float(patch.grad.abs().sum()))
        for host in hosts:
            selected = [(name, branch) for name, branch in zip(slider.names, slider.branches)
                        if (name.startswith("feat_decoder") if host == "dit" else
                            name.startswith(("base_lm.", "residual_lm.")))]
            for name, branch in selected:
                for leaf_name, parameter in branch.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (name, leaf_name)
            down_norm = sum(float(b.lora_down.weight.grad.abs().sum()) for _, b in selected)
            assert down_norm > 0, host
            record_property(f"after_one_up_update_{host}_down_gradient_l1", down_norm)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.items())
        record_property("first_patch_to_final_patch_feedback_gradient_l1", float(feedback.abs().sum()))
        record_property("native_decoder_waveform_to_latent_gradient_l1", float(decoder_gradient.abs().sum()))
    finally:
        slider.close()


def test_distill_fixed_scale_heldout_and_stock_strengths(model, model_factory, tmp_path, record_property):
    from voxcpm.model.voxcpm2 import LoRAConfig
    stock = model_factory(); stock.load_state_dict(model.state_dict())
    slider = ParticleAdapter(model, rank=4, alpha=6., count=8)
    with torch.no_grad():
        for branch in slider.branches: branch.lora_up.weight.normal_(std=.02)
    teacher = tmp_path / "teacher.safetensors"
    save_native(slider, teacher, identity={"random_fixture": True}, provenance={"fixture": True})
    calibration = calibration_profile(frames=3, steps=4, cfg_value=2., cfg_zero_star=True)
    captured = []
    for text, seed in (("fit one", 33), ("held out longer text", 44)):
        with torch.no_grad(), slider.scaled(1.), capture(slider, model, max_rows=64) as bucket:
            waveform(model, make_prefix(model, text), fixed_noise(model, 3, seed), steps=4,
                     guidance=calibration["cfg_value"], cfg_zero_star=calibration["cfg_zero_star"])
        captured.append(matrices(bucket))
    with slider.scaled(2.), pytest.raises(ValueError, match="strength 1"):
        with capture(slider, model): pass
    assert all(b.scale == 1.5 and b.multiplier == 0 for b in slider.branches)
    output = tmp_path / "distill"
    report = export_distilled(slider, *captured, output, teacher=teacher,
        model_identity={"random_fixture": True}, train_ids=["fit-text-seed33"], heldout_ids=["dev-other-text-seed44"],
        data_provenance={"fixture": True}, calibration=calibration)
    assert report["calibration"] == calibration
    assert report["heldout_bottleneck"]["r2"] is not None
    assert report["heldout_projection"]["r2"] is not None
    assert all(p["heldout_rows"] >= 2 for p in report["projections"])
    for coordinate in ("heldout_bottleneck", "heldout_projection"):
        for metric in ("relative_mse", "r2"):
            record_property(f"{coordinate}_{metric}", report[coordinate][metric])
    state = load_file(str(output / "lora_weights.safetensors"))
    stock.lora_config = LoRAConfig(**report["lora_config"]); stock._apply_lora()
    loaded, skipped = stock.load_lora_weights(str(output), device="cpu")
    assert not skipped and len(loaded) == len(slider.names) * 2
    prefix, noise = make_prefix(model, "held out longer text"), fixed_noise(model, 3, 55)
    native_waveforms, unit_deltas = {}, {}
    stock_modules = dict(stock.named_modules())
    with torch.no_grad():
        for strength in (0., 1., 2.):
            with slider.scaled(strength):
                native_waveforms[strength] = waveform(model, prefix, noise, steps=4,
                    guidance=calibration["cfg_value"], cfg_zero_star=calibration["cfg_zero_star"])
                for index, (name, branch) in enumerate(zip(slider.names, slider.branches)):
                    layer = stock_modules[name]; layer.scaling.fill_(layer._base_scaling * strength)
                    x = captured[1][name][0]
                    base = branch.org_forward(x)
                    native_delta = (branch.forward(x) - base)
                    distilled_delta = layer(x) - base
                    fitted_delta = strength * (x @ state[name + ".lora_A"].T) @ state[name + ".lora_B"].T
                    torch.testing.assert_close(distilled_delta, fitted_delta, atol=1e-6, rtol=1e-4)
                    if strength == 0:
                        assert torch.equal(layer(x), base) and torch.count_nonzero(native_delta) == 0
                    else:
                        # Approximation error must scale with runtime strength;
                        # fixed native alpha/r remains 1.5 at every strength.
                        if strength == 1.:
                            unit_deltas[name] = (native_delta.clone(), distilled_delta.clone())
                            record_property(f"projection{index}_heldout_delta_mse", float(
                                (native_delta-distilled_delta).square().mean()))
                        else:
                            torch.testing.assert_close(native_delta, 2 * unit_deltas[name][0], atol=1e-6, rtol=1e-4)
                            torch.testing.assert_close(distilled_delta, 2 * unit_deltas[name][1], atol=1e-6, rtol=1e-4)
                    assert branch.scale == 1.5
            distilled_audio = waveform(stock, prefix, noise, steps=4,
                guidance=calibration["cfg_value"], cfg_zero_star=calibration["cfg_zero_star"])
            metrics = waveform_metrics(native_waveforms[strength], distilled_audio)
            record_property(f"fixture_native_distilled_strength{strength:g}_relative_mse", metrics["relative_mse"])
            if strength == 0:
                assert torch.equal(native_waveforms[strength], distilled_audio)
    slider.close()
