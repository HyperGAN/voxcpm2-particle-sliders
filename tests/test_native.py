import copy
import json

import pytest
import torch
from voxcpm2_particle_sliders.adapter import ParticleAdapter
from voxcpm2_particle_sliders.backend import (
    make_prefix, fixed_noise, waveform, WaveRow, VoxCPM2WaveBackend, Prefix)
from voxcpm2_particle_sliders.artifacts import save_native, load_native
from voxcpm2_particle_sliders.distill import capture, matrices, export_distilled, calibration_profile


def render(model, prefix, noise):
    return waveform(model, prefix, noise, steps=3)


@pytest.mark.parametrize("with_reference", [False, True])
def test_native_stock_fixed_length_equivalence(model, record_property, with_reference):
    prefix = make_prefix(model, "one two")
    if with_reference:
        rt, rf, tm, am = model._make_ref_prefix(torch.randn(2, model.patch_size, model.feat_dim), "cpu")
        prefix = Prefix(torch.cat((rt[None].long(), prefix.tokens), dim=1),
                        torch.cat((rf[None], prefix.feats), dim=1),
                        torch.cat((tm[None], prefix.text_mask), dim=1),
                        torch.cat((am[None], prefix.audio_mask), dim=1))
    noise = fixed_noise(model, 2, 1234)
    with torch.no_grad():
        ours = render(model, prefix, noise)
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        latent, _ = model.inference(prefix.tokens, prefix.text_mask, prefix.feats, prefix.audio_mask,
                                    min_len=999, max_len=2, inference_timesteps=3, cfg_value=2.)
        stock = model.audio_vae.decode(latent).squeeze(1)
    torch.testing.assert_close(ours, stock, atol=3e-5, rtol=3e-5)
    record_property("fixture_stock_waveform_max_abs_error", float((ours-stock).abs().max()))
    assert model.fsq_layer.training is False


@pytest.mark.parametrize("hosts", [("dit",), ("dit", "lm")])
def test_gradient_ownership_and_zero_bypass(model, hosts, record_property):
    prefix = make_prefix(model, "one two")
    noise = fixed_noise(model, 2, 11)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        base = render(model, prefix, noise)
    adapter = ParticleAdapter(model, hosts=hosts, rank=4, alpha=4., count=8)
    backend = VoxCPM2WaveBackend(model, adapter, model_identity={"random_fixture": True})
    # Nonzero up is necessary to witness routed-bank gradients at initialization.
    # Zero up remains the product initialization, preserving the base exactly.
    with adapter.scaled(1.), torch.no_grad():
        torch.testing.assert_close(render(model, prefix, noise), base, atol=0, rtol=0)
    with torch.no_grad():
        for branch in adapter.branches:
            branch.lora_up.weight.normal_(std=.01)
    with adapter.scaled(0.), torch.no_grad():
        assert torch.equal(render(model, prefix, noise), base)
    with adapter.scaled(1.):
        loss = render(model, prefix, noise).square().mean()
        loss.backward()
    groups = dict(up=[], down=[], routed=[], particles=[])
    for name, parameter in adapter.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        group = ("particles" if name == "particles" else "up" if "lora_up" in name
                 else "down" if "lora_down" in name else "routed")
        groups[group].append(parameter.grad.abs().sum())
    assert all(float(torch.stack(v).sum()) > 0 for v in groups.values())
    for group, values in groups.items():
        record_property("random_fixture_" + group + "_gradient_l1", float(torch.stack(values).sum()))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(p.dtype == torch.float32 for p in backend.network.parameters())
    adapter.close()


def test_native_export_and_stock_distilled_loader(model, model_factory, tmp_path):
    from voxcpm.model.voxcpm2 import LoRAConfig
    stock = model_factory()
    stock.load_state_dict(model.state_dict())
    adapter = ParticleAdapter(model, rank=4, alpha=6., count=8)
    with torch.no_grad():
        for branch in adapter.branches: branch.lora_up.weight.normal_(std=.02)
    identity = {"random_fixture": True}
    teacher = tmp_path / "teacher.safetensors"
    save_native(adapter, teacher, identity=identity, provenance={"data": "random CPU fixture"})
    expected = {k: v.clone() for k, v in adapter.state_dict().items()}
    with torch.no_grad(): adapter.particles.add_(1)
    load_native(adapter, teacher, identity=identity)
    assert all(torch.equal(v, adapter.state_dict()[k]) for k, v in expected.items())
    with pytest.raises(ValueError, match="identity"):
        load_native(adapter, teacher, identity={"wrong": True})
    prefix = make_prefix(model, "one two")
    captures = []
    for seed in (33, 44):
        with torch.no_grad(), adapter.scaled(1.), capture(adapter, model, max_rows=32) as bucket:
            render(model, prefix, fixed_noise(model, 2, seed))
        captures.append(matrices(bucket))
    output = tmp_path / "distill"
    report = export_distilled(adapter, *captures, output, teacher=teacher, model_identity=identity,
                              train_ids=["train-text-seed33"], heldout_ids=["dev-text-seed44"],
                              data_provenance={"qualification": "random CPU fixture only"},
                              calibration=calibration_profile(frames=2, steps=3, cfg_value=2., cfg_zero_star=True))
    stock.lora_config = LoRAConfig(**report["lora_config"])
    stock._apply_lora()
    loaded, skipped = stock.load_lora_weights(str(output), device="cpu")
    assert len(loaded) == len(adapter.names) * 2 and not skipped
    assert all(".lora_A" in key or ".lora_B" in key for key in loaded)
    assert report["approximate"] and not report["identity_qualified"]
    with torch.no_grad():
        audio = render(stock, prefix, fixed_noise(stock, 2, 33))
    assert torch.isfinite(audio).all()
    adapter.close()


def test_wide_waveform_guard_is_preserved():
    from particle_sliders.pg09 import validate_pg09_domain
    with pytest.raises(ValueError, match="width >=170"):
        validate_pg09_domain((1, 48000), response_domain="continuous_vector")


def test_preparation_snapshots_conditioning(model):
    adapter = ParticleAdapter(model, rank=4, count=8)
    prefix = make_prefix(model, "one two")
    positive = make_prefix(model, "one two three")
    noise = fixed_noise(model, 2, 4)
    backend = VoxCPM2WaveBackend(model, adapter, model_identity={"random_fixture": True})
    prepared = backend.prepare(WaveRow(prefix, positive, noise, steps=3))
    prefix.tokens.fill_(2); noise.fill_(42)
    assert not torch.equal(prefix.tokens, prepared.row.neutral.tokens)
    assert not torch.equal(noise, prepared.row.noise)
    assert not prepared.target.requires_grad
    assert torch.isfinite(backend.generated(prepared)).all()
    adapter.close()
