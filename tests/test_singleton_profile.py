"""Product adoption of reviewed CPU proof; no native model/game/admission."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import math

import pytest
import torch
from torch import nn

pytest.importorskip("hndl")
if version("hndl") != "0.10.0" or version("particle-sliders-core") != "0.12.0":
    pytest.skip("requires the exact modern candidate cohort", allow_module_level=True)

from voxcpm2_particle_sliders import singleton_binding as binding
from voxcpm2_particle_sliders.hndl_binding import HNDLWaveCritic, PROFILE as ORIGINAL
from voxcpm2_particle_sliders.adapter import ParticleAdapter
from voxcpm2_particle_sliders.artifacts import save_native, load_native


@pytest.fixture(autouse=True)
def cpu_only():
    threads = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)
    assert not torch.cuda.is_initialized()


def pair(width, dtype):
    old, new = HNDLWaveCritic(width, 3).to(dtype), binding.HNDLSingletonWaveCritic(width, 3).to(dtype)
    before = deepcopy(old.state_dict())
    # Test-only mathematical mapping, never a production checkpoint converter.
    with torch.no_grad():
        for name, value in new.named_parameters():
            original = dict(old.named_parameters())[name]
            value.copy_(original.unsqueeze(2) if value.ndim == 4 else original)
    assert all(torch.equal(v, old.state_dict()[n]) for n, v in before.items())
    return old, new


@pytest.mark.parametrize("width", [7680, 30720])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_product_graph_exact_forward_all_vjps_and_second_derivatives(width, dtype):
    old, new = pair(width, dtype)
    x = torch.randn(2, width, dtype=dtype, requires_grad=True); y = x.detach().clone().requires_grad_()
    c = torch.randn(2, 3, dtype=dtype, requires_grad=True); d = c.detach().clone().requires_grad_()
    a, b = old(x, c), new(y, d)
    assert torch.equal(a, b)
    left = torch.autograd.grad(a.sum(), (x, c, *old.parameters()), create_graph=True)
    right = torch.autograd.grad(b.sum(), (y, d, *new.parameters()), create_graph=True)
    for l, r in zip(left, right): assert torch.equal(l, r.squeeze(2) if r.ndim == 4 else r)
    lh = torch.autograd.grad(left[0].square().sum(), (x, c, *old.parameters()), allow_unused=True)
    rh = torch.autograd.grad(right[0].square().sum(), (y, d, *new.parameters()), allow_unused=True)
    for l, r in zip(lh, rh):
        if l is None: assert r is None
        else: assert torch.equal(l, r.squeeze(2) if r.ndim == 4 else r)
    assert old.graph is not new.graph and old.graph["audio0"].weight.ndim == 3
    assert all(isinstance(new.graph[n], nn.Conv2d) for n in ("audio0", "audio2"))
    with pytest.raises(ValueError, match="native waveform"):
        new(y[:, None], d)


@pytest.mark.parametrize("name", ["wave_critic_singleton.hndl", "hndl_binding.py", "hndl_profile.json"])
def test_candidate_rejects_local_graph_or_base_source_drift(name, monkeypatch):
    original = binding._sha
    monkeypatch.setattr(binding, "_sha", lambda p: "0"*64 if Path(p).name == name else original(p))
    with pytest.raises(ValueError, match="source identity|graph differs"):
        binding.HNDLSingletonWaveCritic(7680, 3)


def test_candidate_inherits_complete_shared_core_closure_gate(monkeypatch):
    from voxcpm2_particle_sliders import hndl_binding as base
    original = base._sha
    monkeypatch.setattr(base, "_sha", lambda p: "0"*64 if Path(p).name == "pg09.py" else original(p))
    with pytest.raises(ValueError, match="shared core source differs"):
        binding.verified_runtime()


def test_candidate_drift_refuses_before_adapter_attachment(model, monkeypatch):
    flags = [p.requires_grad for p in model.parameters()]
    forwards = [m.forward for n, m in model.named_modules() if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    original = binding._sha
    monkeypatch.setattr(binding, "_sha", lambda p: "0"*64 if Path(p).name == "wave_critic_singleton.hndl" else original(p))
    with pytest.raises(ValueError, match="graph differs"):
        ParticleAdapter(model, network_profile=binding.PROFILE)
    assert flags == [p.requires_grad for p in model.parameters()]
    assert forwards == [m.forward for n, m in model.named_modules() if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]


def test_new_profile_cli_refuses_before_model_allocation(tmp_path):
    import json
    from voxcpm2_particle_sliders.cli import main
    (tmp_path/"config.json").write_text(json.dumps(dict(patch_size=4)))
    with pytest.raises(ValueError, match=">=170"):
        main(["train", "--network-profile", binding.PROFILE, "--model", str(tmp_path),
              "--revision", "a"*40, "--frames", "1", "--output", str(tmp_path/"run")])
    assert not (tmp_path/"run").exists()


def test_preflight_identifies_candidate_without_claiming_admission(tmp_path, monkeypatch):
    import json
    import voxcpm2_particle_sliders.cli as cli
    monkeypatch.setattr(cli, "resources", lambda device: {"explicit_cpu_fixture": True})
    output = tmp_path/"preflight.json"
    cli.main(["preflight", "--network-profile", binding.PROFILE, "--device", "cpu", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["candidate_identity"]["profile"] == binding.PROFILE
    assert result["candidate_identity"]["production_admitted"] is False
    assert result["waveform_gan_admission"]["status"] == "blocked"


@pytest.mark.parametrize("profile", ["auto", "default"])
def test_real_optimizer_geometry_and_exact_candidate_checkpoint_next_update(profile):
    from particlegan import get_recipe
    from particle_sliders.optimizer_policy import configure_dense_profile
    old, new = pair(7680, torch.float32)
    recipe, _ = configure_dense_profile(get_recipe("bispf_optimized", total_steps=3), profile)
    with pytest.raises(ValueError, match="high-rank"): recipe.make_critic_optimizer(old)
    optimizer = recipe.make_critic_optimizer(new)
    metadata = [g["dualnorm_convolution"] for g in optimizer.param_groups if "dualnorm_convolution" in g]
    assert len(metadata) == 2
    for meta, inputs in zip(metadata, (1, 16)):
        assert meta["kernel_size"] == [1, 15] and meta["groups"] == 1
        assert meta["in_channels"] == inputs and meta["out_channels"] == 16
        assert meta["layout"] == "conv2d_out_in"
        assert math.sqrt(meta["out_channels"] / inputs) / math.prod(meta["kernel_size"]) == (4/15 if inputs == 1 else 1/15)
    for i, p in enumerate(new.parameters()): p.grad = torch.full_like(p, .01*(i+1))
    optimizer.step()
    weights, state = deepcopy(new.state_dict()), deepcopy(optimizer.state_dict())
    clone = binding.HNDLSingletonWaveCritic(7680, 3); clone.load_state_dict(weights)
    restored = recipe.make_critic_optimizer(clone); restored.load_state_dict(state)
    for i, (a, b) in enumerate(zip(new.parameters(), clone.parameters())):
        a.grad = torch.full_like(a, .02*(i+1)); b.grad = a.grad.clone()
    optimizer.step(); restored.step()
    assert all(torch.equal(a, b) for a, b in zip(new.parameters(), clone.parameters()))


def test_distinct_native_identity_refuses_old_packet_and_restores_new(model, tmp_path):
    identity = {"explicit_random_cpu_fixture": True}
    old = ParticleAdapter(model, network_profile=ORIGINAL, seed=171)
    path = tmp_path/"original.safetensors"
    save_native(old, path, identity=identity, provenance={"fixture": "old cohort"})
    original_metadata = old.metadata(); old.close()
    new = ParticleAdapter(model, network_profile=binding.PROFILE, seed=171)
    try:
        assert new.metadata() != original_metadata
        before = deepcopy(new.state_dict())
        with pytest.raises(ValueError, match="identity/projection architecture"):
            load_native(new, path, identity=identity)
        assert all(torch.equal(v, new.state_dict()[n]) for n, v in before.items())
        fresh = tmp_path/"singleton.safetensors"
        save_native(new, fresh, identity=identity, provenance={"fixture": "fresh candidate"})
        with torch.no_grad(): new.particles.add_(1.)
        load_native(new, fresh, identity=identity)
        assert all(torch.equal(v, new.state_dict()[n]) for n, v in before.items())
    finally: new.close()
    original = ParticleAdapter(model, network_profile=ORIGINAL, seed=171)
    try:
        before = deepcopy(original.state_dict())
        with pytest.raises(ValueError, match="identity/projection architecture"):
            load_native(original, fresh, identity=identity)
        assert all(torch.equal(v, original.state_dict()[n]) for n, v in before.items())
        load_native(original, path, identity=identity)  # Original compatible packet stays valid.
    finally: original.close()


@pytest.mark.parametrize("width", [7680, 30720, 307200])
def test_new_candidate_and_real_product_make_game_keep_waveform_refusal(width):
    from particle_sliders.pg09 import validate_pg09_domain
    from voxcpm2_particle_sliders.training import make_game
    options = binding.engine_options(binding.PROFILE)
    assert options == dict(pg_source_profile="release-0.9.4", penalty_input_domain="auto")
    assert binding.verified_runtime()[2]["production_admitted"] is False
    with pytest.raises(ValueError, match=">=170"):
        validate_pg09_domain((1, width), response_domain="continuous_vector", **options)
    backend = SimpleNamespace(slider=SimpleNamespace(network_profile=binding.PROFILE))
    with pytest.raises(ValueError, match=">=170"):
        make_game(backend, [SimpleNamespace(target=torch.zeros(1, width))],
                  conditions=torch.zeros(1, 3), total_steps=2)
