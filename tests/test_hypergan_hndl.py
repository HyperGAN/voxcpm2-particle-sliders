"""Opt-in genuine modern shared/HyperGAN/HNDL tests; random CPU modules only."""
from importlib.metadata import version

import pytest
import torch
from torch import nn

pytest.importorskip("hndl")
import particle_sliders
if version("hndl") != "0.10.0" or particle_sliders.__version__ != "0.12.0":
    pytest.skip("requires the separately pinned owner core0.12/HNDL0.10 candidate cohort", allow_module_level=True)

from voxcpm2_particle_sliders.hndl_binding import PROFILE, HNDLWaveCritic
from voxcpm2_particle_sliders.adapter import ParticleAdapter
from voxcpm2_particle_sliders.backend import make_prefix, fixed_noise, waveform
from voxcpm2_particle_sliders.training import WaveCritic
from voxcpm2_particle_sliders.artifacts import save_native, load_native
from voxcpm2_particle_sliders.distill import capture, matrices, calibration_profile, export_distilled


def copy_affine(destination, source):
    with torch.no_grad():
        destination.weight.copy_(source.weight)
        if source.bias is not None: destination.bias.copy_(source.bias)


def copy_shared_projection(destination, source):
    copy_affine(destination.lora_down, source.lora_down)
    copy_affine(destination.lora_up, source.lora_up)
    for family, sequential in (("router", source.bridge.router), ("bridge", source.bridge.net)):
        for index in (0, 2, 4, 6):
            copy_affine(destination.graph[family + str(index)], sequential[index])


def test_profile_refuses_tampered_shared_source_before_attachment(model, monkeypatch):
    import voxcpm2_particle_sliders.hndl_binding as binding
    forwards = [m.forward for n, m in model.named_modules() if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    flags = [p.requires_grad for p in model.parameters()]
    sha = binding._sha
    monkeypatch.setattr(binding, "_sha", lambda p: "0"*64 if str(p).endswith("fibo_hndl.py") else sha(p))
    with pytest.raises(ValueError, match="shared HNDL binding source"):
        ParticleAdapter(model, network_profile=PROFILE)
    assert flags == [p.requires_grad for p in model.parameters()]
    assert forwards == [m.forward for n, m in model.named_modules() if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]


@pytest.mark.parametrize("name", ["pg09.py", "lifecycle.py", "inference.py",
                                  "distillation.py", "pg094_release.json", "LICENSE"])
def test_profile_refuses_tampered_complete_core_closure_before_attachment(model, monkeypatch, name):
    import voxcpm2_particle_sliders.hndl_binding as binding
    forwards = [m.forward for n, m in model.named_modules()
                if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    original = binding._sha
    monkeypatch.setattr(binding, "_sha", lambda p: "0"*64 if p.name == name else original(p))
    with pytest.raises(ValueError, match="installed shared core source differs"):
        ParticleAdapter(model, network_profile=PROFILE)
    assert forwards == [m.forward for n, m in model.named_modules()
                        if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))]
    assert flags == [p.requires_grad for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]


def test_core_closure_rejects_missing_and_extra_files_and_ignores_only_caches(tmp_path):
    import hashlib
    from voxcpm2_particle_sliders.hndl_binding import _verify_core_closure
    source = tmp_path / "module.py"
    source.write_bytes(b"# source fixture\n")
    expected = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    digest = _verify_core_closure(tmp_path, expected)
    cache = tmp_path / "__pycache__"
    cache.mkdir(); (cache / "module.cpython-312.pyc").write_bytes(b"cache fixture")
    assert _verify_core_closure(tmp_path, expected) == digest
    extra = tmp_path / "unexpected.so"
    extra.write_bytes(b"not executable; source-inventory fixture")
    with pytest.raises(ValueError, match="source inventory differs"):
        _verify_core_closure(tmp_path, expected)
    extra.unlink(); source.unlink()
    with pytest.raises(ValueError, match="source inventory differs"):
        _verify_core_closure(tmp_path, expected)


def test_seeded_modern_initialization_is_reproducible_and_scoped(model,model_factory):
    twin=model_factory();twin.load_state_dict(model.state_dict())
    rng=torch.get_rng_state().clone()
    a=ParticleAdapter(model,network_profile=PROFILE,seed=171)
    b=ParticleAdapter(twin,network_profile=PROFILE,seed=171)
    try:
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    finally:a.close();b.close()


def test_partial_hndl_attachment_conversion_failure_unwinds_and_restores_borrowed_state(model,monkeypatch):
    from particle_sliders.fibo_hndl import HNDLParticleProjection
    model.train();next(model.parameters()).requires_grad_(True)
    before=[m.forward for n,m in model.named_modules() if n.endswith(("q_proj","k_proj","v_proj","o_proj"))]
    flags=[p.requires_grad for p in model.parameters()]
    modes=[m.training for m in model.modules()]
    def fail(*args,**kwargs):raise RuntimeError("conversion fixture failure")
    monkeypatch.setattr(HNDLParticleProjection,"to",fail)
    with pytest.raises(RuntimeError,match="conversion fixture"):
        ParticleAdapter(model,network_profile=PROFILE,seed=171)
    assert before==[m.forward for n,m in model.named_modules() if n.endswith(("q_proj","k_proj","v_proj","o_proj"))]
    assert flags==[p.requires_grad for p in model.parameters()]
    assert modes==[m.training for m in model.modules()]


@pytest.mark.parametrize("hosts", [("dit",), ("dit", "lm")])
def test_shared_hndl_native_waveform_and_bank_vjp_parity(model, model_factory, hosts, record_property):
    twin = model_factory(); twin.load_state_dict(model.state_dict())
    old = ParticleAdapter(model, hosts=hosts)
    new = ParticleAdapter(twin, hosts=hosts, network_profile=PROFILE)
    assert "HNDLParticleProjection" == type(new.branches[0]).__name__
    with torch.no_grad():
        new.particles.copy_(old.particles)
        for destination, source in zip(new.branches, old.branches):
            source.lora_up.weight.normal_(std=.02)
            copy_shared_projection(destination, source)
    prefix = make_prefix(model, "plain modern text")
    noise = fixed_noise(model, 3, 101)
    frozen = {n:p.clone() for n,p in twin.state_dict().items()}
    try:
        with old.scaled(.5), new.scaled(.5):
            expected = waveform(model, prefix, noise, steps=3)
            actual = waveform(twin, prefix, noise, steps=3)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
            left, = torch.autograd.grad(expected.square().sum(), old.particles)
            right, = torch.autograd.grad(actual.square().sum(), new.particles, retain_graph=True)
            torch.testing.assert_close(right, left, atol=2e-6, rtol=1e-4)
            actual.square().sum().backward()
        assert right.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
        assert all(not p.requires_grad and p.grad is None for p in twin.parameters())
        assert all(torch.equal(p, twin.state_dict()[n]) for n,p in frozen.items())
        with new.scaled(0.), old.scaled(0.), torch.no_grad():
            assert torch.equal(waveform(twin, prefix, noise, steps=3), waveform(model, prefix, noise, steps=3))
        record_property("hndl_waveform_max_abs_error", float((actual-expected).detach().abs().max()))
        record_property("hndl_bank_vjp_max_abs_error", float((right-left).detach().abs().max()))
        record_property("hndl_bank_gradient_l1", float(right.abs().sum()))
    finally:
        new.close(); old.close()


def test_hndl_wave_critic_preserves_first_and_second_derivatives(record_property):
    old, new = WaveCritic(3), HNDLWaveCritic(384, 3)
    for name, layer in (("audio0",old.audio[0]),("audio2",old.audio[2]),("condition",old.condition),("score",old.score)):
        copy_affine(new.graph[name], layer)
    x = torch.randn(2,384,requires_grad=True); other=x.detach().clone().requires_grad_()
    condition=torch.randn(2,3)
    expected, actual = old(x,condition),new(other,condition)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    left, = torch.autograd.grad(expected.sum(), x, create_graph=True)
    right, = torch.autograd.grad(actual.sum(), other, create_graph=True)
    torch.testing.assert_close(right,left,atol=1e-6,rtol=1e-5)
    l2, = torch.autograd.grad(left.square().sum(),old.audio[0].weight)
    r2, = torch.autograd.grad(right.square().sum(),new.graph["audio0"].weight)
    torch.testing.assert_close(r2,l2,atol=1e-6,rtol=1e-4)
    record_property("hndl_critic_output_max_abs_error",float((actual-expected).detach().abs().max()))
    record_property("hndl_critic_double_backward_max_abs_error",float((r2-l2).abs().max()))


def test_public_up_hook_distill_native_reload_and_stock_loader(model, model_factory, tmp_path, record_property):
    from voxcpm.model.voxcpm2 import LoRAConfig
    stock = model_factory(); stock.load_state_dict(model.state_dict())
    slider = ParticleAdapter(model, network_profile=PROFILE)
    with torch.no_grad():
        for branch in slider.branches: branch.lora_up.weight.normal_(std=.02)
    teacher=tmp_path/"teacher.safetensors"
    identity={"random_cpu_fixture":True}
    save_native(slider,teacher,identity=identity,provenance={"test":"actual shared HNDL graph, no pretrained weights"})
    expected={k:v.clone() for k,v in slider.state_dict().items()}
    with torch.no_grad(): slider.particles.add_(2.)
    load_native(slider,teacher,identity=identity)
    assert all(torch.equal(v,slider.state_dict()[k]) for k,v in expected.items())
    captures=[]
    for text,seed in (("fit plain text",171),("held out different longer neutral words",2701)):
        with torch.no_grad(),slider.scaled(1.),capture(slider,model,max_rows=64) as bucket:
            waveform(model,make_prefix(model,text),fixed_noise(model,3,seed),steps=4)
        captures.append(matrices(bucket))
    # The actual shared graph called its public up module. Capture did not need
    # a new core API, a copied router or a private HNDL tensor endpoint.
    assert all(len(y)>2 and y.shape[1]==8 for x,y in captures[0].values())
    output=tmp_path/"distilled"
    report=export_distilled(slider,*captures,output,teacher=teacher,model_identity=identity,
        train_ids=["fit-171"],heldout_ids=["dev-2701"],data_provenance={"random_fixture":True},
        calibration=calibration_profile(frames=3,steps=4,cfg_value=2.,cfg_zero_star=True))
    stock.lora_config=LoRAConfig(**report["lora_config"]);stock._apply_lora()
    loaded,skipped=stock.load_lora_weights(str(output),device="cpu")
    assert len(loaded)==len(slider.names)*2 and not skipped
    with torch.no_grad():
        audio=waveform(stock,make_prefix(stock,"held out neutral words"),fixed_noise(stock,3,7),steps=4)
    assert torch.isfinite(audio).all() and not report["waveform_fidelity_qualified"]
    record_property("hndl_distill_heldout_projection_relative_mse",report["heldout_projection"]["relative_mse"])
    record_property("hndl_distill_heldout_projection_r2",report["heldout_projection"]["r2"])
    slider.close()
    # A historical adapter cannot load a modern graph packet under old identity.
    legacy=ParticleAdapter(model)
    try:
        with pytest.raises(ValueError,match="identity/projection architecture"):
            load_native(legacy,teacher,identity=identity)
    finally: legacy.close()


def test_capture_unwinds_only_its_hooks_on_native_failure(model):
    slider=ParticleAdapter(model,network_profile=PROFILE)
    branch=slider.branches[0]
    host=dict(model.named_modules())[slider.names[0]]
    baseline=(len(host._forward_pre_hooks),len(host._forward_hooks),len(branch.lora_up._forward_pre_hooks))
    try:
        with pytest.raises(RuntimeError,match="fixture failure"):
            with slider.scaled(1.),capture(slider,model):
                raise RuntimeError("fixture failure")
        assert baseline==(len(host._forward_pre_hooks),len(host._forward_hooks),len(branch.lora_up._forward_pre_hooks))
        assert all(b.multiplier==0 for b in slider.branches)
    finally: slider.close()


def test_close_attempts_other_owned_forwards_and_retains_only_conflicts_for_retry(model):
    slider=ParticleAdapter(model,network_profile=PROFILE)
    modules=dict(model.named_modules())
    name=slider.names[1];host=modules[name];branch=slider.branches[1]
    def foreign(value):return branch.forward(value)
    host.forward=foreign
    with pytest.raises(RuntimeError,match="ownership changed"):slider.close()
    assert host.forward is foreign and slider._modules_owned==[(host,branch)]
    for other,owned_branch in zip(slider.names,slider.branches):
        if other!=name:assert modules[other].forward==owned_branch.org_forward
    host.forward=branch.forward
    slider.close();slider.close()
    assert not slider._modules_owned
    assert all(modules[n].forward==b.org_forward for n,b in zip(slider.names,slider.branches))


def test_modern_profile_does_not_admit_wide_waveforms():
    from particle_sliders.pg09 import validate_pg09_domain
    from voxcpm2_particle_sliders.hndl_binding import engine_options
    options=engine_options()
    assert options==dict(pg_source_profile="release-0.9.4",penalty_input_domain="auto")
    with pytest.raises(ValueError,match="width >=170"):
        validate_pg09_domain((1,48000),response_domain="continuous_vector",**options)


def test_native_recipe_precision_is_declared_but_trainable_graph_is_fp32_and_ucd_is_inert(model):
    from dataclasses import replace
    from particlegan import get_recipe
    recipe=get_recipe("bispf_optimized",num_particles=128,z_dim=4,batch_size=2,total_steps=2)
    assert recipe.training_precision=="bf16_channels_last" and recipe.ucd_weight==.02 and recipe.ucd_target=="class"
    original,without_ucd=recipe.make_loss(),replace(recipe,ucd_weight=0.).make_loss()
    real,fake=torch.tensor([.1,.8]),torch.tensor([-.4,.2])
    assert torch.equal(original.d_loss(real,fake),without_ucd.d_loss(real,fake))
    assert torch.equal(original.g_loss(fake,real),without_ucd.g_loss(fake,real))
    slider=ParticleAdapter(model,network_profile=PROFILE)
    try:
        critic=HNDLWaveCritic(384,3)
        assert {p.dtype for p in slider.parameters()}=={torch.float32}
        assert {p.dtype for p in critic.parameters()}=={torch.float32}
    finally:slider.close()
