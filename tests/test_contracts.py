import json
import pytest
import torch
from voxcpm2_particle_sliders.backend import fixed_noise, make_prefix, waveform
from voxcpm2_particle_sliders.adapter import ParticleAdapter
from voxcpm2_particle_sliders.evaluation import waveform_metrics
from voxcpm2_particle_sliders.runtime import read_manifest, require_idle_gpu


def test_fsq_ste_forward_is_eval_exact_with_connected_backward(model):
    from voxcpm2_particle_sliders.backend import fsq_training_rounding
    hidden = torch.randn(1, 16, requires_grad=True)
    model.fsq_layer.eval()
    expected = model.fsq_layer(hidden)
    with fsq_training_rounding(model):
        actual = model.fsq_layer(hidden)
        assert torch.equal(expected, actual)
        actual.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert model.fsq_layer.training is False


def test_scale_and_mode_restore_after_exception(model):
    from voxcpm2_particle_sliders.backend import fsq_training_rounding
    slider = ParticleAdapter(model, rank=4, count=8)
    with pytest.raises(RuntimeError):
        with slider.scaled(2.), fsq_training_rounding(model):
            raise RuntimeError("test cleanup")
    assert all(b.multiplier == 0 for b in slider.branches)
    assert not model.fsq_layer.training
    slider.close(); slider.close()


def test_inference_mode_rejected_and_rng_untouched(model):
    prefix = make_prefix(model, "one two")
    rng = torch.get_rng_state().clone()
    noise = fixed_noise(model, 2, 1)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        waveform(model, prefix, noise, steps=3)


def test_metrics_exact_and_invalid_alignment():
    x = torch.sin(torch.linspace(0, 10, 1024))[None]
    result = waveform_metrics(x, x)
    assert result["mse"] == result["relative_mse"] == result["log_spectral_distance_db"] == 0
    with pytest.raises(ValueError): waveform_metrics(x, x[:, :-1])


def test_cli_train_refuses_before_model_load(tmp_path):
    from voxcpm2_particle_sliders.cli import main
    (tmp_path / "config.json").write_text(json.dumps(dict(patch_size=4)))
    with pytest.raises(ValueError, match="width >=170"):
        main(["train", "--model", str(tmp_path), "--revision", "a" * 40,
              "--output", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_manifest_refuses_renamed_train_dev_leakage(tmp_path):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"fixture only")
    row = dict(text="same", seed=1, positive_reference="ref.wav")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(rows=[dict(row, id="one", split="train"),
                                              dict(row, id="two", split="dev")])))
    with pytest.raises(ValueError, match="same calibration example"):
        read_manifest(manifest)


def test_control_teacher_uses_identical_frozen_reference_and_only_positive_text(tmp_path, monkeypatch):
    import voxcpm2_particle_sliders.runtime as runtime
    reference = tmp_path / "reference.fixture"
    reference.write_bytes(b"reference byte fixture; not generated audio")
    row = dict(id="train", split="train", text="The jacket has four pockets.", seed=1009,
               neutral_reference="reference.fixture", positive_reference="reference.fixture",
               positive_control="Very sleepy, clear projected speech.")
    manifest = tmp_path / "controlled.json"
    manifest.write_text(json.dumps(dict(rows=[row])))
    record = read_manifest(manifest)["rows"][0]
    calls = []
    def prefix(model, text, *, reference_wav):
        calls.append((text, reference_wav))
        return text
    monkeypatch.setattr(runtime, "make_prefix", prefix)
    noise = object()
    monkeypatch.setattr(runtime, "fixed_noise", lambda model, frames, seed: noise)
    result = runtime.native_row(object(), record, frames=4, steps=10)
    assert calls == [(row["text"], str(reference)),
                     ("(Very sleepy, clear projected speech.)" + row["text"], str(reference))]
    assert result.noise is noise and result.steps == 10
    assert runtime.positive_teacher_text(dict(text=row["text"])) == row["text"]
    # A named alternate asset is acceptable only when its actual bytes agree.
    twin = tmp_path / "same.fixture"; twin.write_bytes(reference.read_bytes())
    assert runtime.positive_teacher_text(dict(record, positive_reference=str(twin))) == result.positive
    twin.write_bytes(b"different voice fixture")
    with pytest.raises(ValueError, match="identical frozen reference bytes"):
        runtime.positive_teacher_text(dict(record, positive_reference=str(twin)))
    with pytest.raises(ValueError, match="same frozen reference"):
        runtime.positive_teacher_text(dict(record, neutral_reference=None))


@pytest.mark.parametrize("control", [None, "", " ", 2, "(sleepy)", "sleepy\nvoice"])
def test_control_teacher_rejects_ambiguous_description_before_prefix_build(control):
    from voxcpm2_particle_sliders.runtime import positive_teacher_text
    with pytest.raises(ValueError, match="positive_control"):
        positive_teacher_text(dict(text="plain text", positive_control=control))


def test_control_teacher_native_targets_share_reference_and_preserve_adapter_gradient(model, tmp_path, monkeypatch):
    from voxcpm2_particle_sliders.runtime import native_row
    from voxcpm2_particle_sliders.backend import VoxCPM2WaveBackend
    reference = tmp_path / "reference.fixture"
    reference.write_bytes(b"numeric prefix fixture only; not a recording")
    model.ref_audio_start_token, model.ref_audio_end_token = 3, 4
    ref_features = torch.randn(2, model.patch_size, model.feat_dim)
    monkeypatch.setattr(model, "_encode_wav", lambda path: ref_features.clone())
    record = dict(text="The jacket has pockets.", seed=1009,
                  positive_control="Drowsy, clear projected speech.",
                  neutral_reference=str(reference), positive_reference=str(reference))
    row = native_row(model, record, frames=2, steps=3)
    assert row.positive.tokens.shape[1] > row.neutral.tokens.shape[1]
    for name in ("tokens", "feats", "text_mask", "audio_mask"):
        assert torch.equal(getattr(row.positive, name)[:, :4], getattr(row.neutral, name)[:, :4])
    slider = ParticleAdapter(model, rank=4, count=8)
    try:
        backend = VoxCPM2WaveBackend(model, slider, model_identity={"random_cpu_fixture": True})
        prepared = backend.prepare(row)
        with torch.no_grad(), slider.scaled(0.):
            expected = waveform(model, row.positive, row.noise, steps=row.steps)
        assert torch.equal(prepared.target, expected) and not prepared.target.requires_grad
        assert torch.isfinite(prepared.neutral).all()
        backend.generated(prepared).square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in slider.parameters())
        assert any(b.lora_up.weight.grad.abs().sum() > 0 for b in slider.branches)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    finally:
        slider.close()


def test_successful_adapter_close_restores_mixed_borrowed_flags_and_modes(model):
    model.train(); model.fsq_layer.eval(); next(model.parameters()).requires_grad_(True)
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    slider = ParticleAdapter(model, rank=4, count=8)
    slider.close(); slider.close()
    assert flags == [p.requires_grad for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]


def test_adapter_init_primary_error_survives_cleanup_failure(model,monkeypatch):
    import voxcpm2_particle_sliders.adapter as adapter
    model.train();model.fsq_layer.eval();next(model.parameters()).requires_grad_(True)
    flags=[p.requires_grad for p in model.parameters()]
    modes=[m.training for m in model.modules()]
    targets=[m for n,m in model.named_modules() if n.endswith(("q_proj","k_proj","v_proj","o_proj"))]
    forwards=[m.forward for m in targets]
    primary=RuntimeError("primary conversion fixture failure")
    original_close=adapter.ParticleAdapter.close
    def fail_conversion(*a,**k):raise primary
    def cleanup_then_fail(self):
        original_close(self)
        raise RuntimeError("secondary cleanup fixture failure")
    monkeypatch.setattr(adapter.TinyParticleProjection,"to",fail_conversion)
    monkeypatch.setattr(adapter.ParticleAdapter,"close",cleanup_then_fail)
    with pytest.raises(RuntimeError,match="primary conversion") as caught:
        adapter.ParticleAdapter(model,rank=4,count=8)
    assert caught.value is primary
    assert any("secondary cleanup fixture failure" in note for note in primary.__notes__)
    assert forwards==[m.forward for m in targets]
    assert flags==[p.requires_grad for p in model.parameters()]
    assert modes==[m.training for m in model.modules()]


def test_control_changes_do_not_reclassify_same_example_as_heldout(tmp_path):
    reference=tmp_path/"ref.fixture";reference.write_bytes(b"numeric non-audio fixture")
    row=dict(text="Plain neutral words.",seed=171,neutral_reference="ref.fixture",positive_reference="ref.fixture")
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(dict(rows=[dict(row,id="train",split="train",positive_control="Drowsy delivery."),
        dict(row,id="dev",split="dev",positive_control="Very drowsy delivery.")])))
    with pytest.raises(ValueError,match="same calibration example"):read_manifest(path)


def test_native_row_refuses_parenthesized_neutral_control_before_prefix_build(monkeypatch):
    import voxcpm2_particle_sliders.runtime as runtime
    monkeypatch.setattr(runtime,"make_prefix",lambda *a,**k:pytest.fail("prefix built before refusal"))
    with pytest.raises(ValueError,match="neutral target text"):
        runtime.native_row(object(),dict(text=" (sleepy) Plain words.",positive_reference="unused",seed=1),frames=4,steps=10)


def test_official_vox_python_code_revision_is_verified_not_merely_declared(monkeypatch):
    import voxcpm2_particle_sliders.artifacts as artifacts
    identity = artifacts.verified_code_identity()
    assert identity["code_revision"] == "19b6bf7590025418821a86dcb817504e0ad7e5df"
    assert len(artifacts.VOXCPM_PYTHON_SOURCES) == 34
    original = artifacts.sha256
    monkeypatch.setattr(artifacts,"sha256",lambda p:"0"*64 if p.name == "voxcpm2.py" else original(p))
    with pytest.raises(ValueError,match="VoxCPM Python source differs"):
        artifacts.verified_code_identity()


@pytest.mark.parametrize("metadata", [None, {}, {"document":"not json"}, {"document":"[]"}, {"document":"{}"}])
def test_missing_or_malformed_native_metadata_fails_before_adapter_mutation(model,tmp_path,metadata):
    from safetensors.torch import save_file
    from voxcpm2_particle_sliders.artifacts import load_native
    slider = ParticleAdapter(model,rank=4,count=8)
    try:
        state={k:v.detach().clone() for k,v in slider.state_dict().items()}
        path=tmp_path/"metadata-fixture.safetensors";save_file(state,str(path),metadata=metadata)
        with pytest.raises(ValueError,match="metadata missing or malformed"):
            load_native(slider,path,identity={"fixture":True})
        assert all(torch.equal(v,slider.state_dict()[k]) for k,v in state.items())
    finally:slider.close()


@pytest.mark.parametrize("subtype,bits", [("PCM_16",16),("PCM_24",24),("PCM_32",32)])
def test_positive_pcm_rail_is_detected_without_generating_audio(tmp_path,monkeypatch,subtype,bits):
    import numpy as np
    from types import SimpleNamespace
    import voxcpm2_particle_sliders.audio_integrity as integrity
    path=tmp_path/"numeric.fixture";path.write_bytes(b"non-audio fixture")
    rail=1.-2.**(1-bits)
    monkeypatch.setattr(integrity.sf,"info",lambda p:SimpleNamespace(subtype=subtype))
    monkeypatch.setattr(integrity.sf,"read",lambda *a,**kw:(np.full((4000,1),rail),1000))
    result=integrity.inspect_audio(path)
    assert result["clipped_fraction"] == 1. and result["positive_clipping_rail"] == rail
    assert "clipped_samples_over_provisional_0.5_percent_screen" in result["review_flags"]


def test_condition_frame_coordinate_uses_native_frames_and_gradient_probe_ignores_empty_buffers(model):
    from voxcpm2_particle_sliders.backend import VoxCPM2WaveBackend,WaveRow
    from voxcpm2_particle_sliders.runtime import conditions
    slider=ParticleAdapter(model,rank=4,count=8)
    try:
        prefix=make_prefix(model,"plain text")
        backend=VoxCPM2WaveBackend(model,slider,model_identity={"random_fixture":True})
        rows=[backend.prepare(WaveRow(prefix,prefix,fixed_noise(model,n,3),steps=3)) for n in (2,3)]
        result=conditions(model,rows)
        assert torch.equal(result[:,-6],torch.tensor([2.,3.]))
        assert all(p.grad is None for p in slider.parameters())
        evidence=backend.gradient_evidence(rows[0])
        assert evidence["connected_finite"] and all(p.grad is None for p in slider.parameters())
    finally:slider.close()


def test_bf16_decoder_input_matches_actual_vae_dtype_and_keeps_gradients(model):
    model.audio_vae.decoder.to(dtype=torch.bfloat16)
    slider=ParticleAdapter(model,rank=4,count=8)
    try:
        with slider.scaled(1.):
            audio=waveform(model,make_prefix(model,"plain words"),fixed_noise(model,2,17),steps=3)
            assert audio.dtype == torch.float32 and torch.isfinite(audio).all()
            audio.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in slider.parameters())
        assert all(p.grad is None for p in model.parameters())
    finally:slider.close()


def test_unleased_native_gpu_paths_fail_before_any_cuda_allocation():
    from types import SimpleNamespace
    class UnloadedGPUFixture:
        def parameters(self):
            yield SimpleNamespace(device=torch.device("cuda:0"))
    fake=UnloadedGPUFixture()
    for call in (lambda:make_prefix(fake,"plain words"),lambda:fixed_noise(fake,2,17),
                 lambda:waveform(fake,None,None)):
        with pytest.raises(RuntimeError,match="active approved GPU1 lease"):call()
