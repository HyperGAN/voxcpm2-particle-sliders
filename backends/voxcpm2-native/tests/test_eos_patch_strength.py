from contextlib import contextmanager
from types import SimpleNamespace

from torch import nn

from voxcpm2_native.frame_strength import eos_patch_schedule


def test_official_eos_scope_covers_prefill_and_feedback_and_records_only_real_patches():
    model = SimpleNamespace(feat_decoder=nn.Identity(), stop_head=nn.Identity())
    active = 0.0

    @contextmanager
    def scaled(value):
        nonlocal active
        previous, active = active, value
        try:
            yield
        finally:
            active = previous

    native = SimpleNamespace(model=model, network=SimpleNamespace(scaled=scaled))
    with eos_patch_schedule(native, [0.75, 0.25, 0.5]) as applied:
        assert active == 0.75  # Native prefill prepares the first patch.
        for value in (0.75, 0.25, 0.5, 0.5):
            assert active == value
            model.feat_decoder(None)
            model.stop_head(None)
        assert applied == [0.75, 0.25, 0.5, 0.5]
    assert active == 0.0

    with eos_patch_schedule(native, [0.25, 0.75]) as applied:
        model.feat_decoder(None)
        model.stop_head(None)
        assert active == 0.75  # Native termination can skip the next feedback.
    assert applied == [0.25] and active == 0.0


def test_native_eos_gate_matches_off_and_replays_strengths(model):
    import torch
    from particle_sliders.backends import AdapterParameters
    from particle_sliders.adaptive_steering import ActivationGate
    from voxcpm2_native.adapter import ParticleAdapter
    from voxcpm2_native.frame_strength import eos_patch_gate
    from voxcpm2_native.waveform import VoxCPM2WaveBackend

    class StopAtNativeMinimum(nn.Module):
        def forward(self, hidden):
            return hidden.new_tensor([[0, 1]])

    # Keep the actual native loop, KV caches, DiT and AudioVAE. Give the stop
    # head deterministic logits so the native minimum-length check admits EOS.
    model.stop_head = StopAtNativeMinimum()
    slider = ParticleAdapter(model, seed=7)
    native = VoxCPM2WaveBackend(model, slider, model_identity={"fixture": "tiny official VoxCPM2"})
    assert isinstance(native.network, AdapterParameters)
    assert slider.names and all(name.startswith("feat_decoder.estimator.") for name in slider.names)
    with torch.no_grad():
        for branch in slider.branches:
            branch.lora_up.weight.normal_(std=0.05)

    width = model.lm_to_dit_proj.out_features + model.res_to_dit_proj.out_features
    direction = torch.eye(width)[0]
    strength = 0.75
    samples_per_patch = model.patch_size * model._decode_chunk_size

    def render():
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(219)
            return model.generate(target_text="test test", min_len=1, max_len=7,
                                  inference_timesteps=2, retry_badcase=False)

    try:
        with native.network.scaled(0):
            off = render()
        with native.network.scaled(strength):
            used = ActivationGate("random", 0, seed=1)
            with eos_patch_gate(native, strength=strength, gate=used, direction=direction) as off_trace:
                gated_off = render()
            assert all(branch.multiplier == strength for branch in slider.branches)
        assert torch.equal(off, gated_off)
        assert off.shape == (1, 3 * samples_per_patch)
        assert [entry["strength"] for entry in off_trace] == [0, 0, 0]
        assert used.usage.eligible == len(off_trace) == 3

        # Threshold equality is a causal CFP decision on the identical first
        # patch. The remaining scores come from the new steered trajectory.
        cfp = ActivationGate("cfp", off_trace[0]["score"])
        with eos_patch_gate(native, strength=strength, gate=cfp, direction=direction) as trace:
            adaptive = render()
        assert [item["fired"] for item in trace] == [item["score"] <= off_trace[0]["score"] for item in trace]
        assert trace[0]["fired"] and cfp.usage.eligible == len(trace) == 3
        with eos_patch_schedule(native, [item["strength"] for item in trace]) as replayed:
            fixed_replay = render()
        assert len(replayed) == 3
        assert torch.equal(adaptive, fixed_replay)
        assert not torch.equal(adaptive, off)

        prefix = ActivationGate("cfp", off_trace[0]["score"])
        with native.network.scaled(strength):
            with eos_patch_gate(native, strength=strength, gate=prefix,
                                direction=direction, eligible_patches=2) as prefix_trace:
                prefix_audio = render()
            assert all(branch.multiplier == strength for branch in slider.branches)
        assert prefix.usage.eligible == 2 and prefix_trace[:2] == trace[:2]
        assert prefix_trace[2] == dict(score=None, probability=None, fired=False, strength=0.0)
        assert prefix_audio.shape == off.shape
        with eos_patch_schedule(native, [entry["strength"] for entry in prefix_trace]):
            assert torch.equal(render(), prefix_audio)
        assert len(model.feat_decoder._forward_pre_hooks) == len(model.feat_decoder._forward_hooks) == 0
    finally:
        slider.close()


def test_native_eos_gate_rejects_mixed_dit_and_lm_hosts(model):
    import pytest
    import torch
    from particle_sliders.adaptive_steering import ActivationGate
    from voxcpm2_native.adapter import ParticleAdapter
    from voxcpm2_native.frame_strength import eos_patch_gate
    from voxcpm2_native.waveform import VoxCPM2WaveBackend

    slider = ParticleAdapter(model, hosts=("dit", "lm"), seed=7)
    try:
        native = VoxCPM2WaveBackend(model, slider, model_identity={"fixture": "tiny official VoxCPM2"})
        assert any(name.startswith("feat_decoder.estimator.") for name in slider.names)
        assert any(name.startswith("base_lm.") for name in slider.names)
        gate = ActivationGate("cfp", 2)
        with pytest.raises(ValueError, match="DiT-only RAW adapter"):
            with eos_patch_gate(native, strength=0.75, gate=gate, direction=torch.tensor([1.0])):
                pytest.fail("accepted mixed DiT and LM hosts")
        assert gate.usage.eligible == 0
        assert len(model.feat_decoder._forward_pre_hooks) == len(model.feat_decoder._forward_hooks) == 0
    finally:
        slider.close()


def test_gate_restores_strength_and_hooks_if_patch_fails():
    import pytest
    import torch
    from particle_sliders.adaptive_steering import ActivationGate
    from voxcpm2_native.frame_strength import eos_patch_gate

    active = 0.25

    @contextmanager
    def scaled(value):
        nonlocal active
        previous, active = active, value
        try:
            yield
        finally:
            active = previous

    class Decoder(nn.Module):
        def forward(self, *, mu):
            assert active == 0.75
            raise RuntimeError("test decoder failure")

    native = SimpleNamespace(model=SimpleNamespace(feat_decoder=Decoder()),
                             network=SimpleNamespace(hosts=("dit",), scaled=scaled))
    gate = ActivationGate("cfp", 2)
    with pytest.raises(RuntimeError, match="test decoder failure"):
        with eos_patch_gate(native, strength=0.75, gate=gate, direction=torch.tensor([1.0])):
            native.model.feat_decoder(mu=torch.ones(1, 1))
    assert gate.usage.eligible == 1
    assert active == 0.25
    assert len(native.model.feat_decoder._forward_pre_hooks) == 0
    assert len(native.model.feat_decoder._forward_hooks) == 0
    with pytest.raises(ValueError, match="fresh ActivationGate"):
        with eos_patch_gate(native, strength=0.75, gate=gate, direction=torch.tensor([1.0])):
            pytest.fail("reused gate")


def test_eos_gate_refuses_missing_eligible_prefix():
    from contextlib import nullcontext

    import pytest
    import torch
    from particle_sliders.adaptive_steering import ActivationGate
    from voxcpm2_native.frame_strength import eos_patch_gate

    decoder = nn.Identity()
    adapter = SimpleNamespace(hosts=("dit",), scaled=lambda _value: nullcontext())
    native = SimpleNamespace(model=SimpleNamespace(feat_decoder=decoder), network=adapter)
    gate = ActivationGate("cfp", 2)
    with pytest.raises(ValueError, match="positive integer"):
        with eos_patch_gate(native, strength=0.75, gate=gate, direction=torch.tensor([1.0]),
                            eligible_patches=0):
            pytest.fail("accepted an empty prefix")
    with pytest.raises(RuntimeError, match="before emitting the eligible prefix"):
        with eos_patch_gate(native, strength=0.75, gate=gate, direction=torch.tensor([1.0]),
                            eligible_patches=2):
            pass
    assert gate.usage.eligible == 0
    assert len(decoder._forward_pre_hooks) == len(decoder._forward_hooks) == 0


def test_native_eos_spsr_matches_the_shared_edit_on_each_emitted_patch(model):
    import torch
    from particle_sliders.prompt_steering import PromptSteering
    from voxcpm2_native.adapter import ParticleAdapter
    from voxcpm2_native.frame_strength import eos_patch_spsr
    from voxcpm2_native.waveform import VoxCPM2WaveBackend

    class StopAtNativeMinimum(nn.Module):
        def forward(self, hidden):
            return hidden.new_tensor([[0, 1]])

    model.stop_head = StopAtNativeMinimum()
    slider = ParticleAdapter(model, seed=7)
    native = VoxCPM2WaveBackend(model, slider, model_identity={"fixture": "tiny official VoxCPM2"})
    width = model.lm_to_dit_proj.out_features + model.res_to_dit_proj.out_features
    steering = PromptSteering(width).eval().requires_grad_(False)
    with torch.no_grad():
        steering.probe.bias.fill_(2.0)
        for branch in slider.branches:
            branch.lora_up.weight.normal_(std=0.05)

    def render():
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(219)
            return model.generate(target_text="test test", min_len=1, max_len=7,
                                  inference_timesteps=2, retry_badcase=False)

    try:
        with native.network.scaled(0.0):
            off = render()
        replayed = []

        def shared_edit(_decoder, args, kwargs):
            edited, coefficient = steering(kwargs["mu"].unsqueeze(1), 1.0)
            replayed.append(coefficient.item())
            return args, {**kwargs, "mu": edited.squeeze(1)}

        with native.network.scaled(0.0):
            hook = model.feat_decoder.register_forward_pre_hook(shared_edit, with_kwargs=True)
            try:
                expected = render()
            finally:
                hook.remove()

        with native.network.scaled(0.75):
            with eos_patch_spsr(native, steering=steering, alpha=0.0) as zeros:
                assert torch.equal(render(), off)
            assert zeros == [0.0, 0.0, 0.0]
            with eos_patch_spsr(native, steering=steering, alpha=1.0) as trace:
                actual = render()
            assert native.network.scale == 0.75
            assert all(branch.multiplier == 0.75 for branch in slider.branches)
        samples_per_patch = model.patch_size * model._decode_chunk_size
        assert actual.shape == (1, 3 * samples_per_patch)
        assert torch.equal(actual, expected) and not torch.equal(actual, off)
        assert trace == replayed and len(trace) == 3 and len(set(trace)) > 1

        with native.network.scaled(0.75):
            with eos_patch_spsr(native, steering=steering, alpha=1.0, eligible_patches=2) as prefix_trace:
                prefix_audio = render()
            assert native.network.scale == 0.75
        assert prefix_trace == trace[:2] + [0.0]
        assert prefix_audio.shape == actual.shape
        assert not torch.equal(prefix_audio, actual)
        assert len(model.feat_decoder._forward_pre_hooks) == 0
    finally:
        slider.close()


def test_spsr_uses_fp32_dose_and_restores_scope_on_decoder_failure():
    import pytest
    import torch
    from particle_sliders.prompt_steering import PromptSteering
    from voxcpm2_native.frame_strength import eos_patch_spsr

    active = 0.25

    @contextmanager
    def scaled(value):
        nonlocal active
        previous, active = active, value
        try:
            yield
        finally:
            active = previous

    class Decoder(nn.Module):
        def forward(self, *, mu):
            assert active == 0.0
            assert mu.dtype == torch.bfloat16
            assert mu.item() == torch.tensor(2.1, dtype=torch.bfloat16).item()
            raise RuntimeError("test decoder failure")

    decoder = Decoder()
    native = SimpleNamespace(model=SimpleNamespace(feat_decoder=decoder),
                             slider=SimpleNamespace(hosts=("dit",), names=("feat_decoder.estimator.q_proj",)),
                             network=SimpleNamespace(scaled=scaled))
    steering = PromptSteering(1).eval().requires_grad_(False)
    with torch.no_grad():
        steering.probe.weight.zero_()
        steering.probe.bias.fill_(1.1)
        steering.direction.weight.fill_(10.0)

    with pytest.raises(RuntimeError, match="test decoder failure"):
        with eos_patch_spsr(native, steering=steering, alpha=0.1) as trace:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                decoder(mu=torch.ones((1, 1), dtype=torch.bfloat16))
    assert trace == [(torch.tensor(1.1) * torch.tensor(0.1)).item()]
    assert active == 0.25 and len(decoder._forward_pre_hooks) == 0

    with pytest.raises(ValueError, match="one fused mu row"):
        with eos_patch_spsr(native, steering=steering, alpha=1.0):
            decoder(mu=torch.ones((1, 2), dtype=torch.bfloat16))
    assert active == 0.25 and len(decoder._forward_pre_hooks) == 0

    with pytest.raises(ValueError, match="eligible_patches must be a positive integer"):
        with eos_patch_spsr(native, steering=steering, alpha=1.0, eligible_patches=True):
            pytest.fail("accepted an invalid prefix")
    with pytest.raises(RuntimeError, match="finished before emitting the eligible prefix"):
        with eos_patch_spsr(native, steering=steering, alpha=1.0, eligible_patches=2):
            pass
    assert active == 0.25 and len(decoder._forward_pre_hooks) == 0

    native.slider.hosts = ("dit", "lm")
    with pytest.raises(ValueError, match="DiT-only RAW adapter"):
        with eos_patch_spsr(native, steering=steering, alpha=1.0):
            pytest.fail("accepted mixed DiT and LM hosts")
