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
    from particle_sliders.adaptive_steering import ActivationGate
    from voxcpm2_native.adapter import ParticleAdapter
    from voxcpm2_native.frame_strength import eos_patch_gate

    class StopAtNativeMinimum(nn.Module):
        def forward(self, hidden):
            return hidden.new_tensor([[0, 1]])

    # Keep the actual native loop, KV caches, DiT and AudioVAE. Give the stop
    # head deterministic logits so the native minimum-length check admits EOS.
    model.stop_head = StopAtNativeMinimum()
    network = ParticleAdapter(model, seed=7)
    native = SimpleNamespace(model=model, network=network)
    with torch.no_grad():
        for branch in network.branches:
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
        with network.scaled(0):
            off = render()
        with network.scaled(strength):
            used = ActivationGate("random", 0, seed=1)
            with eos_patch_gate(native, strength=strength, gate=used, direction=direction) as off_trace:
                gated_off = render()
            assert all(branch.multiplier == strength for branch in network.branches)
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
        assert len(model.feat_decoder._forward_pre_hooks) == len(model.feat_decoder._forward_hooks) == 0
    finally:
        network.close()


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
    native.network.hosts = ("dit", "lm")
    with pytest.raises(ValueError, match="DiT-only RAW adapter"):
        with eos_patch_gate(native, strength=0.75, gate=ActivationGate("cfp", 2), direction=torch.tensor([1.0])):
            pytest.fail("accepted an LM adapter")
