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
