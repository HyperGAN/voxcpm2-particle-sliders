"""Schedule one gain per native autoregressive acoustic patch."""

from contextlib import ExitStack, contextmanager
from sys import float_info


def resolve(strength, frames, frame_gains):
    if not isinstance(frame_gains, (list, tuple)) or len(frame_gains) != frames:
        raise ValueError("VoxCPM2 requires one gain per native acoustic patch")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in frame_gains):
        raise ValueError("VoxCPM2 frame gains must be finite and nonnegative")
    values = [strength * gain for gain in frame_gains]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("VoxCPM2 effective strengths must be in [0, 1]")
    return values


@contextmanager
def patch_schedule(native, strengths):
    applied = []
    scope = ExitStack()

    def before(_model, _args):
        scope.close()
        if len(applied) >= len(strengths):
            raise RuntimeError("VoxCPM2 generated more acoustic patches than scheduled")
        strength = strengths[len(applied)]
        scope.enter_context(native.network.scaled(strength))
        applied.append(strength)

    # The original full-prefix LM runs once per patch before the residual LM
    # and local DiT. These are the only supported adapter hosts.
    with ExitStack() as handles:
        handles.callback(native.model.base_lm.register_forward_pre_hook(before).remove)
        handles.callback(scope.close)
        yield applied
    if len(applied) != len(strengths):
        raise RuntimeError("VoxCPM2 finished before applying all scheduled strengths")
