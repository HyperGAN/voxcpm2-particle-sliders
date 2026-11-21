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


@contextmanager
def eos_patch_schedule(native, strengths):
    """Scope the official eager EOS loop, using the final strength until it stops."""
    if not strengths or any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("nonempty effective VoxCPM2 strengths in [0, 1] required")
    applied = []
    scope = ExitStack()

    def current():
        return strengths[min(len(applied), len(strengths) - 1)]

    def before_patch(_decoder, _args):
        applied.append(current())

    def after_stop_head(_head, _args, _output):
        scope.close()
        scope.enter_context(native.network.scaled(current()))

    # The cached native loop calls the stop head after emitting each patch.
    # Its LM feedback then prepares the next patch. Record only real DiT calls.
    model = native.model
    with ExitStack() as handles:
        handles.callback(model.feat_decoder.register_forward_pre_hook(before_patch).remove)
        handles.callback(model.stop_head.register_forward_hook(after_stop_head).remove)
        handles.callback(scope.close)
        scope.enter_context(native.network.scaled(current()))
        yield applied


@contextmanager
def eos_patch_gate(native, *, strength, gate, direction):
    """Gate the DiT-only RAW adapter on each eager native-EOS acoustic patch.

    The official cached loop passes the current fused LM conditioning as
    ``mu`` before DiT generation. Its projections are outside this adapter's
    hosts. Hold the gate decision for the full native DiT patch call.
    """
    import torch
    from particle_sliders.adaptive_steering import ActivationGate, activation_projection

    if type(strength) not in (int, float) or not 0 < strength <= 1:
        raise ValueError("VoxCPM2 gated RAW strength must be in (0, 1]")
    adapter = getattr(native, "slider", native.network)
    if tuple(adapter.hosts) != ("dit",):
        raise ValueError("native acoustic patch gating requires a DiT-only RAW adapter")
    if not isinstance(gate, ActivationGate) or gate.usage.eligible:
        raise ValueError("each native EOS request requires a fresh ActivationGate")
    if not isinstance(direction, torch.Tensor) or direction.ndim != 1 or not direction.is_floating_point():
        raise ValueError("supply a calibrated unit direction in native decoder mu coordinates")
    direction = direction.detach()
    length = direction.float().norm()
    if not torch.isclose(length, torch.ones_like(length)):
        raise ValueError("supply a finite unit direction in native decoder mu coordinates")

    trace = []
    scope = ExitStack()

    def before_patch(_decoder, _args, kwargs):
        mu = kwargs.get("mu")
        if not isinstance(mu, torch.Tensor) or mu.ndim != 2 or mu.shape[0] != 1:
            raise RuntimeError("native EOS patch must supply one fused LM conditioning row as mu")
        score = float(activation_projection(mu[0], direction, strength=strength))
        decision = gate.step(score)
        applied = strength if decision.fired else 0.0
        scope.enter_context(native.network.scaled(applied))
        trace.append(dict(score=score, probability=decision.probability,
                          fired=decision.fired, strength=applied))

    def after_patch(_decoder, _args, _output):
        scope.close()

    decoder = native.model.feat_decoder
    with ExitStack() as handles:
        handles.callback(decoder.register_forward_pre_hook(before_patch, with_kwargs=True).remove)
        handles.callback(decoder.register_forward_hook(after_patch, always_call=True).remove)
        handles.callback(scope.close)
        yield trace
