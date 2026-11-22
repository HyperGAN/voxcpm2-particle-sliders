"""Schedule one gain per native autoregressive acoustic patch."""

from contextlib import ExitStack, contextmanager
from math import isfinite
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
def eos_patch_gate(native, *, strength, gate, direction, eligible_patches=None):
    """Gate the DiT-only RAW adapter on each eager native-EOS acoustic patch.

    The official cached loop passes the current fused LM conditioning as
    ``mu`` before DiT generation. Its projections are outside this adapter's
    hosts. Hold the gate decision for the full native DiT patch call.

    When eligible_patches is set, apply the gate to exactly that many initial
    patches and continue to native EOS at zero strength. Tail trace entries
    have no score or probability.
    """
    import torch
    from particle_sliders.adaptive_steering import ActivationGate, activation_projection

    if type(strength) not in (int, float) or not 0 < strength <= 1:
        raise ValueError("VoxCPM2 gated RAW strength must be in (0, 1]")
    if eligible_patches is not None and (type(eligible_patches) is not int or eligible_patches <= 0):
        raise ValueError("eligible_patches must be a positive integer")
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
        if eligible_patches is not None and len(trace) >= eligible_patches:
            scope.enter_context(native.network.scaled(0.0))
            trace.append(dict(score=None, probability=None, fired=False, strength=0.0))
            return
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
    if eligible_patches is not None and len(trace) < eligible_patches:
        raise RuntimeError("VoxCPM2 finished before emitting the eligible prefix")


@contextmanager
def eos_patch_spsr(native, *, steering, alpha):
    """Apply fitted S-PSR to each native EOS decoder conditioning row.

    The RAW adapter is disabled for this standalone intervention. The trace
    records the applied FP32 coefficient of the fitted direction per patch.
    """
    import torch
    from particle_sliders.prompt_steering import PromptSteering

    from .runtime import require_owner

    if type(alpha) not in (int, float) or not 0 <= alpha <= torch.finfo(torch.float32).max:
        raise ValueError("VoxCPM2 S-PSR alpha must be finite and nonnegative in FP32")
    if not isinstance(steering, PromptSteering):
        raise TypeError("VoxCPM2 S-PSR requires the fitted shared PromptSteering module")
    adapter = native.slider
    if (tuple(adapter.hosts) != ("dit",) or not adapter.names
            or any(not name.startswith("feat_decoder.estimator.") for name in adapter.names)):
        raise ValueError("native acoustic patch S-PSR requires a DiT-only RAW adapter")
    device = steering.probe.weight.device
    owner = require_owner(device)
    if owner is not None and owner is not native.owner:
        raise ValueError("the native EOS sampler must retain its original CUDA owner")
    if any(value.dtype != torch.float32 or value.device != device
           or not torch.isfinite(value).all() for value in steering.parameters()):
        raise ValueError("VoxCPM2 S-PSR requires finite, co-located FP32 parameters")
    width = steering.probe.in_features
    trace = []

    @torch.no_grad()
    def before_patch(_decoder, args, kwargs):
        mu = kwargs.get("mu")
        if (args or not isinstance(mu, torch.Tensor) or mu.shape != (1, width)
                or mu.dtype not in (torch.bfloat16, torch.float32) or mu.device != device):
            raise ValueError("native EOS requires one fused mu row matching the S-PSR fit")
        if require_owner(mu.device) is not owner:
            raise ValueError("the native EOS sampler must retain its original CUDA owner")
        with torch.autocast(device_type=device.type, enabled=False):
            edited, coefficient = steering(mu.unsqueeze(1), alpha)
            applied = float((coefficient * coefficient.new_tensor(alpha)).item())
        if not isfinite(applied) or not torch.isfinite(edited).all():
            raise ValueError("nonfinite native S-PSR edit")
        trace.append(applied)
        return args, {**kwargs, "mu": edited.squeeze(1)}

    with ExitStack() as scope:
        scope.enter_context(native.network.scaled(0.0))
        hook = native.model.feat_decoder.register_forward_pre_hook(before_patch, with_kwargs=True)
        scope.callback(hook.remove)
        yield trace
