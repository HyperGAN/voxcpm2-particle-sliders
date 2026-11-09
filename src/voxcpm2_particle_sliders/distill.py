"""Existing Anima/YuE routed-down ridge fit, retained up, stock VoxCPM keys."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import hashlib

import torch
from safetensors.torch import save_file
from particle_sliders import fit_routed_down
from .artifacts import sha256


def calibration_profile(*, frames, steps, cfg_value, cfg_zero_star, sway=1.):
    """Fixed native sampler coordinates used for both calibration and held-out capture."""
    if (type(frames) is not int or not 1 <= frames <= 64 or type(steps) is not int
            or not 2 <= steps <= 32 or type(cfg_zero_star) is not bool
            or not math.isfinite(cfg_value) or cfg_value < 0
            or not math.isfinite(sway) or not 0 <= sway <= 1):
        raise ValueError("invalid fixed native calibration profile")
    return dict(protocol="fixed-noise-full-prefix-waveform-v1", frames=frames, steps=steps,
                cfg_value=float(cfg_value), cfg_zero_star=cfg_zero_star, sway=float(sway),
                native_adapter_strength=1., cfg_adapter_branch="both")


def _require_unit_capture(slider):
    for branch in slider.branches:
        if branch.multiplier != 1.:
            raise ValueError("capture native teacher at runtime strength 1")
        if branch.scale != slider.alpha / slider.rank:
            raise ValueError("shared branch.scale must remain fixed native alpha/r")


@contextmanager
def capture(slider, model, *, max_rows=256):
    """Bounded full projection input -> routed pre-up target, no gradients."""
    if type(max_rows) is not int or not 2 <= max_rows <= 2048:
        raise ValueError("calibration budget must be 2..2048 rows per projection")
    _require_unit_capture(slider)
    buckets = {name: [[], []] for name in slider.names}
    modules = dict(model.named_modules())
    handles = []
    pending = {}
    def hooks_for(name, branch):
        def before_host(module, args):
            if branch.multiplier != 1. or branch.scale != slider.alpha / slider.rank:
                raise ValueError("capture requires strength 1 and fixed alpha/r")
            xs, ys = buckets[name]
            remaining = max_rows - sum(len(x) for x in xs)
            if remaining <= 0:
                pending[name] = None
                return
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            count = min(remaining, len(x), 16)
            ids = torch.linspace(0, len(x) - 1, count, device=x.device).long()
            pending[name] = (x[ids].cpu(), ids, len(x))
        def before_up(module, args):
            value = pending.get(name)
            if value is None:
                return
            x, ids, count = value
            # Public lora_up property + standard module hook observes the actual
            # pre-up activation in either shared Torch or shared HNDL execution.
            # No routing is reimplemented and no private graph endpoint is read.
            y = args[0].detach().float().reshape(-1, slider.rank)
            if len(y) != count:
                raise ValueError("shared projection input/pre-up leading axes differ")
            buckets[name][0].append(x); buckets[name][1].append(y[ids.to(y.device)].cpu())
            pending[name] = None
        def after_host(module, args, output):
            if pending.get(name) is not None:
                raise RuntimeError("shared projection did not expose its actual public lora_up pre-hook; no private fallback")
        return before_host, before_up, after_host
    try:
        for name, branch in zip(slider.names, slider.branches):
            before_host, before_up, after_host = hooks_for(name, branch)
            handles.append(modules[name].register_forward_pre_hook(before_host))
            handles.append(branch.lora_up.register_forward_pre_hook(before_up))
            handles.append(modules[name].register_forward_hook(after_host))
        yield buckets
    finally:
        for handle in reversed(handles):
            handle.remove()


def matrices(buckets):
    result = {}
    for name, (xs, ys) in buckets.items():
        if not xs or sum(len(x) for x in xs) < 2:
            raise ValueError("insufficient calibration rows for " + name)
        result[name] = (torch.cat(xs), torch.cat(ys))
    return result


def _regression_metrics(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("matching held-out activation matrices required")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("finite held-out activations required")
    sse = float((predicted - target).square().sum())
    energy = float(target.square().sum())
    tss = float((target - target.mean(0, keepdim=True)).square().sum())
    return dict(squared_error=sse, teacher_squared_norm=energy, centered_tss=tss,
                elements=target.numel(), mse=sse / target.numel(),
                relative_mse=sse / energy if energy > 0 else None,
                r2=1 - sse / tss if tss > 0 else None)


def _aggregate(reports):
    result = {key: sum(r[key] for r in reports)
              for key in ("squared_error", "teacher_squared_norm", "centered_tss", "elements")}
    sse, norm, tss = (result[k] for k in ("squared_error", "teacher_squared_norm", "centered_tss"))
    return dict(**result, mse=sse / result["elements"],
                relative_mse=sse / norm if norm > 0 else None, r2=1 - sse / tss if tss > 0 else None)


def _disjoint_heldout(x, xv, yv):
    """Exclude exact reused activation inputs and duplicate held-out observations.

    Native CFG and common text prefixes may otherwise repeat deterministic
    activations even across distinct examples. This is stricter than row IDs.
    """
    def key(row):
        return hashlib.sha256(row.detach().float().cpu().contiguous().numpy().tobytes()).digest()
    seen = {key(row) for row in x}
    keep = []
    for index, row in enumerate(xv):
        signature = key(row)
        if signature not in seen:
            keep.append(index)
            seen.add(signature)
    if len(keep) < 2:
        raise ValueError("at least two disjoint held-out activation inputs required per projection")
    return xv[keep], yv[keep]


@torch.no_grad()
def export_distilled(slider, train, heldout, output, *, teacher, model_identity,
                     train_ids, heldout_ids, data_provenance, calibration):
    """Approximate native nonlinear branches; no claim of exact mergeability."""
    if (not train_ids or not heldout_ids or set(train_ids) & set(heldout_ids)
            or len(set(train_ids)) != len(train_ids) or len(set(heldout_ids)) != len(heldout_ids)):
        raise ValueError("nonempty distinct calibration/held-out row identities required")
    if set(train) != set(slider.names) or set(heldout) != set(slider.names):
        raise ValueError("capture projection names differ")
    expected_calibration = calibration_profile(**{k: calibration[k]
        for k in ("frames", "steps", "cfg_value", "cfg_zero_star", "sway")})
    if calibration != expected_calibration:
        raise ValueError("calibration coordinates differ from the declared native protocol")
    if any(branch.scale != slider.alpha / slider.rank for branch in slider.branches):
        raise ValueError("shared branch.scale must remain fixed native alpha/r")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tensors, reports = {}, []
    for name, branch in zip(slider.names, slider.branches):
        x, y = train[name]
        xv, yv = heldout[name]
        heldout_before = len(xv)
        xv, yv = _disjoint_heldout(x, xv, yv)
        down, ridge = fit_routed_down(x.float(), y.float(), ridge_fraction=.01)
        # Stock scaling alpha/r=1; retain native alpha/r in the up matrix.
        up = branch.lora_up.weight.detach().float().cpu() * branch.scale
        tensors[name + ".lora_A"] = down.cpu().contiguous()
        tensors[name + ".lora_B"] = up.contiguous()
        prediction = xv.float() @ down.T
        bottleneck = _regression_metrics(prediction, yv.float())
        projection = _regression_metrics(prediction @ up.T, yv.float() @ up.T)
        reports.append(dict(name=name, rows=len(x), heldout_rows=len(xv), ridge=float(ridge),
                            heldout_excluded_duplicate_inputs=heldout_before-len(xv),
                            heldout_bottleneck=bottleneck, heldout_projection=projection))
    config = dict(enable_lm="lm" in slider.hosts, enable_dit="dit" in slider.hosts,
                  enable_proj=False, r=slider.rank, alpha=slider.rank, dropout=0.,
                  target_modules_lm=["q_proj", "k_proj", "v_proj", "o_proj"],
                  target_modules_dit=["q_proj", "k_proj", "v_proj", "o_proj"])
    document = dict(schema="voxcpm2-distilled-lora-v1", approximate=True,
                    method="shared fit_routed_down dual-ridge .01; retained native scaled up",
                    teacher_sha256=sha256(teacher), model_identity=model_identity,
                    native_adapter=slider.metadata(), network_profile=slider.network_profile,
                    train_ids=list(train_ids), heldout_ids=list(heldout_ids),
                    calibration=calibration,
                    data_provenance=data_provenance, lora_config=config, projections=reports,
                    heldout_bottleneck=_aggregate([r["heldout_bottleneck"] for r in reports]),
                    heldout_projection=_aggregate([r["heldout_projection"] for r in reports]),
                    waveform_fidelity_qualified=False, intelligibility_qualified=False,
                    identity_qualified=False, expressivity_qualified=False, strength_qualified=False)
    save_file(tensors, str(output / "lora_weights.safetensors"),
              metadata={"document": json.dumps(document, sort_keys=True, allow_nan=False)})
    (output / "lora_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "distillation.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document
