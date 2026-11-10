"""Native attention hosts around the shared routed projection implementation."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import math

import torch
from torch import nn
from particle_sliders.routed_projection import TinyParticleProjection


class ParticleAdapter(nn.Module):
    """Float32 routed leaves, one shared cloud, exact strength-zero bypass.

    Keep the scale context active through backward. This product deliberately
    does not enable activation checkpointing or compiled/mutable cache forwards.
    """

    def __init__(self, model, *, hosts=("dit",), rank=8, alpha=8., count=128, network_profile="voxcpm2-current0141-HNDL-singleton-waveform-v1", seed=None):
        super().__init__()
        hosts = tuple(hosts)
        if not hosts or len(set(hosts)) != len(hosts) or set(hosts) - {"dit", "lm"}:
            raise ValueError("hosts must select dit and/or lm exactly once")
        if type(rank) is not int or rank < 1 or type(count) is not int or count < 2:
            raise ValueError("positive rank and at least two particles required")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("finite positive alpha required")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("initialization seed must be a nonnegative integer below2**63")
        from .hndl_binding import PROFILE, verified_profile
        if network_profile!=PROFILE or (rank,alpha,count)!=(8,8.,128):
            raise ValueError("new exact HNDL rank8/alpha8/128x4 profile required")
        projection_class,builder,network_identity=verified_profile(network_profile)
        if any(getattr(m, "gradient_checkpointing", False) for m in model.modules()):
            raise ValueError("disable gradient checkpointing before attaching")
        roots = (["feat_decoder.estimator"] if "dit" in hosts else []) + (
            ["base_lm", "residual_lm"] if "lm" in hosts else [])
        plan = [(name, module) for name, module in model.named_modules()
                if any(name.startswith(root + ".") for root in roots)
                and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
                and isinstance(module, nn.Linear)]
        if not plan or any(hasattr(getattr(m.forward, "__self__", None), "_particles") for _, m in plan):
            raise ValueError("no unwrapped native attention targets")
        if any(hasattr(m, "lora_A") for m in model.modules()):
            raise ValueError("attach native particles to a base without ordinary LoRAs")
        previous_flags = [(p, p.requires_grad) for p in model.parameters()]
        previous_modes = [(m, m.training) for m in model.modules()]
        device = next(model.parameters()).device
        from .runtime import require_active_gpu1_lease
        require_active_gpu1_lease(device)
        self._borrowed_flags = previous_flags
        self._borrowed_modes = previous_modes
        model.requires_grad_(False).eval()
        self.names = tuple(name for name, _ in plan)
        self.hosts, self.rank, self.alpha = hosts, rank, float(alpha)
        self.network_profile, self.network_identity = network_profile, network_identity
        self.branches = nn.ModuleList()
        self._modules_owned = []
        try:
            devices = [device.index or 0] if device.type == "cuda" else []
            scope = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
            with scope:
                if seed is not None:
                    torch.default_generator.manual_seed(seed)
                    if devices: torch.cuda.default_generators[devices[0]].manual_seed(seed)
                self.particles = nn.Parameter(torch.randn(count, 4, device=device, dtype=torch.float32))
                for name, module in plan:
                    kwargs = dict(builder=builder, identity=network_identity) if builder is not None else {}
                    branch = projection_class(name, module, self.particles, rank, alpha, **kwargs)
                    # Track the just-attached forward BEFORE device conversion,
                    # so conversion/OOM failures cannot leave an orphaned hook.
                    self.branches.append(branch)
                    self._modules_owned.append((module, branch))
                    branch.to(device=device, dtype=torch.float32)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                primary.add_note("adapter cleanup also failed: " + type(cleanup).__name__ + ": " + str(cleanup))
            finally:
                for p, flag in previous_flags: p.requires_grad_(flag)
                for m, mode in previous_modes: m.training = mode
            raise

    @contextmanager
    def scaled(self, strength):
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("finite strength required")
        previous = [b.multiplier for b in self.branches]
        try:
            for branch in self.branches:
                branch.multiplier = strength
            yield self
        finally:
            for branch, value in zip(self.branches, previous):
                branch.multiplier = value

    def close(self):
        first_error = None
        for index in range(len(self._modules_owned) - 1, -1, -1):
            module, branch = self._modules_owned[index]
            if module.forward == branch.org_forward:
                del self._modules_owned[index]
                continue
            if module.forward != branch.forward:
                if first_error is None:
                    first_error = RuntimeError("attention forward ownership changed; refusing to overwrite")
                continue
            module.forward = branch.org_forward
            del self._modules_owned[index]
        if first_error is not None:
            raise first_error
        for parameter, flag in self._borrowed_flags:
            parameter.requires_grad_(flag)
        for module, mode in self._borrowed_modes:
            module.training = mode
        self._borrowed_flags = []
        self._borrowed_modes = []

    def metadata(self):
        result = dict(hosts=list(self.hosts), rank=self.rank, alpha=self.alpha,
                    particles=list(self.particles.shape), names=list(self.names),
                    architecture="shared TinyParticleProjection / RoutedMLP",
                    mergeable=False)
        if self.network_profile != "legacy":
            result.update(architecture="shared HNDLParticleProjection via verified HyperGAN builder",
                          network_profile=deepcopy(self.network_identity))
        return result
