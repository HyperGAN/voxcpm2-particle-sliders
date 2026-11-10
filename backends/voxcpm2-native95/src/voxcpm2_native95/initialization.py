"""Product-owned fresh policy and explicit restore context; no optimizer math."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from particlegan.init import R2Normal, Uniform, KEEP
from particle_sliders.initialization import (
    ComponentInitialization, InitializationReceipt, initialize_fresh_components,
)
from .contracts import CORE_REVISION, INITIALIZATION_POLICY, SOURCE_PROFILE, digest, file_hash


@dataclass(frozen=True)
class Construction:
    """Ephemeral intent passed BEFORE prepare/optimizers; never serialized as state."""
    mode: str
    cohort: str
    options_sha256: str
    initialization_json: str | None = None


def source_identity():
    return {"core_revision": CORE_REVISION, "product_source_profile": SOURCE_PROFILE,
            "initialization_policy": INITIALIZATION_POLICY,
            "integration_source_sha256": file_hash(__file__)}


def parameter_seed(seed, name):
    # Name isolation: insertion/reordering elsewhere cannot shift an existing draw.
    data = f"{INITIALIZATION_POLICY}:{seed}:{name}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % 2**63


def declared_affine(leaf, *, keep_weight=False):
    """Declare exact host laws through a holder of the SAME Parameters.

    The verified HNDL0.10 graph uses non-equalized Linear and ordinary Conv2d
    subclasses. Their constructors retain PyTorch Kaiming(a=sqrt5): weight and
    bias have Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)). No host class, forward,
    global initializer registry or Parameter object is replaced.
    """
    from hndl.operators.linear import Linear as HNDLLinear
    from hndl.operators.conv import Conv2d as HNDLConv2d
    if (type(leaf) not in (nn.Linear, nn.Conv1d, nn.Conv2d, HNDLLinear, HNDLConv2d)
            or any(leaf.children()) or list(leaf.buffers(recurse=False))
            or nn.utils.parametrize.is_parametrized(leaf) or "forward" in leaf.__dict__
            or set(leaf._parameters) != {"weight", "bias"}
            or type(leaf.weight) is not nn.Parameter
            or leaf.bias is not None and type(leaf.bias) is not nn.Parameter
            or type(leaf) is HNDLLinear and leaf.equalized is not False):
        raise ValueError("undeclared VoxCPM2 affine trainable leaf")
    if type(leaf) in (nn.Linear, HNDLLinear):
        expected = (leaf.out_features, leaf.in_features)
        fan_in = leaf.in_features
    else:
        expected = (leaf.out_channels, leaf.in_channels // leaf.groups, *leaf.kernel_size)
        fan_in = (leaf.in_channels // leaf.groups) * math.prod(leaf.kernel_size)
    if (fan_in < 1 or tuple(leaf.weight.shape) != expected
            or leaf.bias is not None and tuple(leaf.bias.shape) != (expected[0],)):
        raise ValueError("VoxCPM2 affine host geometry differs")
    holder, declarations = nn.Module(), {}
    bound = 1 / math.sqrt(fan_in)
    for name, parameter in leaf.named_parameters(recurse=False):
        holder.register_parameter(name, parameter)
        if parameter.requires_grad and parameter.numel():
            declarations[name] = KEEP if name == "weight" and keep_weight else Uniform(-bound, bound)
    return holder, declarations


def policies(adapter, critic, seed):
    if adapter.particles.ndim != 2 or adapter.particles.shape[1] != 4:
        raise ValueError("this declared VoxCPM2 policy requires its actual four-coordinate host bank")
    scopes = {}
    zero_up = {id(branch.lora_up) for branch in adapter.branches}
    for prefix, root in (("adapter", adapter), ("critic", critic)):
        for name, leaf in root.named_modules():
            if not any(p.requires_grad for p in leaf.parameters(recurse=False)):
                continue
            if leaf is adapter:  # The actual 128x4 particle Parameter is scoped below.
                if set(dict(leaf.named_parameters(recurse=False))) != {"particles"}:
                    raise ValueError("unexpected direct adapter parameters")
                continue
            key = prefix + "." + name
            holder, declarations = declared_affine(leaf, keep_weight=id(leaf) in zero_up)
            seeds = {n: parameter_seed(seed, key + "." + n)
                     for n, p in holder.named_parameters() if p.requires_grad and p.numel()}
            scopes[key] = ComponentInitialization(holder, "deterministic_orthogonal", seeds,
                                                  distributions=declarations)
    holder = nn.Module()
    holder.register_parameter("particles", adapter.particles)  # Same original object.
    scopes["adapter.bank"] = ComponentInitialization(holder, "deterministic_orthogonal",
        {"particles": parameter_seed(seed, "adapter.bank.particles")},
        distributions={"particles": R2Normal(0.0, 1.0)})
    declared = {id(p) for scope in scopes.values() for p in scope.module.parameters() if p.requires_grad}
    actual = {id(p) for root in (adapter, critic) for p in root.parameters() if p.requires_grad}
    if declared != actual:
        raise ValueError("VoxCPM2 initialization scopes must cover exactly owned trainables")
    return scopes


def validate_receipt(value, *, cohort):
    receipt = InitializationReceipt(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    import particle_sliders.initialization as shared
    shared_source = Path(shared.__file__)
    expected_source = {"file": shared_source.name, "bytes": shared_source.stat().st_size,
                       "sha256": file_hash(shared_source)}
    if (not isinstance(cohort, str) or not cohort
            or type(value.get("completed_steps")) is not int
            or value.get("schema") != "particle-sliders-scoped-fresh-initialization-v1"
            or value.get("cohort_id") != cohort or value.get("source_identity") != source_identity()
            or value.get("completed_steps") != 0 or value.get("restored_from") is not None
            or value.get("helper_source") != expected_source
            or value.get("particlegan", {}).get("version") != "0.9.5"
            or value.get("particlegan", {}).get("initializer_source", {}).get("sha256") !=
               "3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b"):
        raise ValueError("initialization receipt differs from this explicit fresh source cohort")
    return receipt


def initialize_owned(adapter, critic, construction, options):
    if (not isinstance(construction, Construction) or construction.mode not in {"fresh", "resume"}
            or construction.options_sha256 != digest(options)):
        raise ValueError("use open_training with explicit fresh/restore intent before optimizer construction")
    scopes = policies(adapter, critic, options.get("seed", 7))
    if construction.mode == "fresh":
        if construction.initialization_json is not None:
            raise ValueError("fresh construction cannot carry a restored initialization receipt")
        return initialize_fresh_components(scopes, cohort_id=construction.cohort,
            source_identity=source_identity(), completed_steps=0, restored_from=None, optimizers=())
    if construction.initialization_json is None:
        raise ValueError("resume requires its original initialization receipt")
    value = json.loads(construction.initialization_json)
    receipt = validate_receipt(value, cohort=construction.cohort)
    # Validate names/shapes/seeds/method before constructing optimizers. No fresh
    # initializer runs, no old packet is rewritten and no loaded state is reset.
    if set(value["components"]) != set(scopes):
        raise ValueError("saved initialization scope names differ")
    for name, scope in scopes.items():
        saved = value["components"][name]
        current = {n: p for n, p in scope.module.named_parameters() if p.requires_grad and p.numel()}
        if saved["method"] != scope.method or saved["gain"] != scope.gain or set(saved["parameters"]) != set(current):
            raise ValueError("saved initialization methods/parameters differ")
        for parameter_name, parameter in current.items():
            entry = saved["parameters"][parameter_name]
            declaration = scope.distributions[parameter_name]
            expected_law = ({"kind": "KEEP"} if declaration is KEEP else
                {"kind": "R2Normal", "mean": declaration.mean, "std": declaration.std}
                if isinstance(declaration, R2Normal) else
                {"kind": "Uniform", "low": declaration.low, "high": declaration.high})
            if (entry["seed"] != scope.parameter_seeds[parameter_name]
                    or entry["shape"] != list(parameter.shape) or entry["dtype"] != str(parameter.dtype)
                    or entry["host_declaration"] != expected_law):
                raise ValueError("saved initialization parameter identity differs")
    return receipt


def construction_for(config, packet):
    if config.backend != 'voxcpm2_native95':
        raise ValueError('historical VoxCPM2 cohorts require their original source; no packet migration')
    if packet is None:
        return Construction("fresh", config.cohort, digest(config.options))
    from particle_sliders.lifecycle import _check_packet
    from particle_sliders.response import TRAINING
    _check_packet(packet)
    if packet["schema"] != TRAINING or packet["manifest"]["config"] != config.as_dict():
        raise ValueError("resume requires the exact new-cohort training packet/config")
    game = packet["payload"]["game"]
    if game["config"].get("pg_version") != "0.9.5":
        raise ValueError("historical PG packets retain their original source; no migration")
    value = deepcopy(game["config"]["backend"]["fresh_initialization"])
    receipt = validate_receipt(value, cohort=config.cohort)
    return Construction("resume", config.cohort, digest(config.options), receipt.canonical_json)
