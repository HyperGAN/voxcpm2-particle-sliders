"""Local-only model loading and bounded resource checks. Never downloads weights."""
from pathlib import Path
import json
import subprocess
import shutil
import os
import fcntl
import uuid as uuid_module
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from .artifacts import model_identity
from .adapter import ParticleAdapter
from .backend import VoxCPM2WaveBackend, WaveRow, fixed_noise, make_prefix

# Public snapshot placeholders deliberately authorize no real GPU. Binding actual
# UUIDs requires reviewed deployment configuration; never infer permission from
# visible/idle devices. Keep the cooperative lock and protected-GPU checks.
GPU1_UUID = "GPU-00000000-0000-0000-0000-000000000001"
GPU1_LOCK = Path("/tmp/anima-1.lock")
GPU0_UUID = "GPU-00000000-0000-0000-0000-000000000000"
GPU0_SCREENING_LOCK = Path("/tmp/voxcpm2-stock-screening-gpu0.lock")
_ACTIVE_LEASE = ContextVar("voxcpm2_gpu1_lease", default=None)


def canonical_gpu_uuid(value):
    """Normalize Torch string/UUID and CUuuid byte representations, fail closed."""
    try:
        if isinstance(value, str):
            text = value[4:] if value.startswith("GPU-") else value
            parsed = uuid_module.UUID(text)
        else:
            raw = value.bytes
            if isinstance(raw, (list, tuple)):
                if len(raw) != 16 or any(type(x) is not int or not 0 <= x <= 255 for x in raw):
                    raise ValueError("invalid UUID bytes")
                raw = bytes(raw)
            if not isinstance(raw, bytes) or len(raw) != 16:
                raise ValueError("invalid UUID bytes")
            parsed = uuid_module.UUID(bytes=raw)
        return "GPU-" + str(parsed)
    except (ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError("cannot resolve an exact CUDA device UUID") from exc


def _stock_gpu0_state():
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True,text=True,timeout=10)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits"], capture_output=True,text=True,timeout=10)
    if result.returncode or processes.returncode:
        raise RuntimeError("cannot verify shared GPU0 resources/ownership")
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 3 and fields[0] == GPU0_UUID:
            return dict(free_bytes=int(fields[1])*1024**2, utilization=int(fields[2]),
                processes=[line for line in processes.stdout.splitlines() if line.split(",")[0].strip()==GPU0_UUID])
    raise RuntimeError("approved physical GPU0 is not visible")


@contextmanager
def stock_screening_gpu0_lease(device, *, explicit_stock_sharing=False, probe, expected_models):
    """Narrow, measured stock-ASR/speaker sharing; grants no Vox/GAN device access."""
    if not explicit_stock_sharing:
        raise RuntimeError("GPU0 sharing requires explicit stock-screening authorization")
    if (probe.get("purpose") != "stock_screening_memory_probe" or probe.get("real_cuda_probe") is not True
            or probe.get("gpu_uuid") != GPU1_UUID or probe.get("models") != expected_models
            or probe.get("max_input_samples") != 409600 or probe.get("sample_rate") != 16000
            or probe.get("asr_max_tokens") != 128 or probe.get("batch_size") != 1
            or type(probe.get("peak_reserved_bytes")) is not int or not 0 < probe["peak_reserved_bytes"] <= 6*1024**3):
        raise ValueError("GPU0 screening requires matching measured model/shape memory evidence")
    if not device.startswith("cuda:") or not device[5:].isdigit() or not torch.cuda.is_available():
        raise RuntimeError("explicit visible CUDA device required")
    if canonical_gpu_uuid(torch.cuda.get_device_properties(int(device[5:])).uuid) != GPU0_UUID:
        raise RuntimeError("stock sharing is bound to approved physical GPU0")
    own_budget = probe["peak_reserved_bytes"] + 2*1024**3
    headroom = 16*1024**3
    with GPU0_SCREENING_LOCK.open("a+") as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("another stock screening coordinator owns GPU0 lease") from exc
        try:
            initial = _stock_gpu0_state()
            if initial["free_bytes"] < headroom+own_budget or initial["utilization"] > 60:
                raise RuntimeError("GPU0 sharing lacks conservative free headroom or compute capacity")
            def check():
                state = _stock_gpu0_state()
                if state["free_bytes"] < headroom or torch.cuda.memory_reserved(device) > own_budget:
                    raise RuntimeError("GPU0 sharing memory/headroom budget exhausted; release own screening model")
                return state
            # Deliberately do not set _ACTIVE_LEASE: existing Vox/GAN loaders and
            # samplers remain GPU1-only even inside this narrow screening context.
            yield check
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def resources(device="cuda:0"):
    report = dict(torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
                  requested_device=device, gpu_qualification=False)
    usage = shutil.disk_usage(Path.cwd())
    report["workspace_storage_bytes"] = dict(total=usage.total, used=usage.used, free=usage.free)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
        report["nvidia_smi_exit"] = result.returncode
        users = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        report["compute_processes"] = users.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    return report


def require_idle_gpu(device):
    """Fail closed when GPU ownership is unknown. No process is displaced."""
    if device == "cpu":
        return
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("use cpu or explicit logical cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; bounded real-model qualification cannot run")
    index = int(device[5:])
    props = torch.cuda.get_device_properties(index)
    # UUID resolves CUDA_VISIBLE_DEVICES rather than confusing physical ordinals.
    uuid = canonical_gpu_uuid(props.uuid)
    if uuid != GPU1_UUID:
        raise RuntimeError("CUDA work is reserved for physical GPU1; GPU0 studio ownership has not been released")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("cannot verify GPU process ownership")
    if any(line.split(",")[0].strip() == uuid and line.split(",")[1].strip() != str(os.getpid())
           for line in result.stdout.splitlines() if len(line.split(",")) >= 2):
        raise RuntimeError("selected GPU has compute work; choose an idle authorized GPU")
    free, _ = torch.cuda.mem_get_info(index)
    if free < 24 * 1024**3:
        raise RuntimeError("less than 24GiB free; 20GB official LoRA estimate is not a local measurement")


@contextmanager
def gpu1_lease(device):
    """Cooperate with the observed /tmp Anima physical-GPU1 lease protocol.

    Studio GPU0 remains reserved. A later actual release needs an explicit
    reviewed ownership change, not an assumption from momentary low utilization.
    """
    if device == "cpu":
        yield None
        return
    if not device.startswith("cuda:") or not device[5:].isdigit() or not torch.cuda.is_available():
        raise RuntimeError("CUDA execution requires visible explicit cuda:N and verified physical GPU1")
    uuid = canonical_gpu_uuid(torch.cuda.get_device_properties(int(device[5:])).uuid)
    if uuid != GPU1_UUID:
        raise RuntimeError("CUDA work is reserved for physical GPU1; GPU0 studio ownership has not been released")
    with GPU1_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another coordinator owns the physical GPU1 lease") from exc
        try:
            require_idle_gpu(device)
            handle.seek(0); handle.truncate(); handle.write(str(os.getpid())); handle.flush()
            token = _ACTIVE_LEASE.set((device, uuid))
            try:
                yield uuid
            finally:
                _ACTIVE_LEASE.reset(token)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def training_gpu1_lease(device):
    if device == "cpu":
        raise RuntimeError("training requires visible CUDA and verified physical GPU1")
    with gpu1_lease(device) as uuid:
        yield uuid


def require_active_gpu1_lease(device):
    if str(device) != "cpu" and _ACTIVE_LEASE.get() != (str(device), GPU1_UUID):
        raise RuntimeError("CUDA model execution requires an active approved GPU1 lease")


def load_local(folder, revision, *, device="cuda:0", hosts=("dit",), network_profile="legacy", adapter_seed=None):
    require_active_gpu1_lease(device)
    if network_profile != "legacy":
        from .singleton_binding import verified_profile
        verified_profile(network_profile)  # Reject graph source drift before model loading.
    require_idle_gpu(device)
    identity = model_identity(folder, revision)
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    model = VoxCPM2Model.from_local(str(folder), device=device, optimize=False, training=False)
    model.requires_grad_(False).eval()
    if model.audio_vae.sample_rate != 16000 or model.audio_vae.out_sample_rate != 48000:
        raise ValueError("expected VoxCPM2 encoder16k/decoder48k geometry")
    slider = ParticleAdapter(model, hosts=hosts, network_profile=network_profile, seed=adapter_seed)
    backend = VoxCPM2WaveBackend(model, slider, model_identity=identity)
    return model, slider, backend, identity


def read_manifest(path):
    path = Path(path)
    document = json.loads(path.read_text())
    rows = document["rows"]
    if (not rows or any(not isinstance(r["id"], str) or not r["id"] for r in rows)
            or len({r["id"] for r in rows}) != len(rows)):
        raise ValueError("nonempty uniquely identified rows required")
    for row in rows:
        if (row["split"] not in ("train", "dev") or not row["text"].strip()
                or type(row["seed"]) is not int or not 0 <= row["seed"] < 2**63):
            raise ValueError("explicit train/dev split and text required")
        if not row.get("positive_reference"):
            raise ValueError("positive character/style reference required")
        for key in ("positive_reference", "neutral_reference"):
            if row.get(key):
                value = Path(row[key])
                value = value if value.is_absolute() else path.parent / value
                if not value.is_file():
                    raise FileNotFoundError(value)
                row[key] = str(value.resolve())
        positive_teacher_text(row)  # Validate controlled teachers after path resolution.
    # IDs alone do not prevent renamed duplicate train/dev records.
    from .artifacts import sha256
    splits = {}
    for row in rows:
        # Excluding positive_control deliberately keeps this conservative:
        # changed delivery wording must not turn the same base example into dev.
        signature = (row["text"], row["seed"], sha256(row["positive_reference"]),
                     sha256(row["neutral_reference"]) if row.get("neutral_reference") else None)
        if signature in splits and splits[signature] != row["split"]:
            raise ValueError("same calibration example appears in train and dev")
        splits[signature] = row["split"]
    return document


def positive_teacher_text(record):
    """Explicit stock control teacher with one frozen speaker-conditioning asset.

    Legacy records keep their original positive-reference semantics. A controlled
    teacher changes only text conditioning; its audio is generated by prepare()
    at the same fixed noise/duration with every slider branch bypassed.
    """
    if record["text"].lstrip().startswith("("):
        raise ValueError("neutral target text must not begin with a parenthesized voice control")
    if "positive_control" not in record:
        return record["text"]
    control = record["positive_control"]
    if (not isinstance(control, str) or not control.strip()
            or any(c in control for c in "()\r\n")):
        raise ValueError("positive_control must be a nonempty single-line description without parentheses")
    neutral, positive = record.get("neutral_reference"), record.get("positive_reference")
    if not neutral or not positive:
        raise ValueError("controlled teacher requires the same frozen reference for neutral and positive")
    from .artifacts import sha256
    if sha256(neutral) != sha256(positive):
        raise ValueError("controlled teacher requires identical frozen reference bytes")
    return "(" + control.strip() + ")" + record["text"]


def native_row(model, record, *, frames, steps):
    positive_text = positive_teacher_text(record)
    return WaveRow(make_prefix(model, record["text"], reference_wav=record.get("neutral_reference")),
                   make_prefix(model, positive_text, reference_wav=record["positive_reference"]),
                   fixed_noise(model, frames, int(record["seed"])), steps=steps)


@torch.no_grad()
def conditions(model, prepared):
    result = []
    for p in prepared:
        prefix = p.row.neutral
        # Frozen model-derived text coordinates, full embedding width, no
        # learned row-number embedding. Exact full prefixes are checkpoint-bound.
        embedded = model.base_lm.embed_tokens(prefix.tokens).float()
        mask = prefix.text_mask[..., None]
        text = (embedded * mask).sum(1) / mask.sum(1).clamp_min(1)
        extra = text.new_tensor([[len(p.row.noise), p.row.steps, p.row.guidance,
                                  p.row.sway, float(p.row.cfg_zero_star), p.row.scale]])
        result.append(torch.cat((text, extra), dim=1))
    return torch.cat(result)
