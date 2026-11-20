"""Native weights with strict model/projection identity; no frozen base export."""
from pathlib import Path
import hashlib
import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


VOXCPM_CODE_REVISION = '19b6bf7590025418821a86dcb817504e0ad7e5df'
VOXCPM_PYTHON_SOURCES = {
    "__init__.py": "a6a971983ce2777468802b1873fdf455220e49b68c445c503c8cdc677dafa89b",
    "cli.py": "856aabaa3e7c170e1a7d54295e4de1ce9263a987df9c690da0748bd709b8d7f1",
    "core.py": "aa8965af33ebbbc611b2bad18c67b99cfffac7b7b87900f49efa8d454f2755d2",
    "model/__init__.py": "6e3dc8864d5b42bca453a6dedc30463e36b7114ec2e08252391029bd27c491c1",
    "model/utils.py": "cea16e1ab57f15129a7f5dec13c428bd14a771221abcf17dd0a90b3d65e763a2",
    "model/voxcpm.py": "486d1543381e2e312f43b00f0b9517427073e8abefdfec871df461fe9905340f",
    "model/voxcpm2.py": "9c4f1e08495e727ce339e5da93347237b81f386398a783b79c66eb206c3397f6",
    "modules/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "modules/audiovae/__init__.py": "bf72e27063eac02f332ae0489696ecca8c0ceb7a0a41f0af54ca8f1c6f145abf",
    "modules/audiovae/audio_vae.py": "cabe57717c94437ace644b8da7fa4db811a5bdacb437e341a875f890bf8308f9",
    "modules/audiovae/audio_vae_v2.py": "4a8c6f30ad3aa1e18c405cc103de617c125b21894d84d98a377930ca0ca7c429",
    "modules/layers/__init__.py": "4578ed00894dce44a30e2cb153f9ade69382391a0530d718a52634a1beb43c1a",
    "modules/layers/lora.py": "8071035d19057bee86f61c3261c63193b234e49312b122eb635074984375b534",
    "modules/layers/scalar_quantization_layer.py": "3e8c661a0eae8ecbf5abbebbea8971a962e95ebef8ea887b5ca4796d9436ca9d",
    "modules/locdit/__init__.py": "a22d80973102835a5edcca027e59cefc611a6d2afec1af98690727bcfc4b3685",
    "modules/locdit/local_dit.py": "d6e82cee3f5157724b1c998325163c96fe1fff5f11a733c89faacafe9987223b",
    "modules/locdit/local_dit_v2.py": "7ea51eb5acc8475ffa41649852ee8d5d3f2a601b962f313f9e1c7eb109542b2f",
    "modules/locdit/unified_cfm.py": "e4535337024cf44b6888eccfd39ccbe98684e302dd1836e05f940b4d626dcfdf",
    "modules/locenc/__init__.py": "257ee6d2ee516f7065573180c7c2b175b06e7c59b8b89998b34b6ffafd7bb332",
    "modules/locenc/local_encoder.py": "ab1d3c5b4028573495e2bc57d548af0310a577cee04c57055aaafcf94a1e1729",
    "modules/minicpm4/__init__.py": "c2d6c1a428844a709d88e0312f07a79f337b4cfd58334756013786510e399f11",
    "modules/minicpm4/cache.py": "644acd3bf7cf8313f2ba285f68c41967fcedb57e5f5d8f38a883acc385ee030d",
    "modules/minicpm4/config.py": "5bed3ea87242f0ad5d0e8d80d602eea4373c9d175e26f5bb324b3c93ad7db36d",
    "modules/minicpm4/model.py": "0733349295d13b2ec57e979058fa64becec821b9bd475e188eb4033fc016d2d2",
    "training/__init__.py": "51dd16f63384a5db3c5cfd2b9269f63c4607e2920e85e51fc97224496338fee2",
    "training/accelerator.py": "11ae8402082431d0fff18847396a277f3c6159caa12bfaf3606c90df7d836d3a",
    "training/config.py": "2e0f8fb3668f787e67485f556dd58955bd03d083e39f0569aa2719a557ed53fc",
    "training/data.py": "ecf7c3ecd1f12ca7953bb81c7fa44a9136cec21de088a92c3b8f3f398bfc83f1",
    "training/packers.py": "cf8b80c2ac0896056f2a38bbb3ac8afc95b49f80a1d4e58b5edd159f89477ef7",
    "training/state.py": "f4fc2fca40cbcbbcb50ecbc1d7ade51ec1c0a7d8752c2fccdb80817301f552b1",
    "training/tracker.py": "28001dbdd830cdb5f419f24e46f0e05edbbf60ed3180bfd6025a6977def9ada5",
    "training/validate.py": "a107f6669c24fc7137408e8d991fa73c360e66b087419744bba0d99d5529f15c",
    "utils/text_normalize.py": "3e2c86c777ed67fda8c9b001a93dba7239bf79a06503e2bcd46afc5ef0dffe3b",
    "zipenhancer.py": "48e591c4117cc7eea2e3fd54363d2bcf0f306b70e994320eacf0e025636bae27"
}

def verified_code_identity():
    """Drift check of pinned official Python sources, not a security sandbox."""
    import voxcpm
    root = Path(voxcpm.__file__).parent
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    if actual != set(VOXCPM_PYTHON_SOURCES):
        raise ValueError("installed VoxCPM Python source inventory differs")
    for name, digest in VOXCPM_PYTHON_SOURCES.items():
        if sha256(root / name) != digest:
            raise ValueError("installed VoxCPM Python source differs: " + name)
    canonical = json.dumps(VOXCPM_PYTHON_SOURCES, sort_keys=True, separators=(",", ":")).encode()
    return dict(code_revision=VOXCPM_CODE_REVISION, code_python_sources_sha256=hashlib.sha256(canonical).hexdigest(),
                code_verification="installed Python sources match pinned official commit; drift checking only")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_identity(folder, revision):
    folder = Path(folder)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("resolved full model revision required")
    vae = folder / "audiovae.safetensors"
    if not vae.is_file():
        vae = folder / "audiovae.pth"  # Official VoxCPM2 release; stock uses weights_only=True.
    files = [folder / "config.json", folder / "tokenizer.json", folder / "model.safetensors", vae]
    if any(not f.is_file() for f in files):
        raise FileNotFoundError("local config, tokenizer, model.safetensors and official AudioVAE weights required")
    config = json.loads(files[0].read_text())
    if config.get("architecture", "").lower() != "voxcpm2":
        raise ValueError("model architecture must be voxcpm2")
    return dict(repository="openbmb/VoxCPM2", revision=revision,
                files={f.name: sha256(f) for f in files}, **verified_code_identity())


def save_native(slider, path, *, identity, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(schema="voxcpm2-native-particles-v1", model_identity=identity,
                adapter=slider.metadata(), provenance=provenance, quality_qualified=False)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in slider.state_dict().items()},
              str(path), metadata={"document": json.dumps(meta, sort_keys=True, allow_nan=False)})
    return meta


def load_native(slider, path, *, identity):
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        raw = (stream.metadata() or {}).get("document")
    try:
        meta = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError as exc:
        raise ValueError("native checkpoint metadata missing or malformed") from exc
    if not isinstance(meta, dict) or not {"schema", "model_identity", "adapter", "provenance"} <= meta.keys():
        raise ValueError("native checkpoint metadata missing or malformed")
    if (meta["schema"] != "voxcpm2-native-particles-v1" or meta["model_identity"] != identity
            or meta["adapter"] != slider.metadata()):
        raise ValueError("native checkpoint identity/projection architecture differs")
    values = load_file(str(path), device="cpu")
    expected = slider.state_dict()
    if values.keys() != expected.keys():
        raise ValueError("native checkpoint keys differ")
    for key, value in values.items():
        if value.shape != expected[key].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("invalid native tensor: " + key)
    slider.load_state_dict(values, strict=True)
    return meta
