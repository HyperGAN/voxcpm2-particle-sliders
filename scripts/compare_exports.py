"""Render matched native/distilled waveforms with stock VoxCPM generation.

Requires existing local weights and exact native teacher/export; no downloads.
Writes numeric fidelity plus listening files; perceptual gates stay pending.
"""
import argparse
from pathlib import Path
import json
import torch
import soundfile as sf
from voxcpm2_particle_sliders.runtime import load_local, read_manifest, gpu1_lease
from voxcpm2_particle_sliders.artifacts import load_native, sha256
from voxcpm2_particle_sliders.evaluation import waveform_metrics, evidence_template, stock_fixed_waveform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--distill", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--frames", type=int)
    p.add_argument("--steps", type=int)
    args = p.parse_args()
    with gpu1_lease(args.device):
        return _execute(args)


def _execute(args):
    if args.output.exists(): raise FileExistsError(args.output)
    export = json.loads((args.distill / "distillation.json").read_text())
    calibration = export["calibration"]
    args.frames = args.frames if args.frames is not None else calibration["frames"]
    args.steps = args.steps if args.steps is not None else calibration["steps"]
    if (args.frames, args.steps) != (calibration["frames"], calibration["steps"]):
        raise ValueError("stock comparison must use the fixed calibration duration/steps")
    if not 1 <= args.frames <= 64 or not 2 <= args.steps <= 32:
        raise ValueError("bounded comparison supports 1..64 patches and 2..32 Euler steps")
    if not calibration["cfg_zero_star"] or calibration["sway"] != 1.:
        raise ValueError("stock generate exposes only default zero-star/sway; this calibration requires a different sampler")
    if export["teacher_sha256"] != sha256(args.teacher):
        raise ValueError("distilled export teacher differs")
    model, slider, _, identity = load_local(args.model, args.revision, device=args.device,
        hosts=tuple(k for k in ("dit", "lm") if export["lora_config"]["enable_" + k]),
        network_profile=export.get("network_profile", "legacy"))
    try:
        load_native(slider, args.teacher, identity=identity)
        if export["model_identity"] != identity:
            raise ValueError("distilled export base identity differs")
        records = [r for r in read_manifest(args.manifest)["rows"] if r["split"] == "dev"]
        if not records or len(records) > 8:
            raise ValueError("bounded comparison requires 1..8 held-out records")
    except BaseException:
        slider.close()
        raise
    # Stock RNG is scoped and shared between matched runs. Fixed duration is
    # achieved by min_len above max_len; the actual stock KV path runs here.
    def render(record):
        return stock_fixed_waveform(model, record, calibration)
    args.output.mkdir(parents=True)
    native = {}
    try:
        for index, record in enumerate(records):
            for strength in (0., .5, 1., 2.):
                with slider.scaled(strength): audio = render(record)
                native[index, strength] = audio
                sf.write(args.output / f"row{index}-native-{strength:g}.wav", audio[0].cpu().numpy(), 48000)
    finally:
        slider.close()
    from voxcpm.model.voxcpm2 import LoRAConfig
    model.lora_config = LoRAConfig(**export["lora_config"])
    model._apply_lora()
    for module in model._iter_lora_modules():
        module.to(device=args.device, dtype=model._dtype())
    loaded, skipped = model.load_lora_weights(str(args.distill), device=args.device)
    expected = {name + suffix for name in slider.names for suffix in (".lora_A", ".lora_B")}
    if skipped or set(loaded) != expected:
        raise ValueError("stock loader skipped or omitted distilled projections")
    scores = []
    for index, record in enumerate(records):
        for strength in (0., .5, 1., 2.):
            for module in model._iter_lora_modules():
                module.scaling.fill_(module._base_scaling * strength)
            audio = render(record)
            sf.write(args.output / f"row{index}-distilled-{strength:g}.wav", audio[0].cpu().numpy(), 48000)
            scores.append(dict(row=record["id"], strength=strength,
                               **waveform_metrics(native[index, strength], audio)))
    report = dict(teacher_sha256=sha256(args.teacher),
        distill_sha256=sha256(args.distill / "lora_weights.safetensors"),
        manifest_sha256=sha256(args.manifest), model_identity=identity,
        calibration=calibration,
        sampler="stock generate; fixed length; min_len>max_len", scores=scores,
        evidence=evidence_template())
    report["evidence"]["waveform_fidelity"] = dict(status="measured", method="matched unaligned waveform and STFT")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__": main()
