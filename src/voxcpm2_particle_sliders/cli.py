"""Private preparation, qualification, training, rendering and approximate export."""
import argparse
from pathlib import Path
import json

import torch
from .artifacts import load_native, save_native, sha256
from .backend import fixed_noise, make_prefix, waveform
from .runtime import resources, load_local, read_manifest, native_row, conditions


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "smoke", "render", "train", "distill"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--revision", help="resolved full Hugging Face model commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hosts", nargs="+", choices=["dit", "lm"], default=["dit"])
    from .hndl_binding import PROFILE
    from .singleton_binding import PROFILE as SINGLETON_PROFILE
    parser.add_argument("--network-profile", choices=["legacy", PROFILE, SINGLETON_PROFILE], default="legacy")
    parser.add_argument("--phase", choices=["qualification", "pilot"], default="qualification")
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=900.)
    parser.add_argument("--memory-headroom-mib", type=int, default=8192)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-historical-packets", action="store_true",
                        help="explicit unlabeled receipt/resume compatibility for legacy/original profile only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--text", default="We can start again when you are ready.")
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument("--strength", type=float, default=1.)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = resources(args.device)
        from particle_sliders.pg09 import validate_pg09_domain, PG_VERSION
        import hypergan
        result.update(hypergan_version=hypergan.__version__, shared_core_pg_version=PG_VERSION)
        options = {}
        if args.network_profile == SINGLETON_PROFILE:
            from .singleton_binding import engine_options, verified_runtime
            options = engine_options(SINGLETON_PROFILE)
            result["candidate_identity"] = verified_runtime()[2]
        try:
            validate_pg09_domain((1, 48000), response_domain="continuous_vector", **options)
        except (ValueError, RuntimeError) as exc:
            result["waveform_gan_admission"] = dict(status="blocked", reason=str(exc))
        else:
            result["waveform_gan_admission"] = dict(status="admitted")
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if not args.model or not args.revision:
        parser.error("--model local path and --revision full commit required")
    if args.output.exists():
        raise FileExistsError(args.output)
    # Gate real training before allocating the backbone or capturing targets.
    if args.command == "train":
        from .pilot import validate_budget
        validate_budget(args.phase, args.updates, args.max_wall_seconds)
        if args.phase == "pilot" and (args.resume or args.teacher):
            raise ValueError("pilot requires fresh initialization; --resume and --teacher warm starts are refused")
        if args.memory_headroom_mib < 0:
            raise ValueError("memory headroom is an operational nonnegative MiB budget")
        from particle_sliders.pg09 import validate_pg09_domain
        config = json.loads((args.model / "config.json").read_text())
        from math import prod
        # Official defaults apply when optional nested config is absent/null.
        rates = (config.get("audio_vae_config") or {}).get("decoder_rates", [8, 6, 5, 2, 2, 2])
        samples = args.frames * config.get("patch_size", 4) * prod(rates)
        from .singleton_binding import engine_options
        modern = args.network_profile in (PROFILE, SINGLETON_PROFILE)
        options = engine_options(args.network_profile) if modern else {}
        validate_pg09_domain((1, samples), response_domain="continuous_vector", **options)
        if not modern:
            raise ValueError("new training requires the genuine Vox HyperGAN/HNDL profile; legacy remains a historical render/checkpoint path")
        if not args.device.startswith("cuda:"):
            raise ValueError("real model training requires verified CUDA ownership; CPU tests are separate")
        if args.phase == "pilot" and not args.qualification_receipt:
            raise ValueError("pilot requires --qualification-receipt for the same source/model/adapter/duration/GPU")
    from .runtime import gpu1_lease
    with gpu1_lease(args.device):
        return _execute(args, parser)


def _execute(args, parser):
    if args.command == "train" and args.phase == "pilot" and (args.resume or args.teacher):
        raise ValueError("pilot requires fresh initialization; --resume and --teacher warm starts are refused")
    if args.command == "train":
        torch.cuda.reset_peak_memory_stats(args.device)
    model, slider, backend, identity = load_local(args.model, args.revision, device=args.device,
                                                 hosts=tuple(args.hosts), network_profile=args.network_profile,
                                                 adapter_seed=args.seed)
    try:
        if args.teacher:
            load_native(slider, args.teacher, identity=identity)
        if args.command in ("render", "smoke"):
            prefix = make_prefix(model, args.text)
            noise = fixed_noise(model, args.frames, args.seed)
            if args.command == "render":
                import soundfile as sf
                with torch.no_grad(), slider.scaled(args.strength):
                    audio = waveform(model, prefix, noise, steps=args.steps)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(args.output), audio[0].cpu().numpy(), 48000, subtype="FLOAT")
                write_json(str(args.output) + ".json", dict(identity=identity, text=args.text, seed=args.seed,
                    frames=args.frames, steps=args.steps, strength=args.strength,
                    teacher_sha256=sha256(args.teacher) if args.teacher else None,
                    sampler="fixed-length native", quality_qualified=False))
            else:
                # Diagnostic scalar ONLY; no optimizer, denoising loss or training.
                with slider.scaled(1.):
                    audio = waveform(model, prefix, noise, steps=args.steps)
                    audio.square().mean().backward()
                gradients = {n: dict(connected=p.grad is not None,
                    finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,
                    l1=float(p.grad.abs().sum()) if p.grad is not None else None)
                    for n, p in slider.named_parameters()}
                base_ok = all(not p.requires_grad and p.grad is None for p in model.parameters())
                write_json(args.output, dict(model_identity=identity, shape=list(audio.shape),
                    gradients=gradients, frozen_backbone=base_ok,
                    note="zero-up initialization gives zero routed/down/cloud gradients until an up update; connectivity is required",
                    waveform_quality_qualified=False))
                if not base_ok or not all(g["connected"] and g["finite"] for g in gradients.values()):
                    raise RuntimeError("gradient/frozen-base gate failed; inspect receipt")
        else:
            if not args.manifest:
                parser.error("--manifest required")
            document = read_manifest(args.manifest)
            provenance = dict(manifest_sha256=sha256(args.manifest),
                references={r[k]: sha256(r[k]) for r in document["rows"]
                            for k in ("neutral_reference", "positive_reference") if r.get(k)},
                source=document.get("provenance", {}), adapter_initialization_seed=args.seed,
                game_seed=args.seed)
            if args.teacher:
                provenance["warm_start_native_sha256"] = sha256(args.teacher)
            if args.command == "train":
                from .training import make_game
                records = [r for r in document["rows"] if r["split"] == "train"]
                prepared = [backend.prepare(native_row(model, r, frames=args.frames, steps=args.steps)) for r in records]
                if len(prepared) < 2:
                    raise ValueError("at least two training rows required")
                game = make_game(backend, prepared, conditions=conditions(model, prepared),
                                 total_steps=args.updates, seed=args.seed)
                from .pilot import run_bounded, require_qualification, qualification_record, game_state_packet, restore_game_state
                if args.resume:
                    restore_game_state(game, torch.load(args.resume, map_location=args.device, weights_only=True),
                        adapter_metadata=slider.metadata(),
                        allow_historical_packets=getattr(args, "allow_historical_packets", False))
                from .runtime import canonical_gpu_uuid
                gpu_uuid = canonical_gpu_uuid(torch.cuda.get_device_properties(args.device).uuid)
                if args.phase == "pilot":
                    measured = json.loads(args.qualification_receipt.read_text())
                    require_qualification(measured, game,
                        model_identity=identity, adapter_metadata=slider.metadata(), gpu_uuid=gpu_uuid,
                        allow_historical_packets=getattr(args, "allow_historical_packets", False))
                    free, _ = torch.cuda.mem_get_info(args.device)
                    available = free + torch.cuda.memory_reserved(args.device)
                    if measured["peak_reserved_bytes"] + args.memory_headroom_mib * 1024**2 > available:
                        raise RuntimeError("measured qualification peak plus explicit memory headroom does not fit current available capacity")
                outcome = run_bounded(game, wall_seconds=args.max_wall_seconds)
                logs = outcome["logs"]
                if not logs:
                    raise RuntimeError("no completed shared-game updates within the requested budget; nothing promoted")
                args.output.mkdir(parents=True)
                snapshot = game.state_dict()
                receipt = None
                write_json(args.output / "updates.json", logs)
                write_json(args.output / "run-budget.json", {k:v for k,v in outcome.items() if k != "logs"})
                if args.phase == "qualification":
                    covered = sum(int(v) > 0 for v in snapshot["exposures"]["g"])
                    if game.completed_steps < 2 or covered != snapshot["config"]["row_count"]:
                        write_json(args.output / "qualification-incomplete.json", dict(status="incomplete",
                            reason="need>=2 actual shared-game updates and exposure of every generator row",
                            cumulative_steps=game.completed_steps, generator_rows_covered=covered,
                            generator_rows=snapshot["config"]["row_count"]))
                        return
                    gradient_evidence = backend.gradient_evidence(prepared[0])
                    receipt = qualification_record(game, model_identity=identity, adapter_metadata=slider.metadata(),
                        frozen_backbone=all(not p.requires_grad and p.grad is None for p in model.parameters()),
                        connected_finite_gradients=gradient_evidence["connected_finite"],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                        seconds_per_update=outcome["seconds_per_update"], gpu_uuid=gpu_uuid,
                        state_snapshot=snapshot)
                    receipt["gradient_evidence"] = gradient_evidence
                torch.save(game_state_packet(snapshot, adapter_metadata=slider.metadata()), args.output / "training.ckpt")
                torch.save(game_state_packet(game.export_state_dict(), adapter_metadata=slider.metadata()), args.output / "game-export.ckpt")
                save_native(slider, args.output / "native.safetensors", identity=identity,
                            provenance=dict(**provenance, logs=logs))
                if receipt is not None:
                    write_json(args.output / "qualification.json", receipt)
            elif args.command == "distill":
                if not args.teacher:
                    parser.error("--teacher exact native checkpoint required")
                from .distill import capture, matrices, export_distilled, calibration_profile
                captures, ids = [], []
                for split in ("train", "dev"):
                    records = [r for r in document["rows"] if r["split"] == split]
                    if not records:
                        raise ValueError("both train and held-out dev records required")
                    with torch.no_grad(), slider.scaled(1.), capture(slider, model) as bucket:
                        for record in records:
                            row = native_row(model, record, frames=args.frames, steps=args.steps)
                            backend._run(row, row.neutral, 1.)
                    captures.append(matrices(bucket)); ids.append([r["id"] for r in records])
                export_distilled(slider, *captures, args.output, teacher=args.teacher, model_identity=identity,
                                 train_ids=ids[0], heldout_ids=ids[1], data_provenance=provenance,
                                 calibration=calibration_profile(frames=args.frames, steps=args.steps,
                                                               cfg_value=2., cfg_zero_star=True))
    finally:
        slider.close()


if __name__ == "__main__":
    main()
