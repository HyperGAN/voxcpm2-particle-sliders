"""Reproducible local batch preparation. This module never loads a model or trains."""
import argparse
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path


TARGETS = ("sleepy", "medieval_town_herald", "auctioneer", "flat_announcer")
TARGET_SPECS = {
    "sleepy": ("Sleepy", "Projected drowsy reading; slower rate, within-phrase pauses and falling contours; intelligible, not whispery or intimate"),
    "medieval_town_herald": ("Medieval Town Herald", "Projected rhythmic proclamation cadence and phrase-final lengthening on plain modern prose; no inserted archaic wording"),
    "auctioneer": ("Auctioneer", "Bounded intelligible patter; no invented sale vocabulary or different speaker"),
    "flat_announcer": ("Flat Announcer", "Even energy and low pitch variation; maintain intelligibility and the same base speaker"),
}
TEXTS = {
    "train": ["The room is warm and quiet.", "I will wait beside the window.",
              "Please leave the lantern by the door.", "We can start again in the morning."],
    "dev": ["The package arrived before the rain.", "The race begins in ten seconds.",
            "Take your time; there is no hurry.", "We won the final round at last."],
}


def _json(path, document, *, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if "rows" in document and len(document["rows"]) > 16:
        # One request per line keeps larger reproducible manifests reviewable.
        header = json.dumps({k:v for k,v in document.items() if k != "rows"}, indent=2, sort_keys=True)
        lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in document["rows"]]
        text = header[:-1].rstrip() + ',\n  "rows": [\n    ' + ',\n    '.join(lines) + '\n  ]\n}\n'
    else:
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if private:
        with open(path, "x", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
            stream.write(text)
    else:
        path.write_text(text)


def _blinded_token(salt, domain, label):
    return hmac.new(salt, (domain + "\0" + label).encode(), hashlib.sha256).hexdigest()


def registry_template():
    definitions = [(f"base-{split}-neutral", None, split, "neutral") for split in ("train", "dev")]
    definitions += [(f"{target}-{split}-positive", target, split, "positive")
                    for target in TARGETS for split in ("train", "dev")]
    return dict(schema="voxcpm2-reference-registry-v1", base_speaker_id=None, references=[
        dict(id=ident, target=target, split=split, pole=pole,
             path=None, sha256=None, speaker_id=None, recording_session_id=None,
             transcript=None, origin=None, verification_status="pending")
        for ident, target, split, pole in definitions])


def prepare(destination):
    """Write drafts with explicit unresolved references; never invent recordings."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    salt = secrets.token_bytes(32)
    plan = dict(schema="voxcpm2-character-batch-plan-v1", status="blocked_preparation_only",
        targets=[dict(id=target, requested_label=TARGET_SPECS[target][0], direction=TARGET_SPECS[target][1],
                      target_definition_status="nova_hypothesis_unvalidated", references_status="pending")
                 for target in TARGETS],
        additional_targets=[], source_data_available=False,
        spoken_text_policy="Same neutral modern text for OFF/ON; avoid medieval wording as a style confound",
        reference_policy="One verified base speaker across OFF/ON and all four targets; shared neutral train/dev references; separate held-out recording sessions and text",
        dependency_gate=dict(status="pending", owner="shared-core inventory owner",
            required="Exact reviewed official explicit-continuous source profile and coherent core/PG/HyperGAN pins",
            bypass_allowed=False),
        network_integration_gate=dict(status="vox_local_implemented_cpu_candidate_tested",
            prototype="core0.11 TinyParticleProjection + Python WaveCritic; HyperGAN pin/schema witness only",
            inspected_candidate=dict(core_commit="a794e3b4a8813b012f2c4faa3da82b78d4a90214",
                core_version="0.12.0", profile="hypergan-hndl-v2", hndl_version="0.10.0",
                hypergan_commit="5f368a3770cf5bdae0448995c9f76e29af8049f7", particlegan_version="0.9.4"),
            unresolved=["Final official continuous-domain admission; current candidate is not wide-waveform qualified",
                "Final coherent default install pin after the owner selects the reviewed source profile"],
            implemented="Shared HNDL projection + Vox HyperGAN/HNDL waveform critic + public lora_up pre-hook retained-up distillation",
            old_cohort_policy="Preserve prototype identity; new graph/profile is a new source-bound cohort"),
        gpu_gate=dict(status="unverified", qualification_physical_minor=1,
            qualification_uuid="GPU-00000000-0000-0000-0000-000000000001",
            protected_physical_minor=0, protected_uuid="GPU-00000000-0000-0000-0000-000000000000",
            gpu0_owner="studio; retain ownership until actually released",
            admission="Live UUID/process/free-memory and owner lease verification required; no driver/permission changes"),
        qualification=dict(hosts=["dit"], frames=4, steps=10, updates=2, cfg_value=2., cfg_zero_star=True,
            requirements=["finite intended gradients and frozen backbone", "native checkpoint/export/reload",
                          "stock distilled reload", "measure peak allocated/reserved VRAM and wall time"],
            status="blocked_by_dependency_gpu_model_references"),
        pilot=dict(proposed_updates=100, maximum_updates=200, max_wall_seconds=900,
            status="blocked_until_matching_measured_real_qualification",
            qualification_max_updates=10,
            stop_criteria=["explicit update budget", "wall budget between updates",
                           "propagate shared-game nonfinite/disconnection/update failures; never promote a failed step"],
            note="Pilot requires exact source/model/adapter/native-duration/GPU qualification with actual memory/time evidence; no per-card concurrency assumed"),
        parallelism=dict(initial="One bounded qualification on GPU1", first_batch="One job per released/verified idle card",
            maximum_current_proposal=2, same_card_packing="Unqualified; needs measured combined peaks and reviewed memory-aware leases"),
        evaluation=dict(first_pass="2 held-out texts x 1 seed x native strengths0/1 =4 clips per target",
            expanded="4 held-out texts x 2 seeds x strengths0/.5/1/2 x native/distilled =64 clips per target",
            frames=48, steps=10, cfg_value=2., cfg_zero_star=True, sway=1.,
            trained_strength=1., strength2="extrapolation stress test; not a trained endpoint",
            measures=["blinded character identification including none-of-these", "transcription fidelity / WER/CER",
                      "naturalness and artifacts", "same-speaker consistency", "delivery under incongruent content"],
            supplementary="Optional low-pass prosody-only listening; not definitive proof of semantic disentanglement",
            nova_threshold_suggestions=dict(id_accuracy=.7, clarity_rating="4/5",
                status="provisional uncalibrated suggestions; not automatic acceptance criteria"),
            blinded_judge="Future local manifest only; no paid API calls or audio upload authorized here"),
        distillation=dict(status="separate_unqualified_gate", method="existing shared fit_routed_down; retained scaled up",
            requires_successful_native_candidate=True,
            evidence=["disjoint held-out projection MSE/R2", "stock loader and exact zero strength",
                      "native/distilled waveform, intelligibility, identity and expressivity comparisons"],
            promotion="Approximate; no arbitrary R2 threshold substitutes for real character fidelity"))
    _json(destination / "batch-plan.json", plan)
    _json(destination / "reference-registry.template.json", registry_template())
    render_rows, judge_rows, private_rows = [], [], []
    for target in TARGETS:
        rows = []
        for split, texts in TEXTS.items():
            for i, text in enumerate(texts):
                rows.append(dict(id=f"{target}-{split}-{i+1:02}", split=split, text=text,
                    seed=171+i, neutral_reference=None, positive_reference=None,
                    neutral_reference_id=f"base-{split}-neutral",
                    positive_reference_id=f"{target}-{split}-positive"))
        _json(destination / f"{target}.manifest.template.json", dict(
            schema="voxcpm2-slider-manifest-template-v1", executable=False,
            provenance=dict(target=target, reference_registry="reference-registry.template.json",
                            reference_status="pending_verified_recordings", generated_data_available=False), rows=rows))
        for index, text in enumerate(TEXTS["dev"]):
            for seed in (171, 2701):
                ids = {}
                for fmt in ("native", "distilled"):
                    for strength in (0., .5, 1., 2.):
                        key = f"{target}-dev{index+1}-seed{seed}-{fmt}-s{strength:g}"
                        opaque = _blinded_token(salt, "asset", key)[:32]
                        ids[fmt, strength] = opaque
                        render_rows.append(dict(id=key, asset_id=opaque, asset_status="not_rendered", path=None,
                            target=target, split="dev", text=text, seed=seed, format=fmt, strength=strength,
                            stage="first_pass" if index < 2 and seed == 171 and fmt == "native" and strength in (0., 1.) else "expanded",
                            frames=48, steps=10, cfg_value=2., cfg_zero_star=True, sway=1.,
                            neutral_reference_id="base-dev-neutral", model_identity=None,
                            content_condition=("plain_modern" if index == 0 else "incongruent_delivery_probe"),
                            requires_successful_native_candidate=(fmt == "distilled"),
                            native_teacher_sha256=None, distilled_sha256=None, observed_audio_sha256=None,
                            output_sample_rate=48000, quality_qualified=False))
                for fmt in ("native", "distilled"):
                    for strength in (.5, 1., 2.):
                        label = f"{target}:{index}:{seed}:{fmt}:{strength}"
                        pair_id = _blinded_token(salt, "pair", label)[:32]
                        assets = [ids[fmt, 0.], ids[fmt, strength]]
                        # Independent secret domain: public pair IDs reveal no swap bit.
                        swapped = int(_blinded_token(salt, "swap", label)[-1], 16) % 2 == 1
                        if swapped: assets.reverse()
                        judge_rows.append(dict(pair_id=pair_id, asset_a=assets[0], asset_b=assets[1],
                            text=text, status="awaiting_local_audio_and_judge_authorization"))
                        private_rows.append(dict(pair_id=pair_id, target=target, format=fmt, strength=strength,
                            seed=seed, text_index=index+1, off_position="B" if swapped else "A",
                            intended_axis_definition_status="nova_hypothesis_unvalidated"))
    _json(destination / "evaluation-requests.json", dict(schema="voxcpm2-evaluation-requests-v1", executable=False,
        refs_verified=False, rows=render_rows), private=True)
    _json(destination / "judge-pairs.blinded.json", dict(schema="voxcpm2-blinded-pairs-v1", upload_authorized=False,
        instructions="Use opaque asset names; no slider/format/strength labels. Blinding key stays local.", rows=judge_rows))
    _json(destination / "judge-pairs.private-key.json", dict(schema="voxcpm2-private-blinding-key-v2",
        salt_hex=salt.hex(), algorithm="HMAC-SHA256; independent asset/pair/swap domains", rows=private_rows), private=True)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepare(args.output)


if __name__ == "__main__": main()
