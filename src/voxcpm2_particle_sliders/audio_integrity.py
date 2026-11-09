"""Local audio integrity and reference provenance checks; no models or remote calls."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from .artifacts import sha256
from .batch import TARGETS, registry_template


def inspect_audio(path):
    path = Path(path)
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    subtype = sf.info(str(path)).subtype
    hard, review = [], []
    if not len(audio): hard.append("empty_audio")
    if not np.isfinite(audio).all(): hard.append("nonfinite_samples")
    if hard:
        return dict(path=str(path.resolve()), sha256=sha256(path), hard_failures=hard,
                    review_flags=review, integrity_pass=False)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    bits = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}.get(subtype)
    positive_rail = 1. - 2.**(1-bits) if bits else 1.
    clipped = float(np.mean((audio <= -1.) | (audio >= positive_rail)))
    duration = len(audio) / rate
    window = max(1, int(rate * .02))
    windows = [float(np.sqrt(np.mean(audio[i:i+window]**2))) for i in range(0, len(audio), window)]
    near_silent = [value < 1e-4 for value in windows]
    tail_windows = 0
    for silent in reversed(near_silent):
        if not silent: break
        tail_windows += 1
    if peak == 0: hard.append("all_zero_audio")
    if duration < 3 or duration > 30: review.append("reference_outside_official_3_to_30_second_guidance")
    if clipped > .005: review.append("clipped_samples_over_provisional_0.5_percent_screen")
    if np.mean(near_silent) > .5: review.append("majority_near_silent_under_provisional_amplitude_screen")
    return dict(path=str(path.resolve()), sha256=sha256(path), bytes=path.stat().st_size,
        sample_rate=rate, channels=audio.shape[1], samples=len(audio), duration_seconds=duration,
        peak=peak, rms=rms, rms_dbfs=20 * math.log10(rms) if rms else None,
        dc_offset=float(audio.mean()), clipped_fraction=clipped, sample_subtype=subtype,
        positive_clipping_rail=positive_rail,
        near_silent_fraction=float(np.mean(near_silent)), trailing_near_silence_seconds=min(duration, tail_windows*.02),
        hard_failures=hard, review_flags=review, integrity_pass=not hard,
        quality_qualified=False, note="Screens flag review; amplitude/duration heuristics do not establish voice quality")


def audit_registry(path):
    path = Path(path)
    document = json.loads(path.read_text())
    results = []
    for record in document["references"]:
        pending = [key for key in ("path", "sha256", "speaker_id", "recording_session_id", "transcript", "origin")
                   if not record.get(key)]
        if record.get("verification_status") != "verified": pending.append("verification_status")
        result = dict(id=record["id"], status="pending", unresolved_fields=pending)
        if pending:
            results.append(result); continue
        target = Path(record["path"])
        target = target if target.is_absolute() else path.parent / target
        try:
            audio = inspect_audio(target)
        except (OSError, RuntimeError) as exc:
            result.update(status="invalid", error=str(exc)); results.append(result); continue
        result["audio"] = audio
        result["status"] = ("verified_integrity" if audio["integrity_pass"] and audio["sha256"] == record["sha256"]
                            else "invalid")
        if result["status"] == "verified_integrity" and audio["review_flags"] and record.get("audio_review_status") != "accepted_with_notes":
            result["status"] = "needs_audio_review"
        if audio["sha256"] != record["sha256"]: result["error"] = "reference_digest_mismatch"
        results.append(result)
    by_id = {r["id"]: r for r in document["references"]}
    pairing_errors = []
    expected = {r["id"] for r in registry_template()["references"]}
    if set(by_id) != expected:
        pairing_errors.append("reference registry must contain exactly the declared base/target train/dev IDs")
    if len(by_id) != len(document["references"]):
        pairing_errors.append("duplicate reference IDs")
    base_speaker = document.get("base_speaker_id")
    if not base_speaker:
        pairing_errors.append("verified shared base_speaker_id is unresolved")
    elif any(r.get("speaker_id") and r["speaker_id"] != base_speaker for r in by_id.values()):
        pairing_errors.append("all four targets and OFF/ON must use the verified shared base speaker")
    for target in TARGETS:
        for split in ("train", "dev"):
            a, b = by_id.get(f"base-{split}-neutral", {}), by_id.get(f"{target}-{split}-positive", {})
            if a.get("speaker_id") and b.get("speaker_id") and a["speaker_id"] != b["speaker_id"]:
                pairing_errors.append(f"{target}-{split}: neutral/positive speaker mismatch")
        for pole in ("neutral", "positive"):
            a, b = (by_id.get(f"base-{split}-neutral" if pole == "neutral" else f"{target}-{split}-positive", {})
                    for split in ("train", "dev"))
            if a.get("sha256") and a.get("sha256") == b.get("sha256"):
                pairing_errors.append(f"{target}-{pole}: train/dev recording reused")
            if a.get("recording_session_id") and a.get("recording_session_id") == b.get("recording_session_id"):
                pairing_errors.append(f"{target}-{pole}: held-out session not disjoint")
    return dict(schema="voxcpm2-reference-audit-v1", registry_sha256=sha256(path), rows=results,
                pairing_errors=pairing_errors, references_ready=bool(results) and not pairing_errors and
                all(r["status"] == "verified_integrity" for r in results), training_authorized_by_this_audit=False,
                note="Reference readiness does not replace source-profile/GPU/ownership/training gates")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit_registry(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(references_ready=result["references_ready"], rows=len(result["rows"]),
                          unresolved=sum(bool(r["unresolved_fields"]) for r in result["rows"]))))


if __name__ == "__main__": main()
