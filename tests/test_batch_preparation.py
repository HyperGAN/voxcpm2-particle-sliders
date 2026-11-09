import json
import numpy as np
import pytest
import soundfile as sf

from voxcpm2_particle_sliders.batch import prepare, TARGETS, TEXTS, registry_template
from voxcpm2_particle_sliders.audio_integrity import inspect_audio, audit_registry
from voxcpm2_particle_sliders.artifacts import sha256


def test_plan_is_reproducible_except_secret_blinding_and_keeps_references_pending(tmp_path, monkeypatch):
    import copy
    import hashlib
    import voxcpm2_particle_sliders.batch as batch
    captured = {}
    def capture(path, document, **kwargs):
        captured[path.name] = (copy.deepcopy(document), kwargs)
    monkeypatch.setattr(batch, "_json", capture)
    salts = iter((b"a" * 32, b"b" * 32))
    monkeypatch.setattr(batch.secrets, "token_bytes", lambda count: next(salts))
    a, b = tmp_path / "a", tmp_path / "b"
    prepare(a); first = copy.deepcopy(captured); captured.clear(); prepare(b)
    # Only fresh in-memory fixtures: never read/write an excluded private file.
    for name in first:
        if name not in {"evaluation-requests.json", "judge-pairs.blinded.json", "judge-pairs.private-key.json"}:
            assert first[name] == captured[name]
    private, options = first["judge-pairs.private-key.json"]
    assert options == {"private": True} and private["salt_hex"] == (b"a"*32).hex()
    assert all((b"a"*32).hex() not in json.dumps(doc) for name,(doc,kw) in first.items()
               if name != "judge-pairs.private-key.json")
    left = first["judge-pairs.blinded.json"][0]; right = captured["judge-pairs.blinded.json"][0]
    assert len(left["rows"]) == 192 and not left["upload_authorized"]
    assert {r["pair_id"] for r in left["rows"]}.isdisjoint(r["pair_id"] for r in right["rows"])
    evaluation = first["evaluation-requests.json"][0]
    assert len(evaluation["rows"]) == 256 and sum(r["stage"] == "first_pass" for r in evaluation["rows"]) == 16
    assert all(r["path"] is None and r["asset_status"] == "not_rendered" for r in evaluation["rows"])
    assert all(r["asset_id"] != hashlib.sha256(r["id"].encode()).hexdigest()[:16] for r in evaluation["rows"])
    assert any((int(r["pair_id"][-1],16)%2 == 1) != (r["off_position"] == "B") for r in private["rows"])
    for row in private["rows"]:
        label = f"{row['target']}:{row['text_index']-1}:{row['seed']}:{row['format']}:{row['strength']}"
        swapped = int(batch._blinded_token(b"a"*32, "swap", label)[-1],16)%2 == 1
        assert row["off_position"] == ("B" if swapped else "A")
    for target in TARGETS:
        manifest = first[f"{target}.manifest.template.json"][0]
        assert not manifest["executable"] and all(r["positive_reference"] is None for r in manifest["rows"])
    registry = first["reference-registry.template.json"][0]
    assert len(registry["references"]) == 10 and all(r["verification_status"] == "pending" for r in registry["references"])
    assert set(TEXTS["train"]).isdisjoint(TEXTS["dev"])
    a.mkdir()
    with pytest.raises(FileExistsError): prepare(a)


def test_private_output_permissions_with_nonsecret_nonaudio_fixture(tmp_path):
    import stat
    from voxcpm2_particle_sliders.batch import _json
    path=tmp_path/"permissions.fixture"
    _json(path,{"fixture":True},private=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):_json(path,{"fixture":True},private=True)


def test_local_audio_integrity_distinguishes_hard_errors_from_review_flags(tmp_path):
    rate = 16000
    tone = .1 * np.sin(2*np.pi*220*np.arange(rate*4)/rate)
    normal = tmp_path / "normal.wav"; sf.write(normal, tone, rate, subtype="FLOAT")
    result = inspect_audio(normal)
    assert result["integrity_pass"] and result["duration_seconds"] == 4
    assert not result["quality_qualified"] and not result["review_flags"]
    silent = tmp_path / "silent.wav"; sf.write(silent, np.zeros(rate), rate, subtype="FLOAT")
    assert "all_zero_audio" in inspect_audio(silent)["hard_failures"]
    invalid = tmp_path / "invalid.wav"; bad = tone.copy(); bad[0] = np.nan
    sf.write(invalid, bad, rate, subtype="FLOAT")
    assert "nonfinite_samples" in inspect_audio(invalid)["hard_failures"]
    clipped = tmp_path / "clipped.wav"; sf.write(clipped, np.ones(rate*4), rate, subtype="FLOAT")
    clipped_report = inspect_audio(clipped)
    assert clipped_report["integrity_pass"] and clipped_report["review_flags"]


def test_reference_audit_requires_complete_same_speaker_disjoint_digest_and_sessions(tmp_path):
    registry = registry_template(); registry["base_speaker_id"] = "test-speaker"
    rate = 16000
    for i, record in enumerate(registry["references"]):
        path = tmp_path / f"ref{i}.wav"
        sf.write(path, .1*np.sin(2*np.pi*(220+i)*np.arange(rate*3)/rate), rate, subtype="FLOAT")
        record.update(path=path.name, sha256=sha256(path), speaker_id="test-speaker",
                      recording_session_id=f"{record['id']}-session", transcript="synthetic test fixture only",
                      origin="unit test sine wave; not speech or training data", verification_status="verified")
    path = tmp_path / "registry.json"
    def audit():
        path.write_text(json.dumps(registry)); return audit_registry(path)
    assert audit()["references_ready"]  # Integrity/provenance consistency only; never a speech quality gate.
    registry["references"][-1]["speaker_id"] = "different-test-speaker"
    assert not audit()["references_ready"]
    registry["references"][-1]["speaker_id"] = "test-speaker"
    registry["references"][1]["recording_session_id"] = registry["references"][0]["recording_session_id"]
    assert not audit()["references_ready"]
    registry["references"][1]["recording_session_id"] = "different-session"
    registry["references"][-1]["sha256"] = "0" * 64
    assert not audit()["references_ready"]
    registry["references"] = registry["references"][:1]
    assert not audit()["references_ready"]
