"""Operational budgets/qualification records around the shared game, no GAN law."""
import math
import time
import json


def cohort_identity(adapter_metadata):
    value = adapter_metadata.get("network_profile", {"profile": "legacy"})
    if not isinstance(value, dict) or not isinstance(value.get("profile"), str):
        raise ValueError("explicit Vox cohort identity required")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _require_cohort(packet, adapter_metadata, *, allow_historical_packets=False):
    expected = cohort_identity(adapter_metadata)
    if "voxcpm2_cohort" not in packet:
        from .hndl_binding import PROFILE
        if not allow_historical_packets or expected["profile"] not in ("legacy", PROFILE):
            raise ValueError("unlabeled packet requires explicit historical compatibility; singleton rebinding is refused")
    elif packet["voxcpm2_cohort"] != expected:
        raise ValueError("Vox checkpoint cohort/profile identity differs")


def _require_embedded_cohort(state, adapter_metadata):
    config = state.get("config")
    backend = config.get("backend") if isinstance(config, dict) else None
    embedded = backend.get("adapter") if isinstance(backend, dict) else None
    if not isinstance(embedded, dict) or cohort_identity(embedded) != cohort_identity(adapter_metadata):
        raise ValueError("embedded Vox game cohort/profile identity differs")


def game_state_packet(state, *, adapter_metadata):
    """Add a product identity without changing the shared game's state schema."""
    _require_embedded_cohort(state, adapter_metadata)
    if "voxcpm2_cohort" in state:
        _require_cohort(state, adapter_metadata)
    return dict(state, voxcpm2_cohort=cohort_identity(adapter_metadata))


def restore_game_state(game, packet, *, adapter_metadata, allow_historical_packets=False):
    """Reject product/source cohort mismatch before shared-loader mutation."""
    _require_cohort(packet, adapter_metadata, allow_historical_packets=allow_historical_packets)
    _require_embedded_cohort(packet, adapter_metadata)
    game.load_state_dict({k: v for k, v in packet.items() if k != "voxcpm2_cohort"})
    # Shared full-source/recipe/tensor validation still applies to its unchanged payload.


def _source_and_workload(config):
    keys = ("pg_source_profile", "pg_sources_sha256", "penalty_input_domain", "response_shape", "response_domain",
            "runtime", "implementation_sources", "architectures", "backend", "rows_sha256", "conditions_sha256",
            "row_count", "sampling", "response_scale", "sigma")
    result = {key:config[key] for key in keys}
    # A separately declared pilot horizon is a fresh game/cohort, never an
    # optimizer-resume alias. All other authoritative recipe settings must match.
    result["recipe_except_total_steps"] = {k:v for k,v in config["recipe"].items() if k != "total_steps"}
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def validate_budget(phase, updates, wall_seconds):
    maximum = 10 if phase == "qualification" else 200 if phase == "pilot" else None
    if maximum is None or type(updates) is not int or not 1 <= updates <= maximum:
        raise ValueError("qualification admits1..10 updates; measured pilot admits1..200 updates")
    if not math.isfinite(wall_seconds) or not 1 <= wall_seconds <= 1800:
        raise ValueError("wall-time budget must be1..1800 seconds")


def run_bounded(game, *, wall_seconds, clock=time.monotonic):
    start = clock()
    logs = []
    reason = "update_budget"
    while game.completed_steps < game.recipe.total_steps:
        if clock() - start >= wall_seconds:
            reason = "wall_time_budget"; break
        # The authoritative public shared game owns every update and finite/
        # connection check. A failed update propagates; never save it as success.
        logs.append(game.step())
    elapsed = clock() - start
    return dict(logs=logs, stop_reason=reason, elapsed_seconds=elapsed,
                completed_updates=len(logs), cumulative_steps=game.completed_steps,
                seconds_per_update=elapsed / len(logs) if logs else None,
                note="Wall budget checked between updates; it cannot interrupt a single in-flight native update")


def qualification_record(game, *, model_identity, adapter_metadata, frozen_backbone,
                         connected_finite_gradients, peak_allocated_bytes, peak_reserved_bytes,
                         seconds_per_update, gpu_uuid, state_snapshot=None):
    state = game.state_dict() if state_snapshot is None else state_snapshot
    if state_snapshot is not None and state.get("completed_steps") != game.completed_steps:
        raise ValueError("qualification snapshot is not the current completed shared-game state")
    _require_embedded_cohort(state, adapter_metadata)
    config = state["config"]
    covered = sum(int(value) > 0 for value in state["exposures"]["g"])
    if (game.completed_steps < 2 or not frozen_backbone or not connected_finite_gradients
            or peak_allocated_bytes <= 0 or peak_reserved_bytes < peak_allocated_bytes
            or seconds_per_update is None or not math.isfinite(seconds_per_update) or seconds_per_update <= 0
            or not gpu_uuid or covered != config["row_count"]):
        raise ValueError("real qualification requires>=2 successful game updates, actual CUDA memory/timing and frozen/gradient evidence")
    return dict(schema="voxcpm2-real-qualification-v1", model_identity=model_identity,
        voxcpm2_cohort=cohort_identity(adapter_metadata),
        adapter_metadata=adapter_metadata, successful_shared_game_updates=game.completed_steps,
        frozen_backbone=True, connected_finite_gradients=True, gpu_uuid=gpu_uuid,
        peak_allocated_bytes=peak_allocated_bytes, peak_reserved_bytes=peak_reserved_bytes,
        seconds_per_update=seconds_per_update,
        game_source=_source_and_workload(config), generator_rows_covered=covered,
        batch_size=game.recipe.batch_size,
        voice_quality_qualified=False)


def require_qualification(record, game, *, model_identity, adapter_metadata, gpu_uuid,
                          allow_historical_packets=False):
    if game.completed_steps != 0:
        raise ValueError("pilot must start a fresh game; qualification optimizer state cannot be resumed")
    _require_cohort(record, adapter_metadata, allow_historical_packets=allow_historical_packets)
    config = game.state_dict()["config"]
    _require_embedded_cohort({"config": config}, adapter_metadata)
    expected = _source_and_workload(config)
    if (record.get("schema") != "voxcpm2-real-qualification-v1"
            or record.get("model_identity") != model_identity or record.get("adapter_metadata") != adapter_metadata
            or record.get("game_source") != expected or record.get("gpu_uuid") != gpu_uuid
            or record.get("batch_size", 0) < game.recipe.batch_size
            or record.get("successful_shared_game_updates", 0) < 2
            or record.get("generator_rows_covered") != config["row_count"]
            or record.get("frozen_backbone") is not True or record.get("connected_finite_gradients") is not True
            or record.get("peak_allocated_bytes", 0) <= 0
            or record.get("peak_reserved_bytes", 0) < record.get("peak_allocated_bytes", 0)
            or not isinstance(record.get("seconds_per_update"), (int,float))
            or not math.isfinite(record["seconds_per_update"]) or record["seconds_per_update"] <= 0):
        raise ValueError("pilot requires an actual matching source/model/adapter/native-duration/GPU measured qualification")
