from types import SimpleNamespace
import copy
import json
import pytest

from voxcpm2_particle_sliders.pilot import validate_budget,run_bounded,qualification_record,require_qualification


class BudgetFixture:
    """Clock/update-budget test double; not a GAN or training qualification."""
    def __init__(self,steps=100,adapter_metadata=None):
        self.completed_steps=0
        self.adapter_metadata=adapter_metadata or {"unit_fixture":True}
        self.recipe=SimpleNamespace(total_steps=steps,batch_size=2)
    def step(self):
        self.completed_steps+=1
        return dict(step=self.completed_steps)
    def state_dict(self):
        return dict(completed_steps=self.completed_steps,config=dict(pg_source_profile="explicit fixture identity; never admitted",
            pg_sources_sha256="fixture",penalty_input_domain="continuous",response_shape=[48000],response_domain="continuous_vector",
            runtime={"torch":"explicit fixture"},rows_sha256="fixture-rows",conditions_sha256="fixture-conditions",
            implementation_sources={"renderer":"fixture-source"},architectures={"critic":"fixture"},backend={"protocol":"fixture","adapter":self.adapter_metadata},
            row_count=2,sampling="balanced",response_scale=1.,sigma=1.,recipe={"total_steps":self.recipe.total_steps,"betas":(0.,.999)}),
            exposures={"g":[1,1]})


def test_pilot_budget_is_bounded_but_not_immutable_prototype_cap():
    validate_budget("pilot",100,900)
    with pytest.raises(ValueError):validate_budget("qualification",100,900)
    with pytest.raises(ValueError):validate_budget("pilot",201,900)
    with pytest.raises(ValueError):validate_budget("pilot",100,float("inf"))
    game=BudgetFixture()
    ticks=iter((0,0,1,2,3))
    result=run_bounded(game,wall_seconds=2,clock=lambda:next(ticks))
    assert result["stop_reason"]=="wall_time_budget" and game.completed_steps==2
    game=BudgetFixture(2)
    result=run_bounded(game,wall_seconds=900,clock=lambda:0)
    assert result["stop_reason"]=="update_budget" and game.completed_steps==2


def test_pilot_rejects_absent_or_mismatched_measurements():
    qualified=BudgetFixture(2);qualified.completed_steps=2
    kwargs=dict(model_identity={"explicit_unit_fixture":True},adapter_metadata={"profile":"fixture"},
        frozen_backbone=True,connected_finite_gradients=True,peak_allocated_bytes=100,peak_reserved_bytes=200,
        seconds_per_update=1.,gpu_uuid="unit-fixture-not-a-real-GPU")
    record=qualification_record(qualified,**kwargs)
    qualification_record(qualified,**kwargs,state_snapshot=qualified.state_dict())
    stale=qualified.state_dict();stale["completed_steps"]=1
    with pytest.raises(ValueError):qualification_record(qualified,**kwargs,state_snapshot=stale)
    game=BudgetFixture(20)
    def require(value):
        require_qualification(value,game,model_identity=kwargs["model_identity"],adapter_metadata=kwargs["adapter_metadata"],gpu_uuid=kwargs["gpu_uuid"])
    require(record)
    require(json.loads(json.dumps(record)))
    game.recipe.total_steps=100  # Fresh explicit horizon, not an optimizer resume.
    require(record)
    for key,value in (("peak_allocated_bytes",0),("gpu_uuid","other"),("model_identity",{}),
                      ("adapter_metadata",{}),("seconds_per_update",float("nan"))):
        modified=copy.deepcopy(record);modified[key]=value
        with pytest.raises(ValueError): require(modified)
    changed=copy.deepcopy(record);changed["game_source"]["response_shape"]=[480000]
    with pytest.raises(ValueError):require(changed)
    for key in ("rows_sha256","runtime","implementation_sources","architectures","backend","recipe_except_total_steps"):
        changed=copy.deepcopy(record);changed["game_source"][key]="different"
        with pytest.raises(ValueError):require(changed)
    with pytest.raises(ValueError):qualification_record(qualified,**dict(kwargs,peak_allocated_bytes=0))
    game.completed_steps=2
    with pytest.raises(ValueError,match="fresh game"):require(record)


@pytest.mark.parametrize("phase,updates,wall", [("unknown",2,900),("pilot",True,900),
    ("qualification",0,900),("qualification",11,900),("pilot",201,900),
    ("pilot",2,0),("pilot",2,1801),("pilot",2,float("nan"))])
def test_invalid_budget_bounds_are_refused(phase,updates,wall):
    with pytest.raises(ValueError):validate_budget(phase,updates,wall)


@pytest.mark.parametrize("option", ["--resume","--teacher"])
def test_pilot_cli_refuses_qualification_resume_or_warm_start_before_loading(tmp_path,monkeypatch,option):
    import voxcpm2_particle_sliders.cli as cli
    monkeypatch.setattr(cli,"load_local",lambda *a,**k:pytest.fail("model loaded before fresh-pilot refusal"))
    with pytest.raises(ValueError,match="fresh initialization"):
        cli.main(["train","--phase","pilot","--model",str(tmp_path),"--revision","a"*40,
                  option,str(tmp_path/"qualification-checkpoint"),"--output",str(tmp_path/"run")])
    assert not (tmp_path/"run").exists()


def test_cli_default_headroom_matches_eight_gib_pilot_plan(tmp_path,monkeypatch):
    import voxcpm2_particle_sliders.cli as cli
    received=[]
    monkeypatch.setattr(cli,"_execute",lambda args,parser:received.append(args.memory_headroom_mib))
    cli.main(["render","--device","cpu","--model",str(tmp_path),"--revision","a"*40,
              "--output",str(tmp_path/"unused")])
    assert received==[8192]


@pytest.fixture
def cli_game_fixture(tmp_path,monkeypatch):
    """CPU control-flow doubles; never native/CUDA/ParticleGAN qualification."""
    import torch
    import voxcpm2_particle_sliders.cli as cli
    import voxcpm2_particle_sliders.training as training
    game=BudgetFixture(1);closed=[]
    identity={"unit_fixture":True}
    model=torch.nn.Linear(1,1).requires_grad_(False)
    slider=SimpleNamespace(close=lambda:closed.append(True),metadata=lambda:{"unit_fixture":True})
    backend=SimpleNamespace(prepare=lambda row:row)
    monkeypatch.setattr(cli,"load_local",lambda *a,**k:(model,slider,backend,identity))
    monkeypatch.setattr(cli,"read_manifest",lambda p:{"rows":[{"id":"one","split":"train"},{"id":"two","split":"train"}]})
    monkeypatch.setattr(cli,"sha256",lambda p:"unit-fixture-digest")
    monkeypatch.setattr(cli,"native_row",lambda *a,**k:object())
    monkeypatch.setattr(cli,"conditions",lambda *a:torch.zeros(2,1))
    monkeypatch.setattr(training,"make_game",lambda *a,**k:game)
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda *a:None)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda *a:SimpleNamespace(uuid="GPU-00000000-0000-0000-0000-000000000001"))
    args=SimpleNamespace(command="train",phase="qualification",device="cpu",model=tmp_path,revision="a"*40,
        hosts=["dit"],network_profile="legacy",seed=171,teacher=None,manifest=tmp_path/"manifest",
        frames=1,steps=3,updates=1,resume=None,output=tmp_path/"output",max_wall_seconds=900,
        memory_headroom_mib=8192)
    return cli,args,game,closed,identity


def test_failed_update_propagates_without_saving_success(cli_game_fixture,monkeypatch):
    cli,args,game,closed,_=cli_game_fixture
    def fail():raise FloatingPointError("failed fixture update")
    monkeypatch.setattr(game,"step",fail)
    with pytest.raises(FloatingPointError,match="failed fixture update"):cli._execute(args,None)
    assert closed==[True] and not args.output.exists()


def test_incomplete_qualification_saves_diagnostics_without_checkpoints(cli_game_fixture):
    cli,args,game,closed,_=cli_game_fixture
    cli._execute(args,None)
    assert closed==[True]
    assert {p.name for p in args.output.iterdir()}=={"updates.json","run-budget.json","qualification-incomplete.json"}
    assert json.loads((args.output/"qualification-incomplete.json").read_text())["status"]=="incomplete"


def test_cli_rechecks_current_memory_after_matching_fresh_qualification(cli_game_fixture,monkeypatch,tmp_path):
    import torch
    from voxcpm2_particle_sliders.runtime import GPU1_UUID
    cli,args,game,closed,identity=cli_game_fixture
    qualified=BudgetFixture(2);qualified.completed_steps=2
    record=qualification_record(qualified,model_identity=identity,adapter_metadata={"unit_fixture":True},
        frozen_backbone=True,connected_finite_gradients=True,peak_allocated_bytes=100,peak_reserved_bytes=200,
        seconds_per_update=1.,gpu_uuid=GPU1_UUID)
    path=tmp_path/"qualification.json";path.write_text(json.dumps(record))
    args.phase="pilot";args.qualification_receipt=path
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda *a:(1024,2048))
    monkeypatch.setattr(torch.cuda,"memory_reserved",lambda *a:0)
    with pytest.raises(RuntimeError,match="memory headroom"):cli._execute(args,None)
    assert game.completed_steps==0 and closed==[True] and not args.output.exists()


@pytest.mark.parametrize("field", ["teacher","resume"])
def test_direct_pilot_execution_refuses_warm_start_before_any_load(tmp_path,monkeypatch,field):
    import voxcpm2_particle_sliders.cli as cli
    args=SimpleNamespace(command="train",phase="pilot",teacher=None,resume=None)
    setattr(args,field,tmp_path/"qualification-checkpoint")
    monkeypatch.setattr(cli,"load_local",lambda *a,**k:pytest.fail("model loaded before fresh-pilot refusal"))
    monkeypatch.setattr(cli.torch.cuda,"reset_peak_memory_stats",lambda *a:pytest.fail("CUDA accessed before refusal"))
    with pytest.raises(ValueError,match="fresh initialization"):cli._execute(args,None)


def test_qualification_resume_still_loads_checkpoint_before_updates(cli_game_fixture,monkeypatch):
    import torch
    from voxcpm2_particle_sliders.pilot import game_state_packet
    cli,args,game,closed,_=cli_game_fixture
    args.resume=args.model/"qualification-checkpoint"
    packet=game_state_packet(game.state_dict(),adapter_metadata={"unit_fixture":True});loaded=[]
    monkeypatch.setattr(torch,"load",lambda *a,**k:packet)
    monkeypatch.setattr(game,"load_state_dict",lambda state:loaded.append(state),raising=False)
    cli._execute(args,None)
    assert len(loaded)==1 and "voxcpm2_cohort" not in loaded[0]
    assert loaded[0]["config"]==packet["config"] and game.completed_steps==1 and closed==[True]


@pytest.mark.parametrize("source,target", [
    ("voxcpm2-hypergan-hndl-v1","voxcpm2-hypergan-hndl-singleton-v1"),
    ("voxcpm2-hypergan-hndl-singleton-v1","voxcpm2-hypergan-hndl-v1")])
def test_same_shape_game_cohort_refuses_both_directions_before_loader_mutation(source,target):
    import torch
    from voxcpm2_particle_sliders.pilot import game_state_packet,restore_game_state
    old={"network_profile":{"profile":source,"source":"explicit CPU fixture"}}
    new={"network_profile":{"profile":target,"source":"explicit CPU fixture"}}
    tensor=torch.zeros(2,2);calls=[]
    game=SimpleNamespace(load_state_dict=lambda packet:(calls.append(packet),tensor.copy_(packet["network"]["weight"])))
    raw=BudgetFixture(adapter_metadata=old).state_dict();raw["network"]={"weight":torch.ones_like(tensor)}
    packet=game_state_packet(raw,adapter_metadata=old)
    with pytest.raises(ValueError,match="cohort/profile identity"):
        restore_game_state(game,packet,adapter_metadata=new)
    assert not calls and torch.equal(tensor,torch.zeros_like(tensor))
    packet["voxcpm2_cohort"]=new["network_profile"]
    with pytest.raises(ValueError,match="embedded Vox game cohort"):
        restore_game_state(game,packet,adapter_metadata=new)
    assert not calls and torch.equal(tensor,torch.zeros_like(tensor))
    restore_game_state(game,game_state_packet(raw,adapter_metadata=old),adapter_metadata=old)
    assert len(calls)==1 and "voxcpm2_cohort" not in calls[0]
    assert torch.equal(tensor,torch.ones_like(tensor))


def test_unlabeled_historical_game_requires_explicit_original_compatibility():
    from voxcpm2_particle_sliders.pilot import game_state_packet,restore_game_state
    original={"network_profile":{"profile":"voxcpm2-hypergan-hndl-v1"}}
    singleton={"network_profile":{"profile":"voxcpm2-hypergan-hndl-singleton-v1"}}
    calls=[];game=SimpleNamespace(load_state_dict=lambda packet:calls.append(packet))
    for metadata in ({"unit_legacy_fixture":True},original):
        raw=BudgetFixture(adapter_metadata=metadata).state_dict()
        with pytest.raises(ValueError,match="explicit historical"):
            restore_game_state(game,raw,adapter_metadata=metadata)
        restore_game_state(game,raw,adapter_metadata=metadata,allow_historical_packets=True)
        assert calls[-1]==raw and "voxcpm2_cohort" not in raw
        malformed=game_state_packet(raw,adapter_metadata=metadata);malformed["voxcpm2_cohort"]=None
        with pytest.raises(ValueError,match="cohort/profile identity"):
            restore_game_state(game,malformed,adapter_metadata=metadata,allow_historical_packets=True)
    raw=BudgetFixture(adapter_metadata=singleton).state_dict();before=len(calls)
    with pytest.raises(ValueError,match="singleton rebinding"):
        restore_game_state(game,raw,adapter_metadata=singleton,allow_historical_packets=True)
    assert len(calls)==before


def test_qualification_cohort_is_bound_and_old_receipt_needs_explicit_compatibility():
    qualified=BudgetFixture(2);qualified.completed_steps=2;game=BudgetFixture(20)
    metadata={"unit_fixture":True}
    record=qualification_record(qualified,model_identity={"fixture":True},adapter_metadata=metadata,
        frozen_backbone=True,connected_finite_gradients=True,peak_allocated_bytes=100,peak_reserved_bytes=200,
        seconds_per_update=1.,gpu_uuid="fixture-only")
    kwargs=dict(model_identity={"fixture":True},adapter_metadata=metadata,gpu_uuid="fixture-only")
    assert record["voxcpm2_cohort"]=={"profile":"legacy"}
    changed=copy.deepcopy(record);changed["voxcpm2_cohort"]={"profile":"different"}
    with pytest.raises(ValueError,match="cohort/profile identity"):require_qualification(changed,game,**kwargs)
    del record["voxcpm2_cohort"]
    with pytest.raises(ValueError,match="explicit historical"):require_qualification(record,game,**kwargs)
    require_qualification(record,game,**kwargs,allow_historical_packets=True)


def test_gpu_admission_ignores_only_its_own_initial_context_and_preserves_studio_slot(monkeypatch):
    """Mocked query contract only: no CUDA/driver query is executed by this test."""
    import os
    import voxcpm2_particle_sliders.runtime as runtime
    gpu1="GPU-00000000-0000-0000-0000-000000000001"
    monkeypatch.setattr(runtime.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(runtime.torch.cuda,"get_device_properties",lambda device:SimpleNamespace(uuid=gpu1))
    monkeypatch.setattr(runtime.torch.cuda,"mem_get_info",lambda device:(48*1024**3,48*1024**3))
    monkeypatch.setattr(runtime.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout=f"{gpu1}, {os.getpid()}, 12 MiB\n"))
    runtime.require_idle_gpu("cuda:0")
    monkeypatch.setattr(runtime.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout=f"{gpu1}, {os.getpid()+1}, 12 MiB\n"))
    with pytest.raises(RuntimeError,match="compute work"):runtime.require_idle_gpu("cuda:0")
    monkeypatch.setattr(runtime.torch.cuda,"get_device_properties",lambda device:SimpleNamespace(uuid="GPU-00000000-0000-0000-0000-000000000000"))
    with pytest.raises(RuntimeError,match="studio ownership"):
        with runtime.training_gpu1_lease("cuda:0"):pass


@pytest.mark.parametrize("entry", ["render", "smoke", "distill", "compare"])
def test_all_cuda_entry_points_hold_gpu1_lease_and_unwind_on_failure(entry, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    import voxcpm2_particle_sliders.runtime as runtime
    import voxcpm2_particle_sliders.cli as cli
    monkeypatch.setattr(runtime, "GPU1_LOCK", tmp_path / "unit-test.lock")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "get_device_properties", lambda d: SimpleNamespace(uuid=runtime.GPU1_UUID))
    monkeypatch.setattr(runtime.torch.cuda, "mem_get_info", lambda d: (48*1024**3,48*1024**3))
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a,**kw: SimpleNamespace(returncode=0,stdout=""))
    def body(*args):
        assert runtime._ACTIVE_LEASE.get() == ("cuda:0",runtime.GPU1_UUID)
        raise RuntimeError("mock body failure")
    if entry == "compare":
        path = Path(__file__).resolve().parents[1] / "scripts/compare_exports.py"
        spec = importlib.util.spec_from_file_location("vox_compare_cpu_control_flow", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        monkeypatch.setattr(module, "_execute", body)
        monkeypatch.setattr(sys, "argv", ["compare", "--model",str(tmp_path),"--revision","a"*40,
            "--teacher",str(tmp_path/"teacher"),"--distill",str(tmp_path/"distill"),
            "--manifest",str(tmp_path/"manifest"),"--output",str(tmp_path/"output")])
        run = module.main
    else:
        monkeypatch.setattr(cli, "_execute", body)
        run = lambda: cli.main([entry,"--model",str(tmp_path),"--revision","a"*40,"--output",str(tmp_path/"output")])
    with pytest.raises(RuntimeError,match="mock body failure"): run()
    assert runtime._ACTIVE_LEASE.get() is None
    monkeypatch.setattr(runtime.torch.cuda,"get_device_properties",lambda d:SimpleNamespace(uuid="GPU-00000000-0000-0000-0000-000000000000"))
    with pytest.raises(RuntimeError,match="studio ownership"): run()
    assert runtime._ACTIVE_LEASE.get() is None


def test_cuda_loader_requires_lease_before_any_model_or_gpu_call(monkeypatch):
    import voxcpm2_particle_sliders.runtime as runtime
    monkeypatch.setattr(runtime.torch.cuda,"is_available",lambda:pytest.fail("unexpected CUDA query"))
    with pytest.raises(RuntimeError,match="active approved GPU1 lease"):
        runtime.load_local("unused", "a"*40, device="cuda:0")
    with runtime.gpu1_lease("cpu") as value:
        assert value is None


def test_busy_gpu1_lease_is_refused_without_leaking_context(tmp_path,monkeypatch):
    import fcntl
    import voxcpm2_particle_sliders.runtime as runtime
    path=tmp_path/"busy-fixture.lock"
    monkeypatch.setattr(runtime,"GPU1_LOCK",path)
    monkeypatch.setattr(runtime.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(runtime.torch.cuda,"get_device_properties",lambda d:SimpleNamespace(uuid=runtime.GPU1_UUID))
    with path.open("a+") as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match="another coordinator"):
            with runtime.gpu1_lease("cuda:0"):pytest.fail("busy lease admitted")
    assert runtime._ACTIVE_LEASE.get() is None


def test_cuda_uuid_normalization_preserves_exact_identity_and_rejects_unknown_objects():
    import uuid
    from voxcpm2_particle_sliders.runtime import canonical_gpu_uuid,GPU1_UUID
    expected=uuid.UUID(GPU1_UUID[4:])
    for value in [GPU1_UUID,str(expected),expected,SimpleNamespace(bytes=list(expected.bytes)),SimpleNamespace(bytes=expected.bytes)]:
        assert canonical_gpu_uuid(value)==GPU1_UUID
    for value in [object(),"not-a-uuid",SimpleNamespace(bytes=[0]*15),SimpleNamespace(bytes=[999]*16)]:
        with pytest.raises(RuntimeError,match="exact CUDA device UUID"):canonical_gpu_uuid(value)


def test_gpu0_sharing_is_stock_only_measured_and_preserves_default_guards(tmp_path,monkeypatch):
    import voxcpm2_particle_sliders.runtime as runtime
    probe=dict(purpose="stock_screening_memory_probe",real_cuda_probe=True,gpu_uuid=runtime.GPU1_UUID,
        models={"unit":"fixture"},max_input_samples=409600,sample_rate=16000,asr_max_tokens=128,
        batch_size=1,peak_reserved_bytes=2*1024**3)
    monkeypatch.setattr(runtime,"GPU0_SCREENING_LOCK",tmp_path/"stock-fixture.lock")
    monkeypatch.setattr(runtime.torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(runtime.torch.cuda,"get_device_properties",lambda d:SimpleNamespace(uuid=runtime.GPU0_UUID))
    monkeypatch.setattr(runtime.torch.cuda,"memory_reserved",lambda d:2*1024**3)
    state=dict(free_bytes=40*1024**3,utilization=35,processes=["existing Nano/game fixture; preserved"])
    monkeypatch.setattr(runtime,"_stock_gpu0_state",lambda:state.copy())
    args=dict(device="cuda:0",probe=probe,expected_models={"unit":"fixture"})
    with pytest.raises(RuntimeError,match="explicit stock-screening"):
        with runtime.stock_screening_gpu0_lease(**args):pass
    with runtime.stock_screening_gpu0_lease(**args,explicit_stock_sharing=True) as check:
        assert check()["processes"]==state["processes"]
        with pytest.raises(RuntimeError,match="active approved GPU1 lease"):runtime.require_active_gpu1_lease("cuda:0")
        with pytest.raises(RuntimeError,match="studio ownership"):
            with runtime.gpu1_lease("cuda:0"):pass
        state['free_bytes']=8*1024**3
        with pytest.raises(RuntimeError,match="budget exhausted"):check()
    assert runtime._ACTIVE_LEASE.get() is None


@pytest.mark.parametrize("change", [{"peak_reserved_bytes":7*1024**3},{"models":{"wrong":"model"}},
                                   {"real_cuda_probe":False},{"max_input_samples":800000}])
def test_gpu0_sharing_refuses_unmeasured_or_changed_workload(change):
    import voxcpm2_particle_sliders.runtime as runtime
    probe=dict(purpose="stock_screening_memory_probe",real_cuda_probe=True,gpu_uuid=runtime.GPU1_UUID,
        models={"unit":"fixture"},max_input_samples=409600,sample_rate=16000,asr_max_tokens=128,
        batch_size=1,peak_reserved_bytes=2*1024**3)
    probe.update(change)
    with pytest.raises(ValueError,match="matching measured"):
        with runtime.stock_screening_gpu0_lease("cuda:0",explicit_stock_sharing=True,probe=probe,expected_models={"unit":"fixture"}):pass
