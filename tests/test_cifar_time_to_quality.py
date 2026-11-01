"""Predeclared spaced quality goals, fair training clocks and honest censoring."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'apps'),str(ROOT/'tools')]


def test_goals_and_denser_sampling_are_declared_before_training():
    from cifar_time_to_quality import TARGETS, observation_due
    assert TARGETS == [.10,.15,.20,.30]
    assert all(observation_due(n) for n in [20,40,200,400,440,800])
    assert not any(observation_due(n) for n in [1,19,21,420,799])


def test_training_budget_charges_complete_updates_but_not_checkpoint_io():
    from cifar_time_to_quality import train_budget
    now=[0.]
    def step():
        now[0]+=2
        return dict(samples=8,optimizer_seconds=.5)
    def save(point):
        now[0]+=7
        return dict(checkpoint=str(point['steps']))
    r=train_budget(step,save,5,lambda n:n%2==0,clock=lambda:now[0])
    assert r['steps']==3 and r['samples_seen']==24 and r['training_seconds']==6
    assert r['last_step_seconds']==2 and r['checkpoint_seconds']==21
    assert r['acquisition_wall_seconds']==27 and r['optimizer_seconds']==1.5
    assert [p['steps'] for p in r['snapshots']]==[0,2,3]
    assert r['stop_reason']=='training_time_budget'


def test_failed_updates_retain_real_observation_prefix_and_charged_time():
    from cifar_time_to_quality import train_budget
    now=[0.];attempts=[0]
    def step():
        now[0]+=2;attempts[0]+=1
        if attempts[0]==3:raise FloatingPointError('nonfinite')
        return dict(samples=8,optimizer_seconds=.5)
    r=train_budget(step,lambda p:{},10,lambda n:n%2==0,clock=lambda:now[0])
    assert r['status']=='diverged' and r['steps']==2 and r['training_seconds']==6
    assert r['failed_attempt_step']==3 and [p['steps'] for p in r['snapshots']]==[0,2]


def paired_fixture():
    import hashlib
    h=lambda s:hashlib.sha256(str(s).encode()).hexdigest()
    records=[]
    for seed in [9,10,11]:
        for method in ['adamw','muon']:
            curve=[dict(steps=n,training_seconds=n*(1 if method=='adamw' else 1.1),precision=p,
                        training_order_sha256=h(n),augmentation_sha256=h(n),training_sample_order_sha256=h(n))
                   for n,p in [(0,.05),(20,.11),(40,.16),(60,.21)]]
            records.append(dict(optimizer=method,seed=seed,status='complete',protocol_sha256='protocol',
                                initial_parameter_sha256=h(seed),train_ids_sha256='train',validation_ids_sha256='val',
                                gallery_ids_sha256='gallery',query_ids_sha256='queries',curve=curve))
    return records


def test_cifar_targets_keep_all_seeds_and_unreached_higher_goals():
    from cifar_time_to_quality import summarize_targets
    records=paired_fixture()
    summary=summarize_targets(records)
    assert summary['first_observed'][1]['adamw']['steps']['mean']==40
    assert summary['first_observed'][-1]['adamw']['reached_count']==0
    assert summary['first_observed'][-1]['muon']['training_seconds'] is None
    records[-1]['curve']=records[-1]['curve'][:1]
    summary=summarize_targets(records)
    assert summary['first_observed'][0]['muon']['reached_count']==2
    assert summary['first_observed'][0]['muon']['training_seconds'] is None
    with pytest.raises(ValueError):summarize_targets(records[:-1])


@pytest.mark.parametrize('method',['adamw','muon'])
def test_native_update_order_matches_existing_pilot_with_different_checkpoint_schedule(tmp_path,method):
    import needle as ndl
    from cifar_model import load_checkpoint
    from cifar_pilot import make_protocol, train_worker
    from cifar_time_to_quality import TrainingState
    from time_to_quality import parameter_hash
    from common import save_json
    from types import SimpleNamespace
    labels=np.repeat(np.arange(3,dtype='uint8'),8)
    pixels=np.random.default_rng(1).integers(0,256,(24,3072),dtype='uint8')
    path=tmp_path/'train.bin';path.write_bytes(np.column_stack([labels//5,labels,pixels]).tobytes())
    args=SimpleNamespace(train_file=path,smoke=True,phase='final',optimizer=method,
                         candidate=method+'_1',output_dir=tmp_path)
    p=make_protocol(args,{m:dict(median_seconds=.1) for m in ['adamw','muon']})
    selected={m:dict(candidate=p['rate_grid'][m][1]) for m in ['adamw','muon']}
    save_json(tmp_path/'protocol.json',p);save_json(tmp_path/'selection.json',dict(selected=selected))
    train_worker(args,p)
    reference=json.loads((tmp_path/'final'/f'{method}_train.json').read_text())
    p=dict(p,selected=selected)
    state=TrainingState(p,method,8)
    assert state.initial_parameter_sha256==reference['initial_parameter_sha256']
    for _ in range(6):state()
    restored=load_checkpoint(tmp_path/reference['snapshots'][-1]['checkpoint'],ndl.cpu())
    assert parameter_hash(state.model)==parameter_hash(restored)
    assert state.metadata()['augmentation_sha256']==reference['augmentation_sha256']
    assert state.metadata()['training_order_sha256']==reference['training_order_sha256']


def test_native_budget_trajectory_pipeline_and_resume(tmp_path):
    from test_cifar_pilot import synthetic_binary
    path=tmp_path/'train.bin';synthetic_binary(path,classes=3,per_class=8)
    out=tmp_path/'run'
    command=[sys.executable,str(ROOT/'experiments/cifar_time_to_quality.py'),'--smoke',
             '--train-file',str(path),'--output-dir',str(out)]
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert done.returncode==0,done.stdout+done.stderr
    run=json.loads((out/'run.json').read_text())
    assert run['protocol']['targets']==[.10,.15,.20,.30]
    assert len(run['trajectories'])==2
    for record in run['trajectories']:
        assert record['status']=='complete' and record['stop_reason']=='training_time_budget'
        assert .03<=record['training_seconds']<=.03+record['last_step_seconds']+1e-6
        assert record['curve'][-1]['steps']==record['steps']
    before=(out/'seed9/final/adamw_train.json').read_bytes()
    resumed=subprocess.run(command+['--resume'],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    assert (out/'seed9/final/adamw_train.json').read_bytes()==before
