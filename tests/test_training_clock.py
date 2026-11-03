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
    from workflow import TARGETS, observation_due
    assert TARGETS == [.10,.15,.20,.30]
    assert all(observation_due(n) for n in [20,40,200,400,440,800])
    assert not any(observation_due(n) for n in [1,19,21,420,799])


def test_training_budget_charges_complete_updates_but_not_checkpoint_io():
    from workflow import train_budget
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
    from workflow import train_budget
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
    from workflow import summarize_targets
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
