"""The time-to-quality table keeps spaced goals, capped failures and all seeds."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'experiments')]


def fixture():
    from build_report import json_hash
    from common import digest
    from cifar_time_to_quality import TARGETS, SEEDS, summarize_targets
    parent=json.loads((ROOT/'report/results.json').read_text())['cifar_pilot']['pilot']
    p=copy.deepcopy(parent['protocol'])
    for key in ('training_seed','tuning_epochs','final_epochs','rate_grid','selection_rule'):
        p.pop(key,None)
    p.update(study='cifar_time_to_quality',targets=TARGETS,budget_seconds=600.,final_seeds=SEEDS,
             selected=copy.deepcopy(parent['selection']['selected']),parent_pilot_sha256=json_hash(parent),
             parent_selection_sha256=parent['selection_sha256'],
             observation_rule='every20_updates_through400_then_every40;always_terminal',
             offline_quality_workers=3,no_metric_based_stopping=True)
    p['source_sha256']['experiments/cifar_time_to_quality.py']=digest(ROOT/'experiments/cifar_time_to_quality.py')
    h=lambda value:hashlib.sha256(str(value).encode()).hexdigest()
    ids=parent['final'][0]
    records=[]
    for seed in SEEDS:
        for method,step_seconds,steps in [('adamw',5.,120),('muon',4.,150)]:
            points=list(range(0,steps+1,20))
            if points[-1]!=steps:points.append(steps)
            snaps=[dict(steps=n,samples_seen=n//40*10000+(n%40)*256,training_seconds=n*step_seconds,
                        optimizer_seconds=n*.01,checkpoint=f'final/{method}_{n}.npz',checkpoint_sha256=h((method,n)),
                        parameter_sha256=h(seed) if n==0 else h((method,n)),training_order_sha256=h((n+39)//40),
                        training_sample_order_sha256=h(n),augmentation_sha256=h(n),acquisition_wall_seconds=n*step_seconds+.001*i)
                   for i,n in enumerate(points)]
            raw=dict(optimizer=method,seed=seed,candidate=p['selected'][method]['candidate'],
                     status='complete',stop_reason='training_time_budget',steps=steps,samples_seen=snaps[-1]['samples_seen'],
                     training_seconds=600.,optimizer_seconds=steps*.01,last_step_seconds=step_seconds,
                     checkpoint_seconds=.1,acquisition_wall_seconds=600.1,snapshots=snaps,
                     protocol_sha256=json_hash(p),initial_parameter_sha256=h(seed),
                     train_ids_sha256=p['train_ids_sha256'],validation_ids_sha256=p['validation_ids_sha256'],
                     total_parameter_count=54788,muon_eligible_parameter_count=34816,
                     muon_parameter_count=34816 if method=='muon' else 0,
                     optimizer_numeric_state_bytes=438304 if method=='adamw' else 299040,
                     epoch_order_sha256=[h(i) for i in range((steps+39)//40)],data_attempts=steps,
                     training_order_sha256=snaps[-1]['training_order_sha256'],
                     training_sample_order_sha256=h(steps),augmentation_sha256=h(steps),training_losses=[4.]*steps)
            curve=[dict(s,precision=round(.05+s['steps']*.001,4),same_category_matches=round((.05+s['steps']*.001)*10000),
                        validation_loss=4.,validation_accuracy=.1,evaluation_seconds=.1,neighbor_indices_sha256=h(s['steps'])) for s in snaps]
            records.append(dict(raw,training_record_sha256=json_hash(raw),curve=curve,
                                feature_archive='features.npz',feature_archive_sha256=h(method),
                                gallery_ids_sha256=ids['gallery_ids_sha256'],query_ids_sha256=ids['query_ids_sha256']))
    return dict(protocol=p,trajectories=records,summary=summarize_targets(records)),parent


def test_cifar_time_table_has_every_milestone_and_correct_model_costs():
    from build_report import collect_cifar_time_rows
    run,parent=fixture()
    rows=collect_cifar_time_rows(run,parent)
    assert len(rows)==8
    assert rows[0]['gallery_bytes']==2560000
    assert rows[0]['optimizer_state_bytes']==438304
    assert rows[1]['optimizer_state_bytes']==299040
    assert rows[0]['training_cost']==300 and rows[1]['training_cost']==240
    assert rows[-1]['training_cost'] is None and '0/3 reached' in rows[-1]['study']
    assert all('600s' in r['study'] for r in rows)


@pytest.mark.parametrize('change',['target','budget','pair','selection','clock','missing','summary','source'])
def test_cifar_time_table_rejects_changed_protocol_or_incomplete_pairing(change):
    from build_report import collect_cifar_time_rows
    run,parent=fixture()
    if change=='target':run['protocol']['targets']=[.11]
    if change=='budget':run['protocol']['budget_seconds']=1200
    if change=='pair':run['trajectories'][0]['initial_parameter_sha256']='changed'
    if change=='selection':run['protocol']['selected']['adamw']['candidate']['lr']=.1
    if change=='clock':run['trajectories'][0]['training_seconds']=599.
    if change=='missing':run['trajectories'].pop()
    if change=='summary':run['summary']['first_observed'][0]['adamw']['training_seconds']['mean']=1.
    if change=='source':run['protocol']['source_sha256']['apps/cifar_model.py']='changed'
    with pytest.raises(ValueError):collect_cifar_time_rows(run,parent)


def test_cifar_time_table_never_averages_only_successful_seeds():
    from build_report import collect_cifar_time_rows
    from cifar_time_to_quality import summarize_targets
    run,parent=fixture()
    for point in run['trajectories'][-1]['curve'][1:]:
        point.update(precision=.12,same_category_matches=1200)
    run['summary']=summarize_targets(run['trajectories'])
    rows=collect_cifar_time_rows(run,parent)
    assert rows[3]['training_cost'] is None and '2/3 reached' in rows[3]['study']
