"""A one-seed CIFAR feasibility pilot must not masquerade as a three-seed win."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'experiments')]


def fixture():
    from build_report import json_hash
    from cifar_pilot import model_config, rate_grid, choose_candidate
    from common import digest
    from compare_galleries import METHODS, summarize_galleries
    old = json.loads((ROOT / 'report/results.json').read_text())['integrated']
    sources = dict(old['protocol']['source_sha256'])
    for path in ['apps/cifar_model.py', 'experiments/cifar_data.py', 'experiments/cifar_pilot.py']:
        sources[path] = digest(ROOT / path)
    p = dict(dataset='CIFAR-100', study='bounded_cifar_pilot', smoke=False, model=model_config(),
             batch_size=256, train_rows=10000, validation_rows=2000, train_per_class=100,
             validation_per_class=20, query_per_class=10, split_seed=17, training_seed=7, final_seeds=[8],
             official_test_used=False, tuning_epochs=5, final_epochs=20, rate_grid=rate_grid(),
             weight_decay=.01, k=10, metric='cosine', source_sha256=sources,
             backend=old['protocol']['backend'], train_ids_sha256='train', validation_ids_sha256='validation')

    def record(method, config, phase):
        seed = 7 if phase == 'tuning' else 8
        points = [0, 40, 120, 200] if phase == 'tuning' else [0, 40, 200, 400, 600, 800]
        factor = 1 + int(config['id'][-1]) if phase == 'tuning' else 4
        snaps = [dict(steps=n, samples_seen=n//40*10000, training_seconds=n*.2,
                      optimizer_seconds=n*.01, checkpoint=f'{method}_{phase}_{n}.npz',
                      checkpoint_sha256=hashlib.sha256(f'{method}_{phase}_{n}'.encode()).hexdigest(),
                      parameter_sha256='initial'+str(seed) if n == 0 else str(n)+method,
                      training_order_sha256=str(n), augmentation_sha256=str(n), acquisition_wall_seconds=n*.2+.01*i)
                 for i,n in enumerate(points)]
        curve = [dict(s, precision=.02+s['steps']*.00005*factor, validation_loss=5.-s['steps']*.001,
                      validation_accuracy=.01+s['steps']*.0001, evaluation_seconds=.1,
                      same_category_matches=round((.02+s['steps']*.00005*factor)*10000),
                      neighbor_indices_sha256='neighbors') for s in snaps]
        raw = dict(optimizer=method, seed=seed, candidate=config, status='complete',
                   protocol_sha256=json_hash(p), initial_parameter_sha256='initial'+str(seed),
                   training_order_sha256=str(points[-1]), augmentation_sha256=str(points[-1]),
                   train_ids_sha256='train', validation_ids_sha256='validation', total_parameter_count=54788,
                   muon_eligible_parameter_count=34816, muon_parameter_count=34816 if method == 'muon' else 0,
                   optimizer_numeric_state_bytes=438304 if method == 'adamw' else 299040,
                   steps=points[-1], samples_seen=points[-1]//40*10000, training_seconds=points[-1]*.2,
                   optimizer_seconds=points[-1]*.01, checkpoint_seconds=.1, acquisition_wall_seconds=points[-1]*.2+.1,
                   snapshots=snaps, training_losses=[4.] * points[-1])
        return dict(raw, training_record_sha256=json_hash(raw), curve=curve,
                    feature_archive=method+'_features.npz', feature_archive_sha256=method+'features',
                    gallery_ids_sha256='train', query_ids_sha256='queries')
    grid = rate_grid()
    tuning = [record(m,c,'tuning') for m in grid for c in grid[m]]
    selected = {m:choose_candidate([r for r in tuning if r['optimizer']==m]) for m in grid}
    selection = dict(selected=selected, tuning_sha256=hashlib.sha256(json.dumps(tuning,sort_keys=True).encode()).hexdigest())
    final = [record(m,selected[m]['candidate'],'final') for m in grid]
    pilot = dict(protocol=p, tuning=tuning, final=final, selection=selection, selection_sha256=json_hash(selection),
                 probe={m:dict(step_seconds=[.2,.2,.2],median_seconds=.2) for m in grid})
    reports=[];all_rows=[]
    quant_sources=dict(old['protocol']['source_sha256'])
    for relative in ['experiments/compare_galleries.py','experiments/uniform_compact.py']:
        quant_sources[relative]=digest(ROOT/relative)
    sizes=dict(fp32=2560000,uniform_2=240000,uniform_4=400000,compact_2=240001,compact_4=400001,
               symmetric_2=200001,symmetric_4=360001,turbo_2=216400,turbo_4=376448)
    for record_ in final:
        qp=dict(methods=METHODS,gallery_shape=[10000,64],query_shape=[1000,64],k=10,block_size=256,repeats=3,
                rotations=[7,8,9],metric='cosine',backend=p['backend'],source_sha256=quant_sources,
                feature_archive_sha256=record_['feature_archive_sha256'])
        rows=[]
        for method in METHODS:
            for rot in ([7,8,9] if method.startswith('turbo') else [7]):
                row=dict(method=method,rotation_seed=rot,stored_bytes=sizes[method],storage_breakdown={'all':sizes[method]},
                         device='cpu()',search_metric='cosine',normalization_timed=True,
                         label_precision_at_k=.18 if method=='fp32' else .17,recall_at_k=1. if method=='fp32' else .5,
                         query_seconds=[.19,.2,.21],query_seconds_median=.2,build_seconds=.01,relative_reconstruction_mse=.01,
                         compression_ratio_vs_fp32=2560000/sizes[method],serialized_path='example.npz',serialized_sha256='a'*64,
                         runner_provenance=dict(protocol_sha256=json_hash(qp),feature_archive_sha256=record_['feature_archive_sha256']))
                rows.append(row);all_rows.append(dict(row,encoder=record_['optimizer'],training_seed=8))
        reports.append(dict(encoder=record_['optimizer'],training_seed=8,protocol=qp,results=rows))
    return dict(pilot=pilot,reports=reports,quality_summary=summarize_galleries(all_rows,[8])),old


def test_cifar_pilot_rows_have_own_storage_costs_and_explicit_single_seed():
    from build_report import collect_cifar_rows, render_markdown
    data, old = fixture()
    rows=collect_cifar_rows(data,old)
    assert len(rows)==18
    assert all('pilot' in r['study'] and 'seed 8' in r['study'] for r in rows)
    assert rows[0]['gallery_bytes']==2560000 and rows[0]['compression']==1.
    assert rows[0]['training_cost']==160.
    assert all(r['precision_sd'] is None and r['training_sd'] is None for r in rows)
    assert '160.000 s' in render_markdown(rows) and '± 0.00' not in render_markdown(rows)


@pytest.mark.parametrize('change',['missing','selection','clock','fp32','source','repeat','summary'])
def test_cifar_pilot_report_rejects_incompatible_evidence(change):
    from build_report import collect_cifar_rows
    data,old=fixture()
    if change=='missing':data['pilot']['tuning'].pop()
    if change=='selection':data['pilot']['selection']['selected']['adamw']['candidate']['lr']=.9
    if change=='clock':data['pilot']['final'][0]['training_seconds']=1.
    if change=='fp32':data['reports'][0]['results'][0]['label_precision_at_k']=.9
    if change=='source':data['pilot']['protocol']['source_sha256']['apps/models.py']='changed'
    if change=='repeat':data['reports'][0]['results'].append(data['reports'][0]['results'][0])
    if change=='summary':data['quality_summary']['encoders']['adamw']['fp32']['label_precision_at10']['mean']=.9
    with pytest.raises(ValueError):collect_cifar_rows(data,old)


def compact_fixture():
    from build_report import json_hash
    from common import digest
    from compare_galleries import summarize_galleries
    old=json.loads((ROOT/'report/results.json').read_text())['integrated']
    source=dict(old['protocol']['source_sha256'])
    for relative in ['experiments/compare_galleries.py','experiments/uniform_compact.py']:
        source[relative]=digest(ROOT/relative)
    reports=[];trials=[]
    for f in old['feature_metadata']:
        encoder,seed=f['encoder'],f['training_seed']
        methods=['fp32','uniform_2','uniform_4','compact_2','compact_4']
        p=dict(methods=methods,gallery_shape=[55000,64],query_shape=[1000,64],k=10,block_size=256,repeats=3,
               rotations=[7,8,9],metric='cosine',backend=old['protocol']['backend'],source_sha256=source,
               feature_archive_sha256=f['feature_archive_sha256'])
        rows=[]
        for method in methods:
            previous=method.replace('compact','uniform')
            row=copy.deepcopy(next(r for r in old['retrieval_rows'] if
                (r['encoder'],r['training_seed'],r['method'])==(encoder,seed,previous)))
            row.update(method=method,rotation_seed=7,runner_provenance=dict(protocol_sha256=json_hash(p),feature_archive_sha256=f['feature_archive_sha256']))
            if method.startswith('compact'):
                row['storage_breakdown']=dict(codes=row['storage_breakdown']['codes'],scale=220000,minimum_payload=4,mode=1)
                row['stored_bytes']=sum(row['storage_breakdown'].values())
                row['compression_ratio_vs_fp32']=14080000/row['stored_bytes']
            rows.append(row);trials.append(dict(row,encoder=encoder,training_seed=seed))
        reports.append(dict(encoder=encoder,training_seed=seed,protocol=p,results=rows))
    return dict(parent_integrated_sha256=json_hash(old),reports=reports,
                quality_summary=summarize_galleries(trials,[8,9,10])),old


def test_compact_uniform_rows_preserve_quality_and_count_all_metadata():
    from build_report import collect_compact_rows
    bundle,old=compact_fixture()
    rows=collect_compact_rows(bundle,old)
    assert len(rows)==4
    assert [r['gallery_bytes'] for r in rows]==[1100005,1980005,1100005,1980005]
    assert rows[0]['precision_mean']==pytest.approx(old['quality_summary']['encoders']['adamw']['uniform_2']['label_precision_at10']['mean'])


@pytest.mark.parametrize('change',['precision','metadata','parent'])
def test_compact_uniform_report_rejects_changed_decode_or_wrong_parent(change):
    from build_report import collect_compact_rows
    bundle,old=compact_fixture()
    if change=='precision':bundle['reports'][0]['results'][3]['label_precision_at_k']-=.1
    if change=='metadata':bundle['reports'][0]['results'][3]['stored_bytes']-=5
    if change=='parent':bundle['parent_integrated_sha256']='changed'
    with pytest.raises(ValueError):collect_compact_rows(bundle,old)
