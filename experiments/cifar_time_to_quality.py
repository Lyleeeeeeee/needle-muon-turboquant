"""Predeclared CIFAR retrieval milestones under equal pure-training time budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'apps')]
from common import digest, distribution, save_json
from run_report import run
from time_to_quality import THREADS, check_frozen, first_crossing, parameter_hash
os.environ.update(THREADS)
from cifar_pilot import array_hash, data_for, evaluate_worker, make_protocol as pilot_protocol, optimizer_for
from cifar_data import patches
from tune_batch512 import freeze

TARGETS=[.10,.15,.20,.30]
SEEDS=[9,10,11]


def observation_due(steps):
    return steps>0 and (steps%20==0 if steps<=400 else steps%40==0)


def train_budget(step, save_snapshot, seconds, due=observation_due, clock=time.perf_counter):
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Training budget must be positive and finite')
    record=dict(status='complete',steps=0,samples_seen=0,training_seconds=0.,optimizer_seconds=0.,
                checkpoint_seconds=0.,last_step_seconds=0.,snapshots=[])
    wall=clock()
    def save():
        row={k:record[k] for k in ('steps','samples_seen','training_seconds','optimizer_seconds')}
        before=clock()
        try:row.update(save_snapshot(row))
        finally:record['checkpoint_seconds']+=clock()-before
        row['acquisition_wall_seconds']=clock()-wall
        record['snapshots'].append(row)
    save()
    while record['training_seconds']<seconds:
        before=clock()
        try:update=step()
        except FloatingPointError as error:
            record['training_seconds']+=clock()-before
            record.update(status='diverged',failure=str(error),failed_attempt_step=record['steps']+1,
                          stop_reason='numerical_failure')
            break
        elapsed=clock()-before
        record['last_step_seconds']=elapsed
        record['training_seconds']+=elapsed
        record['steps']+=1
        record['samples_seen']+=update['samples']
        record['optimizer_seconds']+=update['optimizer_seconds']
        if due(record['steps']) or record['training_seconds']>=seconds:
            try:save()
            except FloatingPointError as error:
                record.update(status='diverged',failure=str(error),failed_attempt_step=record['steps'],
                              stop_reason='numerical_failure')
                break
    if record['status']=='complete':record['stop_reason']='training_time_budget'
    record['acquisition_wall_seconds']=clock()-wall
    return record


class TrainingState:
    """The pilot's exact update order with an independent, auditable data stream."""
    def __init__(self,p,method,seed):
        import numpy as np
        import needle as ndl
        from cifar_model import PatchMixer
        np.random.seed(seed)
        images,labels,train,_=data_for(p)
        self.labels=labels[train]
        self.prepared=patches(images[train],p['mean'],p['std'])
        self.mirrored=patches(images[train,:,:,::-1],p['mean'],p['std'])
        self.model=PatchMixer(**p['model'],device=ndl.cpu())
        self.optimizer=optimizer_for(self.model,method,p['selected'][method]['candidate'])
        self.initial_parameter_sha256=parameter_hash(self.model)
        self.rng=np.random.default_rng(seed)
        self.batch_size=p['batch_size']
        self.steps_epoch=math.ceil(len(train)/self.batch_size)
        self.updates=0;self.data_attempts=0;self.losses=[];self.epoch_hashes=[]
        self.order_hash=hashlib.sha256();self.sample_hash=hashlib.sha256();self.flip_hash=hashlib.sha256()

    def __call__(self):
        import numpy as np
        import needle as ndl
        from needle import nn
        offset=self.updates%self.steps_epoch
        if offset==0:
            self.order=self.rng.permutation(len(self.labels))
            raw=self.order.astype('int64').tobytes()
            self.order_hash.update(raw);self.epoch_hashes.append(hashlib.sha256(raw).hexdigest())
        ids=self.order[offset*self.batch_size:(offset+1)*self.batch_size]
        flips=self.rng.random(len(ids))<.5
        self.sample_hash.update(ids.astype('int64').tobytes());self.flip_hash.update(flips.tobytes())
        self.data_attempts+=1
        batch=self.prepared[ids];batch[flips]=self.mirrored[ids[flips]]
        self.optimizer.reset_grad()
        scores=self.model(ndl.Tensor(batch,device=ndl.cpu(),requires_grad=False))
        loss=nn.SoftmaxLoss()(scores,ndl.Tensor(self.labels[ids],device=ndl.cpu(),dtype='float32',requires_grad=False))
        value=float(loss.numpy().item())
        if not math.isfinite(value):raise FloatingPointError('Non-finite CIFAR training loss')
        loss.backward()
        before=time.perf_counter();self.optimizer.step();seconds=time.perf_counter()-before
        self.updates+=1;self.losses.append(value)
        return dict(samples=len(ids),optimizer_seconds=seconds)

    def metadata(self):
        return dict(training_order_sha256=self.order_hash.hexdigest(),
                    training_sample_order_sha256=self.sample_hash.hexdigest(),
                    augmentation_sha256=self.flip_hash.hexdigest())


def summarize_targets(records,targets=TARGETS,seeds=SEEDS):
    indexed={(r['optimizer'],r['seed']):r for r in records}
    if len(indexed)!=len(records) or set(indexed)!={(m,s) for s in seeds for m in ('adamw','muon')}:
        raise ValueError('Expected all unique paired CIFAR trajectories')
    for key in ('protocol_sha256','gallery_ids_sha256','query_ids_sha256'):
        if len({r[key] for r in records})!=1:raise ValueError('CIFAR trajectory cohorts or protocols differ')
    for seed in seeds:
        a,b=indexed['adamw',seed],indexed['muon',seed]
        for key in ('initial_parameter_sha256','train_ids_sha256','validation_ids_sha256'):
            if a[key]!=b[key]:raise ValueError('Paired CIFAR starts or split differ')
        points=[{r['steps']:r for r in model['curve']} for model in (a,b)]
        for step in set(points[0])&set(points[1]):
            for key in ('training_order_sha256','augmentation_sha256','training_sample_order_sha256'):
                if points[0][step][key]!=points[1][step][key]:raise ValueError('Paired CIFAR augmented data prefixes differ')
    result={}
    for name,consecutive in [('first_observed',1),('two_consecutive',2)]:
        rows=[]
        for target in targets:
            row=dict(target=target,consecutive=consecutive)
            for method in ('adamw','muon'):
                hits=[dict(seed=s,run_status=indexed[method,s]['status'],
                           **first_crossing(indexed[method,s]['curve'],target,consecutive)) for s in seeds]
                complete=all(h['reached'] for h in hits)
                row[method]=dict(reached_count=sum(h['reached'] for h in hits),total_count=len(hits),per_seed=hits,
                                 **{k:distribution([h[k] for h in hits]) if complete else None
                                    for k in ('steps','training_seconds','precision')})
            for key,suffix in [('steps','steps'),('training_seconds','time')]:
                a,b=[row[m][key] for m in ('adamw','muon')]
                row['paired_muon_over_adamw_'+suffix]=distribution([y/x for x,y in zip(a['values'],b['values'])]) if a and b and all(x>0 for x in a['values']) else None
            rows.append(row)
        result[name]=rows
    return result


def make_protocol(args):
    from types import SimpleNamespace
    parent=json.loads((ROOT/'report/results.json').read_text())['cifar_pilot']['pilot']
    if args.smoke:
        p=pilot_protocol(SimpleNamespace(smoke=True,train_file=args.train_file),
                         {m:dict(median_seconds=.1) for m in ('adamw','muon')})
    else:
        p=copy.deepcopy(parent['protocol'])
        if args.train_file and digest(args.train_file)!=p['data_sha256']:
            raise ValueError('CIFAR data differs from the fixed pilot split')
    for key in ('training_seed','tuning_epochs','final_epochs','rate_grid','selection_rule'):
        p.pop(key,None)
    parent_sha=hashlib.sha256((json.dumps(parent,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    p.update(study='cifar_time_to_quality',targets=TARGETS,budget_seconds=.03 if args.smoke else 600.,
             final_seeds=[9] if args.smoke else SEEDS,selected=copy.deepcopy(parent['selection']['selected']),
             parent_pilot_sha256=parent_sha,parent_selection_sha256=parent['selection_sha256'],
             observation_rule='every20_updates_through400_then_every40;always_terminal',
             offline_quality_workers=3,no_metric_based_stopping=True,
             timing_scope='complete_training_updates_including_NS;excludes_setup_snapshotIO_offline_quality',
             execution='six_sequential_training_workers_then_parallel_offline_quality',
             source_sha256=dict(p['source_sha256'],**{'experiments/cifar_time_to_quality.py':digest(Path(__file__))}))
    return p


def train_worker(args,p):
    import numpy as np
    from cifar_model import save_checkpoint
    state=TrainingState(p,args.optimizer,args.seed)
    folder=args.output_dir/f'seed{args.seed}'
    def save(point):
        if any(not np.isfinite(param.numpy()).all() for param in state.model.parameters()):
            raise FloatingPointError('Non-finite CIFAR model parameters')
        path=folder/'final'/f'{args.optimizer}_step{point["steps"]:06d}.npz'
        save_checkpoint(state.model,path)
        return dict(checkpoint=str(path.relative_to(folder)),checkpoint_sha256=digest(path),
                    parameter_sha256=parameter_hash(state.model),**state.metadata())
    record=train_budget(state,save,p['budget_seconds'])
    record.update(optimizer=args.optimizer,seed=args.seed,candidate=p['selected'][args.optimizer]['candidate'],
                  protocol_sha256=digest(args.output_dir/'protocol.json'),
                  initial_parameter_sha256=state.initial_parameter_sha256,
                  train_ids_sha256=p['train_ids_sha256'],validation_ids_sha256=p['validation_ids_sha256'],
                  total_parameter_count=sum(param.numpy().size for param in state.model.parameters()),
                  muon_eligible_parameter_count=sum(param.numpy().size for param in state.model.muon_parameters()),
                  muon_parameter_count=sum(param.numpy().size for param in state.model.muon_parameters()) if args.optimizer=='muon' else 0,
                  optimizer_numeric_state_bytes=state.optimizer.state_nbytes,training_losses=state.losses,
                  epoch_order_sha256=state.epoch_hashes,data_attempts=state.data_attempts,**state.metadata())
    save_json(folder/'final'/f'{args.optimizer}_train.json',record)


def checked_record(path,p,root):
    record=json.loads(path.read_text())
    if (record['protocol_sha256']!=digest(root/'protocol.json') or record['status'] not in ('complete','diverged')
            or record['candidate']!=p['selected'][record['optimizer']]['candidate']):
        raise ValueError('CIFAR trajectory provenance or selected rates changed')
    if record['status']=='complete' and not p['budget_seconds']<=record['training_seconds']<=p['budget_seconds']+record['last_step_seconds']+1e-6:
        raise ValueError('CIFAR elapsed training budget is invalid')
    folder=root/f'seed{record["seed"]}'
    for snap in record['snapshots']:
        if digest(folder/snap['checkpoint'])!=snap['checkpoint_sha256']:
            raise ValueError('CIFAR trajectory checkpoint changed')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',choices=['all','train','evaluate'],default='all')
    parser.add_argument('--train-file',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/cifar_time_to_quality_20261101')
    parser.add_argument('--worker',choices=['train','evaluate'],help=argparse.SUPPRESS)
    parser.add_argument('--optimizer',choices=['adamw','muon'],help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args();args.output_dir=args.output_dir.resolve()
    if args.train_file:args.train_file=args.train_file.resolve()
    if args.worker:
        p=json.loads((args.output_dir/'protocol.json').read_text());check_frozen(p)
        if digest(Path(p['train_file']))!=p['data_sha256']:raise ValueError('CIFAR input data changed')
        if args.worker=='train':train_worker(args,p)
        else:
            checked_record(args.output_dir/f'seed{args.seed}'/'final'/f'{args.optimizer}_train.json',p,args.output_dir)
            from types import SimpleNamespace
            evaluate_worker(SimpleNamespace(output_dir=args.output_dir/f'seed{args.seed}',phase='final',optimizer=args.optimizer),p)
        return
    p=make_protocol(args);args.output_dir.mkdir(parents=True,exist_ok=True)
    freeze(args.output_dir/'protocol.json',p,args.resume)
    jobs=[(seed,method) for i,seed in enumerate(p['final_seeds']) for method in (('adamw','muon') if i%2==0 else ('muon','adamw'))]
    for seed in p['final_seeds']:
        freeze(args.output_dir/f'seed{seed}'/'protocol.json',p,args.resume)
    def launch(seed,method,worker):
        command=[Path(__file__),'--worker',worker,'--optimizer',method,'--seed',seed,'--output-dir',args.output_dir]
        run(command,args.output_dir/f'{method}_seed{seed}_{worker}.log')
        print(f'DONE {method} seed{seed} {worker}',flush=True)
    if args.stage!='evaluate':
        for seed,method in jobs:
            path=args.output_dir/f'seed{seed}'/'final'/f'{method}_train.json'
            if args.resume and path.exists():checked_record(path,p,args.output_dir)
            else:launch(seed,method,'train')
    if args.stage=='train':return
    for seed,method in jobs:
        checked_record(args.output_dir/f'seed{seed}'/'final'/f'{method}_train.json',p,args.output_dir)
    with ThreadPoolExecutor(max_workers=p['offline_quality_workers']) as pool:
        futures=[]
        for seed,method in jobs:
            path=args.output_dir/f'seed{seed}'/'final'/f'{method}_curve.json'
            if not (args.resume and path.exists()):futures.append(pool.submit(launch,seed,method,'evaluate'))
        for future in as_completed(futures):future.result()
    records=[json.loads((args.output_dir/f'seed{seed}'/'final'/f'{method}_curve.json').read_text()) for seed,method in jobs]
    summary=summarize_targets(records,p['targets'],p['final_seeds'])
    save_json(args.output_dir/'run.json',dict(protocol=p,trajectories=records,summary=summary))
    print('Saved CIFAR matched-quality study',flush=True)


if __name__=='__main__':main()
