"""Ordinary native-model refit on existing random and on-policy transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mbrl.models import ModelTrainer
from mbrl.types import TransitionBatch
from mbrl.util.replay_buffer import BootstrapIterator, TransitionIterator
from pilot_mbrl_rank_general import make_model, model_hash, save


def main(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    model_path=Path(args.model)
    sources=[__file__,'scripts/pilot_mbrl_rank_general.py']
    save(out/'plan.json',dict(arguments=vars(args),
         hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
         data_scope='Previously evaluated native policy episodes become training/development data; use fresh evaluation seeds thereafter.',
         operation='Ordinary data aggregation and native GaussianMLP refit; baseline improvement, not the security contribution.',
         api_requests=0,gpu_work=False))
    keys=['obs','act','next_obs','rewards','terminateds','truncateds']
    old=np.load(model_path/'transitions.npz')
    records=[]
    for path in args.episodes:
        for t in json.loads(Path(path).read_text())['trajectory']:
            records.append((t['obs'],t['action'],t['next_obs'],t['reward'],t['terminated'],t.get('truncated',False)))
    extra=[np.asarray(x,dtype=bool if j>=4 else np.float32) for j,x in enumerate(zip(*records))]
    data=[np.concatenate([old[k],v]) for k,v in zip(keys,extra)]
    batch=TransitionBatch(*data)
    indices=rng.permutation(len(batch));split=int(.9*len(batch))
    tr,va=batch[indices[:split]],batch[indices[split:]]
    np.savez_compressed(out/'transitions.npz',**dict(zip(keys,data)),train_indices=indices[:split],validation_indices=indices[split:])
    model=make_model('halfcheetah');model.load(model_path);model.update_normalizer(tr)
    train=BootstrapIterator(tr,128,5,shuffle_each_epoch=True,permute_indices=False,rng=rng)
    val=TransitionIterator(va,256)
    trainer=ModelTrainer(model,optim_lr=.001,weight_decay=5e-5)
    start=time.monotonic();updates=0
    def callback(model,iteration,epoch,loss,score,best):
        row=dict(epoch=epoch,loss=float(loss),validation=float(score.mean()))
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if epoch%10==0:print(json.dumps(row),flush=True)
    def batch_callback(epoch,loss,meta,mode):
        nonlocal updates
        if mode=='train':updates+=1
    losses,_=trainer.train(train,val,num_epochs=args.epochs,patience=15,
                           callback=callback,batch_callback=batch_callback,silent=True)
    model.save(out)
    result=dict(status='trained',task='halfcheetah',gradient_updates=updates,epochs=len(losses),
                training_seconds=time.monotonic()-start,total_training_transitions=len(batch),
                reused_on_policy_transitions=len(records),native_collection_steps=0,
                validation_note='Repartitioned seen training data; validation is a training diagnostic, not independent model generalization evidence.',
                model_hash=model_hash(model),final_validation=float(trainer.evaluate(val).mean()),
                api_requests=0,gpu_work=False)
    save(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=19)
    main(p.parse_args())
