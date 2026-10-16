#!/usr/bin/env python3
"""Expanded AlphaBrain fixed-action forecast matrix with one model load per arm.

The matrix reuses the released CosmosPolicy, public VAE, recorded executed
16-step actions, and observed +16-frame targets over three windows per episode.
Native and corrected fixed-action world-cut arms are evaluated in separate model
loads, with all latents and targets held fixed between arms.
"""
from __future__ import annotations
import gc, hashlib, json, os, pickle, sys, time
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/AlphaBrain/cosmos_policy_libero/model.safetensors'
TEXT=ROOT/'models/AlphaBrain/cosmos_policy_libero/libero_t5_embeddings.pkl'
PRETRAINED=ROOT/'models/AlphaBrain/Cosmos-Predict2-2B-Video2World'
OUT=Path(os.environ.get('ALPHABRAIN_MATRIX_OUT',str(ROOT/'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/summary.json')))
DATA_FILES=[Path(x) for x in os.environ.get('ALPHABRAIN_MATRIX_DATA_FILES',','.join(str(p) for p in sorted((ROOT/'data/worldagen_goal_probe_extra/data/chunk-000').glob('file-*.parquet')))).split(',') if x]
TASK_OFFSET=int(os.environ.get('ALPHABRAIN_MATRIX_GOAL_OFFSET','5'))
SEED=int(os.environ.get('ALPHABRAIN_MATRIX_SEED','1'))
BASE_FRAMES=[int(x) for x in os.environ.get('ALPHABRAIN_MATRIX_BASE_FRAMES','30,60,90').split(',')]

def load_arm():
    sys.path.insert(0,str(ROOT/'scripts'))
    import pilot_alphabrain_cosmos_policy_goal_swap as fixed
    if os.environ.get('ALPHABRAIN_USE_NATIVE_TE')!='1': fixed.install_transformer_engine_stub()
    sys.path.insert(0,str(ROOT/'vendor/AlphaBrain'))
    from AlphaBrain.model.modules.world_model.cosmos.latent_utils import replace_latent_with_action_chunk, replace_latent_with_proprio
    from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from safetensors.torch import load_file
    model=MinimalV1LVGDiT(max_img_h=240,max_img_w=240,max_frames=128,in_channels=16,out_channels=16,patch_spatial=2,patch_temporal=1,concat_padding_mask=True,model_channels=2048,num_blocks=28,num_heads=16,crossattn_emb_channels=1024,use_crossattn_projection=False,atten_backend='torch',pos_emb_cls='rope3d',mlp_ratio=4.0,use_adaln_lora=True,adaln_lora_dim=256,use_wan_fp32_strategy=False,pos_emb_learnable=True,rope_enable_fps_modulation=False,rope_h_extrapolation_ratio=3.0,rope_w_extrapolation_ratio=3.0).cuda().to(dtype=torch.bfloat16).eval()
    sd=load_file(str(MODEL),device='cpu'); sd={k.removeprefix('dit.'):v for k,v in sd.items() if k.startswith('dit.')}
    missing,unexpected=model.load_state_dict(sd,strict=False)
    # Transformer-Engine and the torch fallback register backend metadata
    # buffers with different names.  The released checkpoint contains every
    # learned parameter; backend-only unexpected keys are harmless, while a
    # missing learned key remains a hard failure.
    if missing: raise RuntimeError(f'state mismatch missing={len(missing)} unexpected={len(unexpected)}')
    del sd
    vae=WanVAEWrapper(str(PRETRAINED),dtype=torch.bfloat16,device='cuda',temporal_window=16).eval()
    return model,vae,CosmosPolicySampler(),arch_invariant_rand,replace_latent_with_proprio,replace_latent_with_action_chunk,fixed

def state_to_cosmos(row):
    raw=np.asarray(row['observation.state'],dtype=np.float32); quat=Rotation.from_rotvec(raw[3:6]).as_quat().astype(np.float32)
    return np.concatenate([raw[6:8],raw[:3],quat])

def image(row,key): return Image.open(BytesIO(row[key]['bytes'])).convert('RGB').transpose(Image.Transpose.ROTATE_180)

def prepare(rows,row,task_names,vae,replace_prop,replace_action,rand):
    ep_rows=sorted((r for r in rows if int(r['episode_index'])==int(row['episode_index'])),key=lambda r:int(r['frame_index']))
    frame=int(row['frame_index']); future=[r for r in ep_rows if int(r['frame_index'])>=frame+16]
    action_rows=[r for r in ep_rows if int(r['frame_index'])>=frame][:16]
    if len(action_rows)<16 or not future: raise RuntimeError(f'incomplete window ep={row["episode_index"]} frame={frame}')
    fr=future[0]
    p=np.array(image(row,'observation.images.image').resize((224,224)),copy=True); w=np.array(image(row,'observation.images.wrist_image').resize((224,224)),copy=True)
    fp=np.array(image(fr,'observation.images.image').resize((224,224)),copy=True); fw=np.array(image(fr,'observation.images.wrist_image').resize((224,224)),copy=True)
    def stack(primary,wrist):
        primary=torch.from_numpy(primary).permute(2,0,1).unsqueeze(1).expand(3,4,224,224); wrist=torch.from_numpy(wrist).permute(2,0,1).unsqueeze(1).expand(3,4,224,224); blank=torch.zeros_like(primary)
        return torch.cat([blank[:,:1],blank,wrist,primary,blank,blank,wrist,primary,blank],dim=1).unsqueeze(0).cuda(),blank
    video,blank=stack(p,w); target_video,_=stack(fp,fw)
    with torch.inference_mode(): encoded=vae.encode(video.to(torch.bfloat16)/127.5-1.0); target_encoded=vae.encode(target_video.to(torch.bfloat16)/127.5-1.0)
    latent=encoded.to(torch.bfloat16).clone(); target=target_encoded.to(torch.bfloat16).clone()
    prop=torch.tensor([state_to_cosmos(row)],device='cuda',dtype=torch.bfloat16); fprop=torch.tensor([state_to_cosmos(fr)],device='cuda',dtype=torch.bfloat16)
    action=torch.tensor(np.asarray([r['action'] for r in action_rows],dtype=np.float32),device='cuda',dtype=torch.bfloat16).unsqueeze(0)
    latent=replace_prop(latent,prop,torch.tensor([1],device='cuda')); latent=replace_action(latent,action,torch.tensor([4],device='cuda'))
    target=replace_prop(target,prop,torch.tensor([1],device='cuda')); target=replace_action(target,action,torch.tensor([4],device='cuda')); target=replace_prop(target,fprop,torch.tensor([5],device='cuda'))
    task_a=task_names[int(row['task_index'])]; task_b=task_names[(int(row['task_index'])+TASK_OFFSET)%len(task_names)]
    text=pickle.load(open(TEXT,'rb')); a=text[task_a].cuda().to(torch.bfloat16); b=text[task_b].cuda().to(torch.bfloat16); z=torch.zeros_like(a)
    cond=torch.zeros((1,1,9,28,28),device='cuda',dtype=torch.bfloat16); cond[:,:,:4]=1; cond[:,:,:5]=1
    return {'episode':int(row['episode_index']),'frame':frame,'future_frame':int(fr['frame_index']),'task_a':task_a,'task_b':task_b,'latent':latent,'target':target,'gt':latent[:,:,:5].float().clone(),'cond':cond,'padding':torch.zeros((1,1,224,224),device='cuda',dtype=torch.bfloat16),'fps':torch.tensor([16],device='cuda',dtype=torch.bfloat16),'noise':rand(tuple(latent.shape),seed=SEED).to(device='cuda',dtype=torch.float32)*80.0,'contexts':(a,b,z)}

def sample(model,sampler,item,ctx):
 latent=item['latent']; gt=item['gt']; cond=item['cond']; padding=item['padding']; fps=item['fps']; noise=item['noise']
 def x0_fn(xt,sigma):
    sigma=sigma.float(); s=sigma.view(-1,1,1,1,1); t=s/(s+1.0); inp=xt.float()*(1.0-t); inp[:,:,:5]=gt
    frame_noise=(sigma/(sigma+1.0)).view(-1,1).expand(-1,9)
    with torch.autocast('cuda',dtype=torch.bfloat16): out=model(inp.to(torch.bfloat16),frame_noise.to(torch.bfloat16),ctx,cond,fps=fps,padding_mask=padding)
    x0=(1.0-t)*xt.float()-t*out.float(); x0[:,:,:5]=gt; return x0
 with torch.inference_mode(): return sampler.forward(x0_fn=x0_fn,x_sigma_max=noise,num_steps=2,sigma_min=4.0,sigma_max=80.0)

def metrics(item,ys):
 target=item['target'].float()[:,:,5:8]
 def mse(y): return float((y.float()[:,:,5:8]-target).pow(2).mean().item())
 def slot_mse(y, slot):
  return float((y.float()[:, :, slot] - item['target'].float()[:, :, slot]).pow(2).mean().item())
 def delta(y,z): return float((y.float()-z.float()).abs().mean(dim=(0,1,3,4))[5:8].mean().item())
 out={'native_goal_world_delta_mean':delta(ys[0],ys[1]),'cut_goal_world_delta_mean':delta(ys[3],ys[4]),'native_task_a_world_forecast_mse':mse(ys[0]),'native_task_b_world_forecast_mse':mse(ys[1]),'cut_task_a_world_forecast_mse':mse(ys[3]),'cut_task_b_world_forecast_mse':mse(ys[4]),'zero_context_world_forecast_mse':mse(ys[2]),'repair_mse_delta':mse(ys[3])-mse(ys[0])}
 # Slots 5/6/7 are future proprioception, wrist view, and primary view.
 # These are reported separately so a Goal effect on physical-state slots
 # is not hidden by the aggregate visual metric.
 for slot, name in ((5, 'future_proprio'), (6, 'future_wrist'), (7, 'future_primary')):
  out[f'native_task_a_{name}_mse']=slot_mse(ys[0],slot)
  out[f'native_task_b_{name}_mse']=slot_mse(ys[1],slot)
  out[f'cut_task_a_{name}_mse']=slot_mse(ys[3],slot)
  out[f'cut_task_b_{name}_mse']=slot_mse(ys[4],slot)
  out[f'zero_context_{name}_mse']=slot_mse(ys[2],slot)
  out[f'native_goal_{name}_mse_delta']=out[f'native_task_b_{name}_mse']-out[f'native_task_a_{name}_mse']
  out[f'cut_goal_{name}_mse_delta']=out[f'cut_task_b_{name}_mse']-out[f'cut_task_a_{name}_mse']
  out[f'native_goal_{name}_abs_delta']=float((ys[0].float()[:, :, slot] - ys[1].float()[:, :, slot]).abs().mean().item())
  out[f'cut_goal_{name}_abs_delta']=float((ys[3].float()[:, :, slot] - ys[4].float()[:, :, slot]).abs().mean().item())
 return out

def main():
 start=time.monotonic(); torch.set_grad_enabled(False)
 import pyarrow.parquet as pq
 all_rows=[]
 for p in DATA_FILES: all_rows.extend(pq.read_table(p).to_pylist())
 task_table=pq.read_table(DATA_FILES[0].parents[2]/'meta/tasks.parquet').to_pylist(); task_names=[r.get('task',r.get('__index_level_0__')) for r in task_table]
 candidates=[]
 for ep in sorted(set(int(r['episode_index']) for r in all_rows)):
  er={int(r['frame_index']):r for r in all_rows if int(r['episode_index'])==ep}
  for frame in BASE_FRAMES:
   if frame in er: candidates.append(er[frame])
 if not candidates: raise RuntimeError('no matrix windows')
 model,vae,sampler,rand,replace_prop,replace_action,fixed=load_arm()
 items=[]; native=[]
 for i,row in enumerate(candidates):
  item=prepare(all_rows,row,task_names,vae,replace_prop,replace_action,rand); items.append(item)
  a,b,z=item['contexts']; ya=sample(model,sampler,item,a); yb=sample(model,sampler,item,b); yz=sample(model,sampler,item,z)
  native.append((ya.cpu(),yb.cpu(),yz.cpu())); del ya,yb,yz; print(json.dumps({'arm':'native','index':i,'episode':item['episode'],'frame':item['frame']}),flush=True)
 del model,vae; gc.collect(); torch.cuda.empty_cache()
 model,vae,sampler,rand,replace_prop,replace_action,fixed=load_arm(); fixed.install_fixed_action_world_cut(model)
 rows=[]
 for i,(item,nat) in enumerate(zip(items,native)):
  # Move native outputs back only for metrics; cut arm evaluated on identical item tensors.
  a,b,z=item['contexts']; ca=sample(model,sampler,item,a); cb=sample(model,sampler,item,b); cz=sample(model,sampler,item,z)
  ys=[nat[0].cuda(),nat[1].cuda(),nat[2].cuda(),ca,cb,cz]
  m=metrics(item,ys); m.update({'episode':item['episode'],'frame':item['frame'],'future_frame':item['future_frame'],'task_a':item['task_a'],'task_b':item['task_b'],'cut_mode':'fixed_action_world','sampler_seed':SEED})
  rows.append(m); print(json.dumps({'arm':'cut','index':i,'episode':item['episode'],'frame':item['frame'],**{k:m[k] for k in ['native_goal_world_delta_mean','cut_goal_world_delta_mean','repair_mse_delta']}}),flush=True)
  del ca,cb,cz,ys
 native_mean=float(np.mean([r['native_goal_world_delta_mean'] for r in rows])); cut_max=float(np.max([r['cut_goal_world_delta_mean'] for r in rows])); repair=float(np.mean([r['repair_mse_delta'] for r in rows]))
 out={'status':'completed_alphabrain_forecast_matrix','cut_mode':'fixed_action_world','sampler_seed':SEED,'data_files':[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in DATA_FILES],'windows':len(rows),'episodes':sorted(set(r['episode'] for r in rows)),'base_frames':BASE_FRAMES,'native_goal_world_delta_mean':native_mean,'native_goal_world_nonzero_all':all(r['native_goal_world_delta_mean']>0 for r in rows),'cut_goal_world_delta_max':cut_max,'cut_goal_world_exact_zero_all':all(r['cut_goal_world_delta_mean']==0 for r in rows),'repair_mse_delta_mean':repair,'repair_mse_delta_positive_fraction':float(np.mean([r['repair_mse_delta']>0 for r in rows])),'rows':rows,'scope':'One released AlphaBrain checkpoint, public compatible VAE, recorded executed action chunks fixed, three windows per episode across the extra LIBERO parquet files; latent held-out forecast metric, no environment rollout.','elapsed_seconds':time.monotonic()-start}
 OUT.parent.mkdir(parents=True,exist_ok=True); out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ['status','windows','native_goal_world_delta_mean','cut_goal_world_delta_max','repair_mse_delta_mean','repair_mse_delta_positive_fraction']},ensure_ascii=False))
if __name__=='__main__': main()
