---
license: mit
language:
  - en
library_name: pytorch
pipeline_tag: robotics
tags:
  - robotics
  - vla
  - vision-language-action
  - cosmos-policy
  - world-model
  - diffusion-transformer
  - libero
datasets:
  - LIBERO
---

# CosmosPolicy · LIBERO-Goal (10 k steps)

> World-model-native Vision-Language-Action (VLA) checkpoint released
> with the [AlphaBrain](https://github.com/AlphaBrainGroup/AlphaBrain)
> framework. Provided for direct download and evaluation — no
> retraining needed.

A CosmosPolicy model: a 3-D DiT (diffusion transformer) that predicts
action chunks in the latent space of a world model, conditioned on T5
text embeddings and the robot's proprioceptive state. This is the
**10 000-step checkpoint** from the `cosmos-8gpu-bs20-fix2-0408` run
on the LIBERO-Goal task suite (chunk size 16).

## Overview

| | |
|:---|:---|
| **Architecture**        | CosmosPolicy (3-D DiT + T5 text cond + proprioception) |
| **Backbone DiT**        | 28 blocks · 16 heads · `model_channels=2048` · rope3d positional embedding |
| **Max image size**      | 240 × 240, up to 128 frames                              |
| **Action head**         | Continuous chunk decoding · `action_dim=7`, `chunk_size=16` |
| **Proprioception**      | 9-dim · `state_t=9`                                       |
| **Diffusion**           | σ-training ∈ [0.01, 200.0]; σ-inference ∈ [4.0, 80.0]; 5 inference steps |
| **Training data**       | LIBERO-Goal · Cosmos-Policy format · `use_stronger_image_aug: true` |
| **Optimiser**           | AdamW · `lr_base = 1e-4` · `warmup = 1000` · `lambda_linear` schedule |
| **Hardware / batch**    | 8 GPUs · `bs=20/device` · `grad_acc=12` → effective batch 1920 |
| **Training step**       | 10 000 / 40 000 (released checkpoint)                     |

## Files

```
├── README.md                             model card
├── framework_config.yaml                 AlphaBrain / CosmosPolicy training config
├── config.json                           minimal HF-style metadata (arch, dims, step)
├── model.safetensors                     framework checkpoint (3.9 GB) — primary load path
├── cosmos_dit.pt                         DiT-only PyTorch tensor file (3.7 GB) — alt loader / resume
├── libero_dataset_statistics.json        action normalisation statistics
├── libero_t5_embeddings.pkl              precomputed T5 embeddings for LIBERO prompts (~41 MB)
└── resume_meta.json                      training resume metadata
```

## Usage

```bash
git clone https://github.com/AlphaBrainGroup/AlphaBrain.git
cd AlphaBrain
pip install -e .

huggingface-cli download AlphaBrainGroup/cosmos-policy-libero-goal \
    --local-dir ./cosmos_policy_ckpt

# Launch the WebSocket inference server (CosmosPolicy route)
python deployment/model_server/server_policy.py \
    --ckpt_path ./cosmos_policy_ckpt --port 10093 --use_bf16
```

For the LIBERO-Goal evaluation pipeline, see
[`benchmarks/LIBERO/eval/`](https://github.com/AlphaBrainGroup/AlphaBrain/tree/dev/benchmarks/LIBERO).

## Reproduction

```bash
bash scripts/run_world_model/train/run_cosmos_policy.sh
```

The shipped `framework_config.yaml` is the exact configuration used for
this checkpoint. Expect multi-day training on 8 × A800 80 GB for the
full 40 000-step schedule; the 10 k release is a useful mid-training
snapshot with good LIBERO-Goal performance and a manageable download
footprint.

## Notes

- **Dual checkpoint format.** `model.safetensors` is the canonical
  framework-loadable checkpoint used by `server_policy.py`;
  `cosmos_dit.pt` is the raw DiT tensor file kept for resume /
  selective loading. Most users only need `model.safetensors`.
- **T5 embeddings are shipped** so users do not need to re-run the T5
  text encoder at inference time — load the `.pkl` directly.

## License

MIT — see the [parent repository](https://github.com/AlphaBrainGroup/AlphaBrain).

## Citation

```bibtex
@misc{alphabrain2026,
  title  = {AlphaBrain: A Modular Open-Source Framework for Embodied Intelligence Research},
  author = {AlphaBrain Team},
  year   = {2026},
  url    = {https://github.com/AlphaBrainGroup/AlphaBrain}
}
```
