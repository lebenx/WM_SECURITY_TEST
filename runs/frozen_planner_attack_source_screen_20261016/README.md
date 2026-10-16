---
language:
- en
tags:
- reinforcement-learning
- world-model
- robustness
- adversarial-evaluation
- checkpoint
---

# World-model attack research checkpoints

Frozen checkpoints and supporting configurations for the research project
**Misranking Imagined Futures: Test-Time Attacks on Frozen World-Model Planners**.
This collection preserves the available checkpoints used by the local research
implementations, including upstream pretrained models and locally trained models.

| Directory | Included artifacts |
|---|---|
| `le-wm/` | Cube, Push-T, Reacher, and Two-Rooms model weights and architecture configurations; residual HGD and paper HGD defense checkpoints |
| `dino_wm/` | Released Push-T and Wall/Two-Rooms model checkpoints and training configurations |
| `planet/` | Cartpole Swingup, Reacher Easy, and Cheetah Run native TensorFlow checkpoints, frozen PyTorch inference exports, and TensorFlow parity fixtures |
| `tdmpc2/` | The four RGB checkpoints selected by the research reference manifests: Finger Spin at 1,000,000 steps; Cup Catch and Cheetah Run at 1,500,000; Cartpole Balance at 250,000 |
| `dreamerv3/` | Cheetah Run and Reacher Easy training configurations and immutable checkpoint directories |

These checkpoints have different observation preprocessing, architectures, task
interfaces, and Python dependencies. Use the corresponding research model loader;
they are not interchangeable Transformers models. The PlaNet PyTorch exports
contain 32 inference tensors and require the research implementation of PlaNet.
The DINO-WM checkpoints also require their DINOv2 encoder dependency and native
task data. LeWM and defense evaluations require their datasets and simulators.

## Download and integrity

Download the repository using the Hugging Face CLI:

```bash
hf download mohendy/world-model-attack-checkpoints --local-dir ./checkpoints
python ./checkpoints/verify_download.py
```

If the repository is private, authenticate with `hf auth login` first. For a
durable experiment record, pass `--revision` with the repository commit SHA
returned by the upload, instead of relying on the moving `main` branch. A partial
download can be checked with `python verify_download.py --model planet` from the
downloaded directory.

`manifest.json` records the SHA-256 and size of every checkpoint/configuration,
plus its original relative location in the research workspace. Its
`source_relative_path` field provides the destination mapping for the code
release's artifact layout. Binary model checkpoints preserve their source bytes.
Configuration output paths are sanitized and relative. HGD checkpoints are
re-exported with sanitized metadata; their `state_dict` tensors were verified to
be exactly equal to the originals. Both original and released hashes are recorded.

Copy or link artifacts into the code release using that mapping, and set explicit
checkpoint/data paths for your machine. Run directories for Dreamer must include
their `config.yaml` and chosen `ckpt/<checkpoint-id>/` directory. Select the
immutable directory explicitly instead of a training run's `latest` pointer.

Model-specific `runtime.json` files record the installed Python and package
versions from the research environment. They are provenance records, not a
guarantee that all packages resolve unchanged on another platform. The code
release maintains environment installation instructions and validation results.

## Scope and provenance

Datasets, experiment logs, videos, and per-episode adversarial perturbations are
not included. Checkpoint availability alone does not reproduce the paper tables:
the task datasets, episode identities, seed schedules, preprocessing, planner
configuration, and matched clean reference results must also agree.

Upstream projects and weight sources:

- [LeWM code](https://github.com/lucas-maes/le-wm) and
  [checkpoint collection](https://huggingface.co/collections/quentinll/lewm).
- [DINO-WM code](https://github.com/gaoyuezhou/dino_wm) and its
  [released artifacts](https://osf.io/bmw48/).
- [PlaNet code](https://github.com/google-research/planet); the local inference
  exports record source artifact revision
  `b26a7fe027c018f62b3178c8e126371fbdee33e5` in `attack_export.json`.
- [TD-MPC2 code](https://github.com/nicklashansen/tdmpc2).
- [DreamerV3 code](https://github.com/danijar/dreamerv3).

See [LICENSES.md](LICENSES.md) for the accompanying upstream notices. Original
upstream work and local research additions should be credited separately.
