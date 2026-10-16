DINO CPU oracle-feature engineering calibration

This directory contains the independently seeded seed1000 engineering pilot.
The separately fixed confirmatory seeds are1001..1030. The pilot is excluded
from confirmatory statistics. No outcome-dependent bank, score, metric or
root changes are permitted.

Implementation: scripts/run_decision_leverage_dino_cpu.py
Protocol: configs/decision_leverage_dino_e1.json
Each root: separate expert goal100 controls, one native warmup, five frozen
candidate sequences replayed for30 controls, eight encoder calls and one H6
model unroll. The original forecast is committed before candidate replays.
Offline interpolation lambda=0,.25,.5,.75,1 is a diagnostic using future
information; it is not a deployment defense or native CEM policy. All action
selections use the terminal30-control native objective. The15-control utility
is the prefix of the H30-selected action, not a separate H15 oracle.

Observed pilot: exit0,30.043452715 seconds/root;35.551480169 seconds including
one model construction and verification. PeakRSS1564172 KiB. Both existing
checkpoints loaded once; CUDA uninitialized; zero network/API/training.
All initial-state restoration, fixed target/context, candidate bank, native
FP32 score, truth encoding, and commitment checks passed.

Original and actual-future score selectcandidate1. Terminalreward6.3986411095
under both, paired effect0. All five interpolation values also select1.
This is a null engineering pilot; it is neither confirmatory evidence for
P1 nor permission to select a different candidate bank or metric.

Verification without model/encoder/environment/checkpoint execution:
.venv-jepa-native/bin/python scripts/run_decision_leverage_dino_cpu.py --phase calibration --out runs/decision_leverage_dino_cpu_pilot --verify-only

Reinvocation without--verify-only also verifies and skips already completed
roots. Incomplete or failed roots are preserved and rejected, not rerun.
