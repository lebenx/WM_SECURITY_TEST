# 共享动作 token 因果总线：复现与证据索引

> **历史材料，不能作为当前投稿或完成证明（2026-10-15）。** 下文保留旧协议和旧结论；部分 codec、原生决策路径、Clean 判断及不变性前提已被后续审计修正。当前证据见[研究进度](research_progress.html)与[现行英文工作稿](world_model_decision_audit_working.pdf)。完整安全机制、有效防御、独立泛化和论文竞争力尚未建立。原样版本保存在 [historical_goal_bus_package_20261015/](historical_goal_bus_package_20261015/)。历史复现命令不代表当前实验建议。

本清单对应 `goal_bus_paper_draft_cn.md` 和 [English submission draft](goal_bus_paper_draft_en.md) 的主结论。已编译的 [English PDF](goal_bus_paper_draft_en.pdf) 可直接交给合作者审阅。所有命令在仓库根目录 `/home/xhj/WM_Security` 执行；实验使用本地 checkpoint、固定真实输入和记录动作，不调用外部 API。

## 结论—证据映射

| 要验证的命题 | 最小证据 | 判据 | 结果 |
| --- | --- | --- | --- |
| 合法 goal 能改变固定输入下的 world forecast | [WorldAgen 30 窗口](../runs/worldagen_goal_bus_replicate_v2/summary.json) | native world 差异非零 | mean rel. L2 0.2999 |
| WorldAgen 修复保留真实动作条件 | [WorldAgen alternate-action control](../runs/worldagen_goal_bus_action_control_v1/summary.json) | 固定图像/state/goal，只替换同 episode action chunk，cut 后仍非零 | 30/30 非零；cut mean rel. L2 0.1655，CI [0.1469, 0.1855] |
| 直接切断 goal→input-action 边可恢复 world invariance | WorldAgen 同一结果 | edge-cut world 全部为零 | 30/30 |
| 修复是 carrier-specific 而非删边数量效应 | [matched edge-cut control](../runs/worldagen_goal_bus_edge_cut_controls_v0/summary.json) | 同样 9 边但只切 policy rows 后 world 仍敏感 | policy-only world mean 0.2999，30/30 非零 |
| 该路径不是简单 world-row 文本泄漏 | [AlphaBrain world-text-only ablation](../runs/alphabrain_cosmos_policy_real_input_sampler_refined_cut_v0/summary.json) | 仅切 world text 后仍有差异 | mean 0.04235，3/3 非零 |
| action carrier 是必要中间节点 | [AlphaBrain expanded matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json) | fixed-action cut 精确恢复 | 21/21 为零 |
| 修复没有抹去物理动作条件 | [action control](../runs/alphabrain_cosmos_policy_real_input_sampler_action_control_v0/summary.json) | alternate action 仍改变 world | repaired mean 0.08715 |
| goal effect 的实际量级 | [matched goal/action effect](../runs/alphabrain_goal_vs_action_effect_v0/summary.json) | 与同 episode alternate-action control 比较 | goal/action world-effect ratio 0.819；描述性三 episode 对照 |
| policy 分支仍可保持 goal 条件 | [WorldAgen 同一结果](../runs/worldagen_goal_bus_replicate_v2/summary.json) | cut 后 policy 差异非零 | rel. L2 0.17596 |
| forecast repair cost 不是普适常数 | [noise seed 2](../runs/alphabrain_cosmos_policy_real_input_sampler_noise2_v0/summary.json) 与 [noise seed 3](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/aggregate.json) | invariance 保留但 MSE delta 可变号 | deltas −0.00079 与 −0.03820 |
| zero context 不是唯一 no-goal 对照 | [learned goal-free context](../runs/alphabrain_goal_free_context_v1/summary.json) 与 [seed-23 replicate](../runs/alphabrain_goal_free_context_v2/summary.json) | disjoint-episode 学习后 held-out one-step MSE 改善 | 0.58519 → 0.46215/0.46219 |
| learned context 不依赖单一时间点 | [multi-timepoint check](../runs/alphabrain_goal_free_context_v3/summary.json) | frame 30/60/90，disjoint episode，held-out MSE 改善 | 0.60197 → 0.47852；delta CI [−0.18414, −0.01299] |
| 中间 action carrier 的 activation mediation | [WorldAgen activation patch](../runs/worldagen_goal_bus_activation_patch_v1/summary.json) | 中层 patch 后 world residual 降低且 policy residual 保持接近 1 | layer 4/8 world 0.2062/0.1998；policy 0.9555/0.9643；21/21 向记录 goal 移动 |
| WorldAgen 核心任务语义覆盖 | [WorldAgen task groups](../runs/worldagen_goal_bus_task_groups_v0/summary.json) | 按记录 task text 分层后每组仍 native 非零、cut exact zero | 6 组；native 全部非零、cut 全部为零 |

## 推荐复现顺序

先做不需要 GPU 的代码和结果一致性检查：

```bash
PYTHONPATH=src pytest -q tests
python -m py_compile \
  scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix.py \
  scripts/analyze_alphabrain_forecast_matrix.py \
  scripts/analyze_alphabrain_refined_cut.py \
  scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast.py
python - <<'PY'
import json
from pathlib import Path
for p in [
    'configs/world_model_goal_bus_security.json',
    'configs/research_roadmap.json',
    'runs/goal_bus_main_results_v0/summary.json',
    'runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json',
]:
    json.loads(Path(p).read_text())
    print('ok', p)
PY
```

WorldAgen 的 activation mediation 是一个短的冻结模型诊断，运行 7 个 episode 的 21 个固定真实窗口，不训练也不执行环境 rollout：

```bash
.venv-worldagen/bin/python scripts/pilot_worldagen_goal_bus_activation_patch.py
python scripts/plot_worldagen_activation_mediation.py
```

结果见 [逐层聚合](../runs/worldagen_goal_bus_activation_patch_v1/summary.json)、[SVG 图](figures/worldagen_activation_mediation.svg) 和 [PDF 图](figures/worldagen_activation_mediation.pdf)。该内部 patch 只用于机制定位，不能当作攻击者可用的输入操作。

WorldAgen 的固定真实窗口结果和 empty-goal comparator 位于：

```text
runs/worldagen_goal_bus_replicate_v2/summary.json
```

WorldAgen 的同 episode alternate-action control 使用相同图像、state 和 instruction，只替换六步真实 action chunk：

```bash
WORLDAGEN_DATA_ROOTS=data/worldagen_goal_probe,data/worldagen_goal_probe_extra \
PYTHONPATH=vendor/LIBERO:vendor/WorldAgen \
.venv-worldagen/bin/python scripts/pilot_worldagen_goal_bus_action_control.py
```

结果见 [动作对照聚合](../runs/worldagen_goal_bus_action_control_v1/summary.json)。

AlphaBrain 的扩展 held-out matrix 可用以下命令重跑；该实验会加载公开 VAE、DiT 和本地 episode 数据，预计需要 GPU：

```bash
ALPHABRAIN_FORECAST_MATRIX_SEED=1 \
.venv-a2world/bin/python \
scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix.py

.venv-a2world/bin/python \
scripts/analyze_alphabrain_forecast_matrix.py
```

bounded learned goal-free context 对照使用冻结 DiT/VAE，只优化共享 context；它是短时 one-step forecast comparator，不是完整 dynamics-head 训练：

```bash
.venv-worldagen/bin/python scripts/train_alphabrain_goal_free_context.py
```

多时间点短复现（仍然只训练共享 context，不是完整 dynamics head）：

```bash
ALPHABRAIN_GOALFREE_OUT=runs/alphabrain_goal_free_context_v3/summary.json \\
ALPHABRAIN_GOALFREE_FRAMES=30,60,90 \\
ALPHABRAIN_GOALFREE_LR=0.001 \\
ALPHABRAIN_GOALFREE_STEPS=8 \\
ALPHABRAIN_GOALFREE_SEED=29 \\
.venv-worldagen/bin/python scripts/train_alphabrain_goal_free_context.py
```

训练资源审计可复现为：

```bash
python scripts/audit_alphabrain_training_resources.py
```

该审计确认本地有 AlphaBrain 2B 推理权重、统计文件和官方训练入口，但没有 `LIBERO-Cosmos-Policy/success_only` 与 `all_episodes` 轨迹目录；因此当前 package 不把 learned context 误写成完整 fine-tuned goal-free dynamics head。结果见 [resource audit](../runs/alphabrain_training_resource_audit_v0/summary.json)。

三次对照的图由 [绘图脚本](../scripts/plot_goal_free_context_baseline.py) 生成，输出为 [SVG](figures/goal_free_context_baseline.svg) 和 [PDF](figures/goal_free_context_baseline.pdf)。

当前 seed-1 matrix 固定七个 episode、三个 frame index（每个 episode 三个窗口），使用同一 target frame、同一记录动作 chunk、同一噪声输入分别评估 native 与 fixed-action cut。独立 seed-3 小矩阵覆盖 episode 3、4、5 的 frame 30；两者都固定 native/cut 输入，只改变 sampler noise seed。结果和聚类 bootstrap 汇总分别在：

- [raw 21-window summary](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/summary.json)
- [aggregate](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json)
- [tradeoff figure](figures/alphabrain_forecast_tradeoff.svg)
- [seed-3 aggregate](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/aggregate.json)

## 读数规范

- `native world` 是交换合法 goal 后的 world 输出差异；输入图像、state、记录动作、时间步和噪声保持相同。
- `fixed-action world cut` 只切除 condition 对 action/value carrier 以及 world rows 的相关文本通路，保留 world rows 对记录 action carrier 的读取；因此它不是“删除所有条件”的 full cut。
- `world-text-only` 只切 world rows 的直接文本入口，用来检验是否存在中间 carrier；它不能作为修复结果。
- `alternate-action control` 改变同一 episode 的记录动作 chunk，检验修复后 world branch 是否仍响应物理控制条件。
- 重复窗口按物理 episode 聚类计算置信区间；不能把同一 episode 的多个 task offset 当作独立世界。
- seed 1 的正 forecast repair delta 是条件性结果；seed 2 和 seed 3 的 MSE delta 均为负，因此不能写成“修复必然牺牲预测精度”。

## 任务级边界

[三任务族短 pilot](../runs/alphabrain_cosmos_libero_short_control_fixed_action_v1/aggregate_summary.json)、旧版 [608-step spatial pilot](../runs/alphabrain_cosmos_libero_full_control_v1/aggregate_summary.json)、官方一致预处理的 [修正版 608-step spatial pilot](../runs/alphabrain_cosmos_libero_full_control_v2/spatial_600.json) 和与 checkpoint 配置匹配的 [304-step LIBERO-Goal task0–2 aggregate](../runs/alphabrain_cosmos_libero_goal_control_v0/aggregate_summary.json) 都没有通过 clean-task gate。它们只能作为 negative control，不能支持任务损害率、闭环攻击成功率或策略性能保持的强结论。learned shared context 只补充预测层 no-goal 对照，仍不能替代完整训练的 goal-free dynamics head。要继续扩展任务级实验，需要先提供能完成同一 LIBERO 任务的 clean checkpoint，以及匹配架构的训练过的 goal-free dynamics head。

## 当前 checkpoint 的可用范围

| 本地载体 | 是否能直接补强当前 claim | 原因 |
| --- | --- | --- |
| WorldAgen `libero-38.pth` | 已纳入 | 有 goal-conditioned LIBERO world/policy 路径；30 窗口因果修复通过，但 600-step clean task gate 为 0/3。 |
| AlphaBrain CosmosPolicy | 已纳入 | 有真实图像/state/recorded-action 与 T5 task conditioning；21 窗口矩阵和两个独立 sampler seed 通过。 |
| A2World `a2world-pretrained.pt` | 暂不纳入主 claim | 现有短门只测 action suffix 的 carrier 传播，没有合法 task-instruction 条件，也没有可比的 goal-bus intervention。 |
| FACT RoboTwin transformer | 暂不纳入主 claim | 本地 checkpoint 面向 RoboTwin world-action，不是当前 LIBERO 指令条件载体；直接并入会改变数据、任务和威胁模型。 |
| TD-MPC2 mt30-48M | 暂不纳入主 claim | 原生载体没有当前 goal-to-world action-token 总线；已有筛选结果不支持把它当作同一机制的强 baseline。 |

这份范围表说明了为什么当前论文只声称两个独立载体的 forecast-level 机制，并没有把不具备相同指令条件接口的本地 checkpoint 强行包装成第三个复现。
