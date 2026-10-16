# Goal-bus 研究交付审计

> **历史材料，不能作为当前投稿或完成证明（2026-10-15）。** 下文保留旧协议和旧结论；部分 codec、原生决策路径、Clean 判断及不变性前提已被后续审计修正。当前证据见[研究进度](research_progress.html)与[现行英文工作稿](world_model_decision_audit_working.pdf)。完整安全机制、有效防御、独立泛化和论文竞争力尚未建立。原样版本保存在 [historical_goal_bus_package_20261015/](historical_goal_bus_package_20261015/)。历史复现命令不代表当前实验建议。

## 已满足

| 要求 | 当前证据 | 状态 |
| --- | --- | --- |
| 一句话、相对宽的 claim | 指令文本可经共享 action-token bus 改写固定 state/action world forecast；edge split 恢复 invariance 并保留 policy/action sensitivity | 已满足 |
| 多个独立载体 | WorldAgen + AlphaBrain CosmosPolicy | 已满足 |
| 强对照 | WorldAgen empty-goal、9-offset/270-row sweep、同 cardinality policy-only edge cut、同 episode alternate-action control（30 窗口）、AlphaBrain world-text-only cut、fixed-action cut、alternate-action control、sampler seeds 2/3、disjoint-episode learned goal-free context | 已满足（预测层） |
| 效应量对照 | AlphaBrain 同三 episode 的 goal-swap world effect / alternate-action effect = 0.819；明确标为描述性三 episode 比较 | 已满足（量级证据） |
| 中间机制定位 | direct world-text cut 不够；action/value carrier cut 才恢复 exact invariance | 已满足 |
| 中介位置的独立证据 | WorldAgen 21 窗口（7 episode × frame 30/60/90）layer-wise activation patch；layer 4/8 world residual 0.2062/0.1998，policy residual 0.9555/0.9643，21/21 向记录 goal 移动 | 已满足（内部机制诊断） |
| 核心任务覆盖 | LIBERO spatial/object/goal 轨迹；WorldAgen 30×9 合法指令；WorldAgen 六个记录 task 语义组；AlphaBrain 五个 task-A 语义组 | 已满足（forecast-level） |
| 可复现交付 | 中英文稿、11 页 PDF、复现清单、机器 manifest、SVG/PDF figures、动作对照脚本与主结果 JSON | 已满足 |
| novelty 边界 | 已对照 WorldVLA、LingBot-VA、AffordanceWAM、FFDC-WAM，明确当前机制差异 | 已满足 |
| 决策级固定候选测试 | WorldAgen 独立 forecast-driven planner；6 tasks、8 条冻结真实动作轨迹、Clean/Goal-Swap/Bus-Cut/Control-Cut；真实 LIBERO EEF 回放 | 已满足（独立 planner 范围） |
| 决策级 seed 与重复复核 | seed 17/23 共 12 个 task-seed 配对；每条件 3 次 fresh replay；Control-Cut 分数恒等 | 已满足（描述性统计） |
| 修复层位消融 | 6-task activation patch（primary-view/first-chunk/step-3 Forecast Error v2）；layer 4/8 降低 forecast residual，但 Clean selected action 仅 2/6 恢复 | 已满足（证明预测修复不等于决策修复） |
| planner 校准与预测对齐 | 严格 primary-view/first-future-chunk/step-3 protocol；三个 seed、18 个 task-seed 配对的 reference-scalar Clean regret 0.0087 m 低于 random 0.0131 m；30-step consequence 复核的 Clean regret 0.0758 m 低于 random 0.1345 m；旧 global-minmax 未通过该校准门 | 已满足（决策指标边界已显式化） |
| 跨任务 value probe 反证 | 固定 alpha、只用训练 task Clean forecast 拟合，在两个互补 held-out task split 上复核；Goal-Swap 改变选择，但 Clean 校准和 EEF 方向不跨 split 稳定 | 已记录（支持选择扰动，不支持普遍效用损害） |
| 合法 Goal dose-response | 6 个任务遍历 offset 1--9，seed 17/31/37 共 18 个 task-seed 对、162 个 counterfactual；step-3 选择、30-step 真实回放；所有 offset 的跨任务 EEF 均值为负，pairwise ranking flip 为 0.448--0.556；Forecast Error 与 flip/EEF/regret 的 pooled Spearman 仅 0.099/-0.080/-0.080；Bus-Cut 12/18 选到候选真实最佳但 0/18 恢复 Clean action | 已记录（支持跨任务 planner 扰动与部分 repair，说明 Forecast Error 不能直接替代决策风险，仍不支持原生任务损害或完整恢复） |
| 物理目标 Goal-invariance audit | 同一初始状态、冻结候选和前三步真实回放 target；162 条 counterfactual 的 MSE 重算误差为 0；Goal-Swap physical-target MSE 差均值 −0.0121、task-cluster CI [−0.0434, 0.0181]，offset 间混合正负；divergence 与物理误差变化 Spearman 0.117 | 已记录（证明固定物理未来下的预测污染边界；不把 latent divergence 当成物理错误） |
| 同场景 Goal crossing | LIBERO-Goal 10 个合法 Goal、3 个共享初始 simulator state、4 组固定动作，共 120 次 fresh replay；初始状态和三步未来观测 120/120 exact；native 非 Clean Goal effect 平均 MSE 0.4913；Bus-Cut 跨 Goal 方差最大 6.2e-15，动作敏感性保留；均匀非 Clean Goal 的 state-cluster target-MSE 差为 +0.0965，三状态均为正，bootstrap CI [+0.0615,+0.1260]；Bus-Cut baseline MSE 上移 +0.0884 | 已记录（环境端条件独立控制通过；Clean Goal 不是无偏 physical-target comparator 的描述性证据，但不排除模型训练中的部分可观测性，也不支持每个 Goal 都使预测变差；修复存在 fidelity cost） |
| 同场景 empty-Goal comparator | 完整 50 个初态、4 组动作，共 200/200 exact target replay；empty−Clean target-MSE 总体 +0.0744，50/50 初态为正，state-cluster CI [+0.0665,+0.0824]；empty/Clean forecast MSE 0.2139 | 已记录（prompt-independent 诊断；Clean 携带稳定任务先验，但不是训练过的 goal-free dynamics head） |
| AlphaBrain future physical-output slots | 21 个固定 image/state/recorded-action 窗口、7 个 episode；Goal-Swap 改变 future proprio 槽 21/21，绝对 latent delta 0.0607，episode-cluster CI [0.0376, 0.0970]；offset 1/5/9 扩展为 63 窗口，各自 21/21 非零（均值 0.0512/0.0607/0.0728）；fixed-action cut 三个输出槽均 exact zero；proprio target-MSE 差 +0.0084、wrist 差 +0.0248；cut 相对 Clean 的 wrist/primary fidelity cost 为 +0.0918/+0.0416 | 已记录（直接加强物理输出槽的预测污染证据；invariance repair 有 fidelity tradeoff，错误方向仍不能普遍化） |
| AlphaBrain forecast-driven planner 可行性门 | task 0/seed 17、8 条冻结候选、future-primary CLIP 评分、30-step 真实 LIBERO replay；Clean native regret 0.0621 m，Goal-Swap regret 0.4228 m；fixed-action Bus-Cut 的 Clean/Swap 分数 exact equal 但两者均选中候选 6、regret 0.4228 m | 已记录为负结果（证明第二载体可进入独立 planner 决策回路，但未恢复 Clean decision；不支持跨载体任务损害或普遍 repair） |
| AlphaBrain 三 task planner 扩展 | task 0/2/4、seed 17、每 task 8 候选、16-step forecast、30-step replay；Goal-Swap 改变 2/3 个选中动作；Clean/Swap/Bus-Cut regret 均值 0.1836/0.2251/0.2796 m；Bus-Cut Clean-action recovery 1/3；task 4 效用方向反转 | 已记录（跨载体 planner reachability 可复现，但效用方向 task-dependent，不支持普遍损害或完整 repair） |
| AlphaBrain 六 task planner 扩展 | task 0/2/4/6/7/9、seed 17、每 task 8 候选、16-step forecast、30-step replay；Goal-Swap 改变 5/6 个选中动作；Clean/Swap/Bus-Cut regret 均值 0.1646/0.1911/0.2433 m；Bus-Cut Clean-action recovery 2/6；task 4/6 改善、0/7/9 变差 | 已记录为跨载体主 pilot（支持 planner reachability，明确不支持普遍 utility harm 或完整 decision repair） |
| AlphaBrain 六 task × 三显式初态复放 | `init_index=0/1/2` 共 18 个 task-state rows；Goal-Swap 改变 15/18 个 selected action；Clean/Swap/Bus-Cut regret 0.1511/0.1885/0.2607 m；Swap−Clean +0.0374 m；初态聚类 CI [+0.0098,+0.0758]、task 聚类 CI [−0.1086,+0.1931]；Bus-Cut recovery 3/18 | 已记录（跨状态描述性 reachability；task 聚类区间跨 0，不支持普遍损害或完整 repair） |
| 语义表面与同义改写 control | 表面变体 3 个窗口；同义改写 3 个窗口×2 个 paraphrase，native world relative L2 在 6/6 非零（均值 0.00844、最大 0.01131），Bus-Cut 6/6 exact zero，policy 仍文本敏感 | 已记录（文本条件控制，不写成物理攻击） |
| RNG-persisted state-target audit | 6 个任务、3 个 seed 的 144 个候选双 fresh replay，共 288 个 replay episode；初始 hash、qpos/qvel/contact、前三步图像 hash、EEF/进度均 144/144 exact | 已记录（验证完整 task-seed 的 state-target 协议；不增加 Goal 效应量） |
| 决策修复路径 | 固定 Clean/Bus-Cut rank 混合权重；Bus-Cut 权重 0.25 的 regret 0.0291 m，权重 0.75 的 Clean-action recovery 0.778；不读取 replay utility | 已记录（显式展示 utility/recovery Pareto；需要 Clean reference branch，不能写成原生修复） |
| 候选数量敏感性 | 同一已回放候选的 8 条主集合与固定 linspace-4 子集；offset 5 regret 0.1481/0.1550 m，4-candidate ranking flip 0.407--0.583，Control-Cut 恒等 | 已记录（稳健性复核；4-candidate 不是重新采样协议） |
| Forecast-only attacker | 只用模型 score/ranking 在 offset 1--9 中选最大 pairwise flip；EEF −0.0995 m、regret 0.1753 m，相对均匀 offset 期望的 regret 差 +0.0278 m，task-cluster CI [−0.0230, 0.0786] m；其他 divergence selector 反而改善 | 已记录（明确分离 Forecast Divergence 与攻击成功，不支持稳定的最大 divergence 攻击定律） |
| Latent-level divergence attacker | 只用保存的 `worlds_step3` latent MSE 选最大 Goal divergence；平均 MSE selector 的 regret 差 +0.0235 m，task-cluster CI [−0.0558, 0.1111] m；最大候选 MSE selector 差 +0.0262 m | 已记录（张量级反证：方向相容但不构成可靠攻击定律） |
| Leave-one-seed-out forecast-only attacker | 每个 held-out seed 只用另外两个 seed 的 forecast 特征选一个全局 offset；latent/raw-z 选择器均选 offset 9，regret 差 −0.0338 m、CI [−0.1053, 0.0534] m；pairwise-flip 差 −0.0245 m、CI [−0.0985, 0.0446] m | 已记录（跨 seed 外推不支持 divergence 作为稳定攻击目标） |

## 尚未满足

| 要求 | 缺口 | 不能过度声称的结论 |
| --- | --- | --- |
| 任务级 ICLR 基线 | 当前释放的 WorldAgen/AlphaBrain checkpoint 在 LIBERO clean-task gate 均失败 | 不能声称 task-harm、闭环攻击成功率或策略性能保持 |
| 完整强 goal-free dynamics baseline | 已加入冻结 DiT 上训练的共享 goal-free context（3 train / 4 held-out，seed 17/23），并用 frame 30/60/90 的 9 train / 12 held-out 窗口做 seed-29 扩展；它仍不是匹配架构的完整 dynamics head | 不能把 learned context 写成完整训练过的 no-goal dynamics head |
| 训练资源可复核性 | 已审计发布权重、官方训练入口和本地数据路径；权重在，但 `success_only`/`all_episodes` 兼容轨迹不存在 | 不能声称已完成匹配架构的完整 fine-tune；见 [resource audit](../runs/alphabrain_training_resource_audit_v0/summary.json) |
| 任务成功率与 native repair utility | 官方一致预处理的 AlphaBrain 608-step spatial pilot，以及匹配 `dataset_mix: libero_goal` 的 task0–2、304-step/task 聚合 pilot，均为 native/cut 0/3；短 horizon sparse reward 全为 0；两种独立 planner scorer 的 EEF 效用方向相反 | 不能声称原生 policy 已被攻击、修复必然提升或损害任务性能 |

## 当前主张范围

论文可以稳健声称：在两个发布的 instruction-conditioned world/action 载体上，合法 task text 能通过共享 action-token carrier 改变固定物理 state 与已执行 action 下的 world forecast；matched physical-target audit 证明这种变化发生在同一个固定物理未来之上，但其 target-error 方向跨 Goal offset 混合，不能把 latent divergence 直接称为物理预测错误；匹配的冻结权重 edge split 能恢复 goal-invariant world prediction，同时保留 policy 与 action sensitivity。当经过真实后果校准的独立 forecast-driven planner 消费 WorldAgen forecast 时，Goal-Swap 会改变候选 ranking 和 selected action，并在部分合法 offset 上改变真实 EEF 后果；但 utility 方向依赖 scorer，Bus-Cut 与 activation patch 都不能保证 Clean decision recovery。原生 policy 的决策级攻击和任务级损害仍未建立。

## 首要实验门：物理 Goal-invariance 的 estimand

在攻击成功率之前，必须先确认接口语义允许使用的条件独立性：固定完整物理状态 (S_t) 与已执行动作 (A_{t:t+H})，物理未来 (Y^{phys}) 应满足 (Y^{phys}\perp Goal\mid S_t,A_{t:t+H})。Goal 可以改变 policy、value、reward、termination 和 goal-relative semantic 输出。若 Goal 是隐藏状态的线索，或改变 simulator reset，这个 estimand 就不成立。当前 120/120 同场景 replay 通过了环境端控制；均匀抽取非 Clean Goal 的 state-cluster target-MSE 差为 +0.0965，三状态均为正，但单元格方向仍混合；Bus-Cut 恢复不变性却带来 +0.0884 baseline MSE shift。模型 partial-observability sufficiency 和 state/contact/proprio target 仍是下一道 validity check；Forecast Divergence 不能替代物理 target error。

## 下一道真正的实验门

若要把当前独立 planner 证据升级为原生任务级结论，仍需要以下任一条件：

1. 一个能在同一 LIBERO suite 完成 clean task 的 task-capable checkpoint；或
2. 一个与 AlphaBrain 匹配、训练过的 goal-free dynamics head。

在该门通过前，报告应保持 Go（预测/机制/独立 planner 决策扰动）与 No-Go（原生任务损害/普遍效用损害）的边界。
