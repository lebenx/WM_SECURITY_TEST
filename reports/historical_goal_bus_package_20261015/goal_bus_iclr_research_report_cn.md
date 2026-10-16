# Goal-Bus：从预测污染到控制平面风险

## 一句话核心 Claim

**在共享 action-token 的目标条件世界模型中，任务文本不仅是语义条件，还能作为控制平面消息写入动作载体并改写未来预测；这条 Goal-Bus 可以因果地改变预测驱动的动作排序和真实轨迹，而影响方向及切断后的效用恢复取决于下游 planner，因此预测不变性不能直接等同于决策安全。**

这个表述同时覆盖预测、机制和决策三个层面，并明确区分“预测安全不变量恢复”和“真实决策安全”。

按 ICLR 审稿视角的主观 novelty 估计：若把“Goal-Bus 控制平面 + Goal-invariance/Decision-invariance 分离 + repair-induced baseline shift”作为统一主线，目前约 **6.5–7/10**；若只保留 forecast leakage，则会退回常规的 conditioning leakage 叙事。

## 首要前提：为什么物理未来应当对 Goal 条件独立

Goal-Bus 在进入“攻击成功率”之前，必须先回答一个更基础的 estimand 问题。把环境状态记为 (S_t)，已执行动作序列记为 (A_{t:t+H})，合法任务指令记为 (G)，物理目标记为 (Y^{phys}_{t+1:t+H})。在 Goal 只描述任务偏好、奖励或终止条件的接口中，因果结构应满足

\[
G\rightarrow A_t,\qquad (S_t,A_{t:t+H})\rightarrow Y^{phys}_{t+1:t+H},\qquad
G\rightarrow (R,\mathrm{terminal},Y^{semantic}),
\]

因此在固定完整状态并执行同一动作时，正确的物理不变量是

\[
Y^{phys}_{t+1:t+H}\perp G\mid S_t,A_{t:t+H},
\]

或者等价地，在环境上施加 `do(A=a)` 后，Goal 只能改变 policy、value、reward、termination 和 goal-relative semantic 输出，不能改变真实物理转移。这个条件独立性属于环境动力学的定义，不是“模型看到 Goal 后应该输出相同字符串”的经验偏好。

这里有一个必须保留的反证边界：如果模型只看到部分观测 (O_t)，而 Goal 恰好是未观测状态 (S_t) 的信息源，那么 (P(Y^{phys}\mid O_t,A,G)) 可以合法地依赖 Goal；如果不同 Goal 改变 reset 或场景定义，同样不能使用上述 estimand。因此“Goal 不应改变物理预测”只有在完整状态、执行动作、物理 target 和 Goal 的数据语义都被固定后才成立。

现有同场景 crossing 已经验证了环境端而不是模型端的第一道门：10 个合法 Goal、3 个完整初始 simulator state、4 组固定动作的 120 次 fresh replay 中，sim/qpos/qvel/contact 与三步未来观测均 120/120 exact。随后 WorldAgen 在完全相同的物理输入上仍产生非零 Goal effect，而 Bus-Cut 恢复 Goal-invariance；这支持“模型预测被 Goal 污染”的边界定位。它还没有证明发布模型的观测 (O_t) 足以排除 hidden-state 解释，也没有证明 divergence 一定增加 physical-target error；后者在 target-MSE audit 中是混合符号。

所以当前下一道门应按顺序检查：先扩大并复核同状态/同动作的环境不变量，再用 state/contact/proprio 等物理 target 做 Goal permutation 与 hidden-state sufficiency 对照，最后才把 forecast 改变送入 planner。任何只报告 Forecast Divergence 或 selected-action flip、却没有这条物理 target 对照的结果，都不能单独称为攻击成功。state-cluster 与 Goal-level target-error 图见 [`same-scene target figure`](figures/worldagen_same_scene_goal_target_delta.svg)。

## 为什么这个 Idea 有新意

现有目标条件 world model 工作通常把 instruction 当作普通条件变量，安全分析则常把 forecast divergence 直接当成控制风险。本文提出并实证区分两个性质：

1. **Goal-invariance**：固定真实状态和已执行动作时，合法 Goal 不应改变物理未来预测。 这里有明确的可证伪前提：如果 Goal 本身编码了未观测物理状态，或改变了 simulator reset，那么该条件独立性就不是正确 estimand。当前审计固定 simulator state 与已执行动作序列，把 policy/value/reward/termination 及 goal-relative semantic 输出排除在物理 target 外，并把 partial observability 作为开放的反证条件。
2. **Decision invariance**：当下游 planner 使用 forecast 选动作时，Goal 改变不应改变动作排序或真实后果。

两者不是同一个性质。因果切断可以让 World Forecast 对 Goal 不敏感，却因为改变了共享表征而产生 planner baseline shift。这个“预测修复不等于决策修复”的分离，是比单纯报告 forecast leakage 更强、也更可检验的安全框架。

## 预测级证据

载体是公开 WorldAgen `libero-38.pth`，输入来自真实 LIBERO 窗口。固定图像、state、已执行动作和模型参数，只切换合法 task text：

- 30 个窗口中 native Goal-Swap 的 world relative L2 均值约 **0.2999**，所有窗口非零。
- 切断九条 Goal→input-action carrier 边后，world 输出在 **30/30** 窗口精确不变。
- 同样删除 9 条 policy-only 边的 matched control 仍保持 world goal sensitivity，排除了“删边数量”解释。
- same-episode alternate-action control 在 bus cut 后仍产生非零 world 差异，说明修复没有冻结真实物理动作条件。

这些结果只支持预测级 Goal leakage 与其路径定位，不单独支持攻击成功。

### 同场景 Goal crossing：先验证环境端条件独立

为排除“不同 task 的 BDDL 或隐藏场景状态本身改变了物理未来”的解释，新增了同场景 crossing。LIBERO-Goal 的 10 个合法任务文本在去掉语言、goal predicate 和 object-of-interest 后具有相同的物理 BDDL；对同一完整初始 simulator state、同一随机状态和同一固定动作，分别在 10 个 Goal 下 fresh replay。3 个初始状态、4 组动作共 **120** 次 replay 中，初始 sim/qpos/qvel/contact 和三步未来观测均 **120/120 exact**。这给出了环境端的条件独立控制：Goal 没有改变真实物理未来。

在同一批输入上，WorldAgen native forecast 的非 Clean Goal effect 平均 MSE 为 **0.4913**，所有非 Clean 条件均非零；Bus-Cut 的跨 Goal 方差最大为 **6.2×10⁻¹⁵**，并且仍保留全部动作敏感性。相对真实三步 target 的单元格 MSE 差仍为混合符号（均值 **+0.0869**，范围 **−0.1610 到 +0.2006**）；但按共享初始状态聚类，对均匀抽取的非 Clean 合法 Goal 求平均后，三状态均为正（**+0.0615/+0.1260/+0.1021**，总体 **+0.0965**，state-cluster bootstrap CI **[+0.0615,+0.1260]**）。因此它支持“Clean Goal 不是无偏物理 target comparator”的描述性证据，仍不是“每个 Goal 都让预测变差”。Bus-Cut 虽恢复了跨 Goal 不变性，但相对 native Clean target 的 baseline MSE 平均上移 **+0.0884**（三状态分别 +0.0530/+0.1425/+0.0696），再次说明预测修复可能引入 fidelity cost。结果见 [`same-scene summary`](../runs/worldagen_same_scene_goal_v1/summary.json)、[`analysis`](../runs/worldagen_same_scene_goal_v1/analysis.json) 和 [`script`](../scripts/pilot_worldagen_same_scene_goal.py)。

同一场景又加入了空指令（`Goal=""`）的 prompt-independent diagnostic，并扩展到完整的 **50 个合法初始 simulator state × 4 组固定动作**。200 次真实 target replay 的初始状态均 exact；相对 Clean，空指令 target-MSE 在 **50/50** 个初态为正，均值 **+0.0744**，state-cluster bootstrap CI **[+0.0665,+0.0824]**；empty 与 Clean forecast 的平均 MSE 距离为 **0.2139**。这支持 Clean Goal 在该 checkpoint 上携带稳定的任务先验，但不能把“空指令更差”解释为 Goal 普遍有益或证明完整 no-goal baseline。结果见 [`50-state empty-goal comparator`](../runs/worldagen_same_scene_empty_goal_all50_v1/summary.json)、[`figure`](figures/worldagen_empty_goal_comparator.svg)、[`3-state pilot`](../runs/worldagen_same_scene_empty_goal_v1/summary.json) 和 [`script`](../scripts/pilot_worldagen_same_scene_empty_goal.py)。

在此基础上，我们加入了意义保持的自然语言改写控制：对 3 个真实窗口的每个任务手写 2 个同义 Goal，固定 image、state、已执行 action、RNG 和 mask。native World 输出相对 canonical Goal 的 relative-L2 在 6/6 个改写中非零，均值 **0.00844**、最大 **0.01131**；Bus-Cut 下 6/6 个改写均 exact zero，而 policy 输出仍保持文本敏感。这一效应明显小于合法 Goal-Swap 的主效应，但证明模型的 Goal-Bus 不是只对句号或空格变化敏感；同时它也不把同义改写升级成物理损害证据。结果见 [`semantic paraphrase control`](../runs/worldagen_semantic_paraphrase_v0/summary.json) 和 [`script`](../scripts/pilot_worldagen_semantic_paraphrase_control.py)。

## 机制级证据

WorldAgen 的 native mask 没有直接 Goal→World 边，但 Goal 可以写入 input-action token，再由该共享 carrier 影响 future-world token。证据链包括：

- 精确 edge intervention：切 carrier 恢复 world invariance，同时保留 policy 的 Goal sensitivity。
- matched policy-only cut：同样的删边预算不恢复 invariance。
- activation patch：在中层替换 goal-A 的 action-token hidden state 后，world residual 显著向 goal-A 移动，而 policy residual 大体保留 goal-B。
- AlphaBrain 的独立审计显示其原生动作同样由 action latent 直接解码；World Forecast 在原生 rollout 中不参与 Action Selection。

因此本文把新增 planner 明确标为 **independent forecast-driven planner**，不把它冒充成原生 WorldAgen controller。

WorldAgen/AlphaBrain 的 checkpoint 与调用顺序审计见 [`audit_goal_bus_action_selection.py`](../scripts/audit_goal_bus_action_selection.py) 和 [`summary.json`](../runs/goal_bus_action_selection_audit_v0/summary.json)。

## 决策级实验：严格固定控制

代码：[`pilot_worldagen_counterfactual_action_ranking.py`](../scripts/pilot_worldagen_counterfactual_action_ranking.py)。

修正后的协议使用 6 个与真实 LIBERO-10 demonstration task 对应的任务（task ids 0、2、4、6、7、9），每个任务固定 8 个真实 demonstration action trajectories：

- 候选动作集合在所有条件间完全相同；动作来自一次固定的真实演示采样，不因 Goal 重新生成。
- WorldAgen 分支分别接收 Clean Goal、相邻合法 Goal、Bus-Cut Goal 和 matched Control-Cut Goal。
- 所有条件的 CLIP ranking target 都固定为原始 Clean Goal。这个约束在早期脚本中曾被错误实现，已修正；早期结果被标记为 superseded，不纳入结论。
- 每个被选动作轨迹在同一 LIBERO 初始状态中执行 30 个 simulator steps。
- 真实效用记录两类量：B​DDL sparse reward/success，以及 simulator 的真实 EEF 到首个操作实体的距离进展。后者是中间接近效用，不冒充任务成功。

完整结果（修正 primary-view/first-chunk Forecast Error）：[`summary_fixedscore_v2_tasks6.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_v2_tasks6.json)。统计脚本：[`analyze_worldagen_counterfactual_action_ranking.py`](../scripts/analyze_worldagen_counterfactual_action_ranking.py)。

这里的 Forecast MSE 只比较第 3 步对应的第一个 primary-view future chunk；候选动作选择仍基于该 forecast，而真实 EEF 后果按选中轨迹继续回放 30 步。

### 相邻 Goal-Swap 结果（6 tasks；global-minmax planner）

| 条件 | Ranking flip rate 相对 Clean | Forecast MSE | EEF 接近进展 | 相对 Clean 的 EEF 差值 |
| --- | ---: | ---: | ---: | ---: |
| Clean | — | 1.9085 ± 0.1784 | −0.0608 ± 0.1474 m | — |
| Goal-Swap | 0.4464 ± 0.0774 | 1.9087 ± 0.1723 | −0.1155 ± 0.1200 m | **−0.0547 m** |
| Goal-Swap + Bus-Cut | 0.4881 ± 0.0949 | 1.9032 ± 0.1982 | −0.0454 ± 0.1079 m | **+0.0154 m** |
| Goal-Swap + Control-Cut | 0.4464 ± 0.0774 | 1.9087 ± 0.1723 | −0.1155 ± 0.1200 m | **−0.0547 m** |

matched Control-Cut 与 Goal-Swap 的全部候选分数最大绝对差为 **0**，因此其真实动作结果也完全相同。这个 control 证明：结果来自 World branch 的排序，而不是 policy-only 删边。

Bus-Cut 在这组相邻 Goal 的 selected action 只在 **1/6** 个任务恢复为 Clean；EEF 均值相对 Clean 为 **+0.0154 m**，所以它是部分的 World 分支干预结果，不是已验证的决策恢复。

所有四条件的 sparse reward 都是 0，success 都是 0/6。因而当前决策证据是“真实动作轨迹的中间后果变化”，不是“任务完成率下降”。

为检查回放实现和控制器确定性，随后对同一固定 seed 的每个条件做了 **3 次独立 fresh-environment 回放**（不是把动作块串成更长轨迹）。6 个任务、4 个条件的 EEF 进展在三次重复中的最大范围均为 **0 m**（浮点误差约 `3.4e-17`），所以表中差异不是重复回放噪声造成的；这仍是固定 seed 的确定性复核，不应当解读为跨 seed 的方差估计。结果见 [`summary_fixedscore_tasks6_repeat3.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6_repeat3.json)，统计见 [`summary_fixedscore_tasks6_repeat3_statistics.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6_repeat3_statistics.json)。

另外用 **seed 23** 重复了同一 6-task 协议，只改变 LIBERO 初始 seed，候选动作、checkpoint 和评分目标不变。将 seed 17 与 seed 23 合并为 12 个 task-seed 配对后，global-minmax 的 Goal-Swap ranking flip 为 **0.4435 ± 0.1092**，EEF 差值为 **−0.0701 ± 0.1493 m**；Bus-Cut 的 EEF 差值为 **+0.0216 ± 0.1841 m**，Control-Cut 与 Goal-Swap 仍逐项恒等。按物理 task 聚类的 bootstrap 区间为 ranking flip **[0.3750, 0.5238]**、EEF 差值 **[−0.1680, 0.0520] m**；后者跨过 0，所以不作显著性结论。两次 seed 的 selected action 对第一 seed 的一致率为 **11/12**，说明排序扰动不是单个 seed 的偶然现象，但样本仍不足以作显著性检验。seed 23 原始结果见 [`summary_fixedscore_tasks6_seed23.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6_seed23.json)，合并统计见 [`fixedscore_seed_replicate_statistics.json`](../runs/worldagen_counterfactual_action_ranking_v0/fixedscore_seed_replicate_statistics.json)。

在额外的 candidate-utility audit 中，每个任务的 8 个候选轨迹都被独立回放一次，再计算被选动作相对候选集合真实最佳 EEF 进展的 regret。对这个 global-minmax planner，Clean 的平均 regret 为 **0.161 m**；相邻 Goal-Swap 为 **0.301 m**，Control-Cut 与 Goal-Swap 相同。这个量使用同一 simulator 初始状态和真实 EEF 状态，未使用模型预测分数。

为校准 planner 是否真的能区分候选动作，新增了严格对齐的 step-3 protocol：只使用 WorldAgen 第一个 primary-view future chunk 评分，并只执行候选轨迹的前 3 步；所有 8 个候选都回放后才读取真实后果。用当前观测 patch 的 scalar mean/std 做 `reference_scalar` 重建时，Clean 的真实 EEF regret 为 **0.0081 m**，低于均匀随机候选期望的 **0.0131 m**；Goal-Swap offset 1 的 regret 为 **0.0083 m**，offset 5 为 **0.0138 m**。相邻 Goal-Swap 改变了 6/6 个任务的 selected action（offset 1），但 EEF 均值只从 **+0.0053 m** 变为 **+0.0050 m**；offset 5 变为 **−0.0005 m**。这说明 calibrated planner 能提供真实后果排序的最低校准，而 Goal-Bus 造成的决策效应是候选级、任务依赖且幅度有限。

同一协议再用 seed 31 和 37 复核后，三个 seed 共 **18 个 task-seed 配对**。`reference_scalar` 的 Clean regret 为 **0.0087 ± 0.0088 m**，随机基线为 **0.0131 ± 0.0024 m**；offset 1 regret 为 **0.0093 ± 0.0080 m**，相对 Clean selected action 的 EEF 差值为 **−0.0007 ± 0.0066 m**，task-cluster bootstrap CI 为 **[−0.0056, 0.0040] m**；offset 5 regret 为 **0.0138 ± 0.0091 m**，EEF 差值为 **−0.0051 ± 0.0105 m**，CI 为 **[−0.0139, 0.0026] m**。offset 1/5 分别改变 **17/18** 和 **13/18** 个 selected action；Bus-Cut 的 regret 为 **0.0069 ± 0.0042 m**，但 **18/18** 次仍改变 Clean action。Control-Cut 对两个 offset 的分数最大差均为 **0**。这些是按 task 聚类的描述性结果，区间跨 0 时不作显著性结论。完整聚合见 [`aligned planner replicate statistics`](../runs/worldagen_aligned_planner_replicates_v1/statistics.json) 和 [`aggregation script`](../scripts/aggregate_worldagen_aligned_replicates.py)。

该三 seed 复核的可视化见 [`aligned planner replicate figure`](figures/worldagen_aligned_planner_replicates.svg)。

为观察较长真实后果，保持第 3 步 forecast 选择不变，只把选中轨迹的真实回放延长到 30 步，并在三个 seed 上重复。`reference_scalar` 的 Clean regret 为 **0.0758 ± 0.1048 m**，低于随机基线 **0.1345 ± 0.0329 m**；offset 1 regret 为 **0.1044 ± 0.0412 m**，EEF 差值为 **−0.0286 ± 0.1302 m**；offset 5 regret 为 **0.1481 ± 0.1102 m**，EEF 差值为 **−0.0723 ± 0.1570 m**，task-cluster bootstrap CI 为 **[−0.1956, 0.0226] m**。offset 5 因此在这个经过 Clean 校准的长 horizon planner 上呈现更大的真实后果损害，但 CI 仍跨 0，不作显著性结论。Bus-Cut 将 regret 降到 **0.0373 ± 0.0582 m**、EEF 差值变为 **+0.0385 m**，却在 **18/18** 次改变 Clean action，说明它修复的是一个新的 planner baseline，而不是恢复原决策。完整结果见 [`30-step aligned replicate statistics`](../runs/worldagen_aligned_planner_h30_replicates_v1/statistics.json)。

对应图见 [`30-step aligned consequence figure`](figures/worldagen_aligned_h30_replicates.svg)。

进一步的合法 Goal dose-response sweep 扩展到已有的 6 个 LIBERO 任务（0、2、4、6、7、9），并用 seed 17/31/37 重复，得到 **18 个 task-seed 对、162 个 counterfactual**。三 seed 聚合的 `reference_scalar` 结果显示，offset 1–9 的跨 task-seed 平均真实 EEF 差值全部为负，范围 **−0.0193 到 −0.1330 m**；offset 5 的 EEF 差值为 **−0.0723 m**、regret 为 **0.1481 m**，与此前固定 offset 的 6-task 长回放结果完全一致。任务层面仍有正值，不能把跨任务均值写成普遍单任务损害。各 offset 的 pairwise Action Ranking Flip Rate 为 **0.448–0.556**，selected-action 改变率为 **0.722–1.000**。这说明在六任务三种子范围内，Goal-Swap 的排序扰动和中间真实后果比 2-task pilot 更稳定，但仍不意味着任务完成率损害。primary-view Forecast Error 仍不是单调剂量指标：Clean 为 **1.9070**，offset 5 为 **1.8621**，因此不能用 Forecast Error 直接替代决策后果。对 162 个样本做的预先定义中介诊断也显示，Forecast Error 改变量与 ranking flip 的 pooled Spearman 仅 **0.099**，与 EEF 差值为 **−0.080**，与 regret 为 **−0.080**；按 offset 分层时相关符号反转。因此“预测误差变大”不是决策损害的充分指标。Bus-Cut 在 **12/18** 个 task-seed 对中选到候选集合真实 EEF 最佳动作，regret 为 **0.0373 m**，但 **0/18** 次恢复 Clean selected action；这把 Bus-Cut 定位为部分有效的 planner-specific repair，而不是完整决策恢复。此前 2-task sweep 中的 54/54 结果保留为 pilot，不作为跨任务主结论。结果见 [`six-task three-seed legal-goal sweep`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/statistics.json)、[`forecast-error mediation diagnostic`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/mediation_statistics.json)、[`sweep aggregation script`](../scripts/aggregate_worldagen_aligned_goal_sweep.py) 和 [`six-task dose-response figure`](figures/worldagen_aligned_goal_sweep_all6_replicates.svg)。
为直接检查“预测修复”和“决策修复”是否可以分离，新增了一个不读取 replay utility 的固定 repair path：把 Clean 与 Bus-Cut 的候选 rank 按预先固定权重混合。Bus-Cut 权重 1.0 时 regret 为 **0.0373 m**、Clean-action recovery 为 0；Clean 权重 0.75 时 recovery 上升到 **0.778**、regret 为 **0.0483 m**；权重 0.25 时 regret 降到 **0.0291 m**，但 recovery 只有 **0.056**。这形成明确的 utility/recovery Pareto，而不是单一“恢复成功”指标。它是需要 Clean reference branch 的 planner-level diagnostic，不是原生策略修复。结果见 [`decision repair path`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/decision_repair_path.json)、[`repair-path script`](../scripts/analyze_worldagen_decision_repair_path.py) 和 [`repair-path figure`](figures/worldagen_decision_repair_path.svg)。
候选数量敏感性也使用同一批已回放动作做了固定子集复核：8-candidate 主协议的 offset 1/5/8 regret 为 **0.1044/0.1481/0.2088 m**，降到固定 linspace-4 子集后为 **0.0915/0.1550/0.1940 m**；对应 ranking flip 仍为 **0.546–0.583**，Control-Cut 分数恒等。四候选结果不是重新采样的独立协议，但说明排序扰动并不依赖恰好 8 条候选。结果见 [`candidate-cardinality sensitivity`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/candidate_cardinality.json) 和 [`cardinality script`](../scripts/analyze_worldagen_candidate_cardinality.py)。
为先验证“物理未来是否应当 Goal-invariant”，新增了 matched physical-target audit。每个 task-seed 的 8 个冻结候选都使用同一初始模拟状态和同一真实回放，在执行前三步后保存一个物理 target patch；Clean、Goal-Swap、Bus-Cut 和 Control-Cut 只改变预测分支，target 不随条件改变。对 18 个 task-seed 对、162 个 counterfactual 重新计算的 target-MSE 与原始存储值最大误差为 0。Goal-Swap 的 model-space forecast divergence 平均为 **0.4603**，但相对 Clean 的 physical-target MSE 差跨 offset 混合：全体均值 **−0.0121**，task-cluster bootstrap CI **[−0.0434, 0.0181]**；offset 2 为 **−0.0581**，offset 3 为 **+0.0340**，offset 5 为 **−0.0449**。Goal divergence 与 physical-error delta 的 Spearman 相关为 **0.117**（p=**0.139**）；这不支持“divergence 越大，物理预测越差”的单调规律。Control-Cut 与对应 Goal-Swap 的 target-MSE 最大差为 **0**，而 Bus-Cut 分支相对 Clean 的 target-MSE 差为 **−0.0144**。因此当前最稳健的基础结论是：同一物理未来下，Goal 会改变 world forecast；但这本身不等于物理误差增加，且 Goal-invariance 需要按物理 target 定义，而不能把所有 goal-conditioned latent 语义都当作动力学。后续尝试从同一候选重放中补提 qpos/qvel/contact target 被主动作废：虽然重建 init state hash 一致，但未持久化的 controller/global RNG 使 EEF replay 最大偏差达到 **0.3046 m**，因此没有把重建状态用于任何分析；完整失败记录见 [`invalidated state-target extraction`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/physical_target_extraction_failure.json)。该审计的 target 是前三步后的 decoded primary-view patch，不是完整 latent-state 或任务成功率。同一原始 ranking 进程内的 repeat-3 fresh-environment audit 中，所有 EEF/进度数值的重复跨度为 **0**；但这只证明进程内确定性。结果见 [`goal-invariance physical audit`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/goal_invariance_physical_audit.json) 、[`audit script`](../scripts/analyze_worldagen_goal_invariance_physical.py) 和 [`invariance specification`](../configs/goal_bus_physical_invariance_spec.json)。图见 [`physical invariance figure`](figures/worldagen_goal_invariance_physical.svg)。

为把 physical target 从图像 patch 扩展到更直接的物理输出，新增 AlphaBrain CosmosPolicy 的 future-output slot audit。固定 21 个真实 image/state/recorded-action 窗口（7 个 episode、同一 observed +16 frame target、同一 sampler noise），比较合法 Goal-Swap 与 fixed-action world cut 的 slot 5/6/7：future proprio、future wrist 和 future primary。Native Goal-Swap 对 future proprio 的绝对 latent delta 在 **21/21** 窗口非零，均值 **0.0607**，episode-cluster CI **[0.0376, 0.0970]**；wrist 和 primary 分别为 **0.0521 [0.0401, 0.0632]** 与 **0.0462 [0.0303, 0.0687]**。Fixed-action cut 在三个槽上均为 **21/21 exact zero**。但与真实 +16 frame target 比较时，future proprio 的 MSE 差为 **+0.0084**、CI **[−0.0139, 0.0325]**，future wrist 为 **+0.0248**、CI **[0.0091, 0.0394]**，primary 的 CI 仍跨零。作为 repair tradeoff，fixed-action cut 相对 native Clean 的 target-MSE cost 在 wrist 为 **+0.0918**、CI **[0.0673, 0.1206]**，primary 为 **+0.0416**、CI **[0.0318, 0.0558]**，说明恢复 invariance 可能牺牲 forecast fidelity。因此新增结果证明 Goal 会改变直接的未来 proprio 输出，但不支持把所有 Goal divergence 写成普遍物理预测误差；“改变预测”和“预测变差”仍需分开。将合法 offset 扩展到 **1/5/9** 后共 63 个窗口，future proprio 的绝对 Goal effect 在各 offset 的 **21/21** 窗口均非零，均值分别为 **0.0512/0.0607/0.0728**，fixed-action cut 在三槽、63 个窗口中仍全部 exact zero。结果见 [`AlphaBrain physical-slot audit`](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v1/physical_slot_audit.json)、[`offset extension`](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v1/physical_slot_offsets.json)、[`matrix`](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v1/summary.json) 和 [`figure`](figures/alphabrain_physical_slot_offsets.svg)。

作为语义不变性的轻量控制，在 3 个真实输入窗口中固定图像、state、记录动作、mask 和 RNG，只改变同一 task text 的表面形式。双空格和小写变体在 native 与 Bus-Cut 下均为 **3/3 exact equal**；只追加句号时 native world relative L2 平均 **0.00259**（最大 **0.00279**），而 Bus-Cut world 在 **3/3** 窗口仍精确不变。该结果只能说明 released tokenization/共享 carrier 对无关文本表面存在小幅敏感性，不能当作物理误差或攻击成功；它也不是完整 paraphrase benchmark。结果见 [`semantic surface control`](../runs/worldagen_semantic_goal_invariance_v0/summary.json) 和 [`surface-control script`](../scripts/pilot_worldagen_semantic_goal_invariance.py)。

为修复 state/contact target 的复现缺口，新增了一个隔离的 RNG-persisted audit：不重新运行模型，只读取冻结候选；在 6 个任务（0/2/4/6/7/9）、3 个 seed（17/31/37）上共 144 个候选，每个候选在保存并恢复 Python/NumPy/Torch/CUDA RNG 后做两次独立 fresh replay，共 288 个 replay episode。144/144 候选的初始 simulator hash、qpos/qvel、contact count、前三步图像 hash、EEF 和真实进度均逐项 exact。该结果把 state-target 持久化协议扩展到完整 task-seed 集合，但它本身不比较 Goal 条件，也不增加攻击效应量。结果见 [`aggregate audit`](../runs/worldagen_reproducible_state_target_all6_replicates_v1/statistics.json)、[`per-seed audit`](../runs/worldagen_reproducible_state_target_all6_seed17/summary.json) 和 [`audit scripts`](../scripts/pilot_worldagen_reproducible_state_target.py)。

最后做了 forecast-only attacker selection：攻击者只用每个 task-seed 的模型分数，事先从合法 offset 1–9 选择最大 pairwise ranking flip 的 Goal，完全不读取 replay utility、未来观测或目标 MSE。该选择器的 EEF 差值为 **−0.0995 m**、regret **0.1753 m**，相对所有合法 offset 的均匀期望分别差 **−0.0278 m / +0.0278 m**；task-cluster bootstrap CI 为 **[−0.0786, 0.0230] m**。以 rank-L1 或 z-score-L1 选择 offset 则分别优于均匀期望，说明“最大 Forecast/Ranking Divergence”不是稳定的攻击目标。该实验把 Forecast Divergence 与攻击成功明确分开。结果见 [`forecast-only attacker`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/forecast_only_attacker.json) 和 [`attacker analysis script`](../scripts/analyze_forecast_only_goal_attacker.py)。
进一步直接使用保存的 WorldAgen `worlds_step3` latent 张量计算 swap 相对 Clean 的 model-space divergence，并在读取 replay utility 前选择最大 divergence offset。按候选平均 latent MSE 选择时，EEF 差值 **−0.0953 m**、regret **0.1711 m**，相对均匀 offset 的 regret 差 **+0.0235 m**，task-cluster CI **[−0.0558, 0.1111] m**；按候选最大 latent MSE 选择时对应 regret 差 **+0.0262 m**，CI **[−0.0533, 0.1132] m**。因此 latent-level Forecast Divergence 与更差真实后果在方向上相容，但仍不能被写成可靠攻击定律；它是模型空间扰动指标，不是物理误差或攻击成功。结果见 [`latent-divergence attacker`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/latent_divergence_attacker.json) 和 [`latent attacker script`](../scripts/analyze_latent_divergence_goal_attacker.py)。

增加了一个留一 seed 外推的 forecast-only 选择测试：每个 held-out seed 的 Goal offset 只由另外两个 seed 的 forecast divergence 特征决定，再把同一个 offset 应用于该 seed 的全部六个任务；测试 seed 的真实 replay utility 只在 offset 固定后读取。latent mean/max-MSE 和 raw-z-L1 选择器在三个 fold 都选 offset 9，EEF 差值 **−0.0380 m**、regret **0.1138 m**，相对均匀 offset 的 regret 差 **−0.0338 m**，task-cluster CI **[−0.1053, 0.0534] m**；pairwise-flip 选择器选 **7/9/7**，regret 差 **−0.0245 m**、CI **[−0.0985, 0.0446] m**；rank-L1 选择器选 **2/9/9**，regret 差 **−0.0428 m**、CI **[−0.1002, 0.0246] m**。因此最大 latent/ranking divergence 没有跨 seed 泛化为更强的攻击选择，反而在这组留出测试中低于均匀 offset 期望；这进一步限制了从 Forecast Divergence 推出攻击成功的解释。结果见 [`out-of-sample attacker`](../runs/worldagen_aligned_goal_sweep_h30_all6_replicates_v1/oos_attacker.json) 和 [`OOS attacker script`](../scripts/analyze_worldagen_oos_goal_attacker.py)。

另外做了一个固定 alpha 的 cross-task ridge value probe：只用训练 task 的 Clean forecast 和真实 EEF 结果拟合，在未见 task 上对所有条件选择动作。两种互补 task split 都显示 Goal-Swap 会改变选择（test tasks 6/7/9 的 offset 1/5 为 5/9、6/9；test tasks 0/2/4 为 3/9、4/9），但 Clean 校准一侧只有一个 split 优于随机，EEF 方向在两个 split 间反转。该 probe 因而支持“forecast 可以穿过独立 value head 改变选择”，同时反驳跨任务普遍效用损害；它不进入主效应量，只作为预先固定的负结果与边界审计。结果见 [`cross-task value probe split 1`](../runs/worldagen_cross_task_value_probe_v1/test_tasks679.json)、[`split 2`](../runs/worldagen_cross_task_value_probe_v1/test_tasks024.json) 和 [`probe script`](../scripts/pilot_worldagen_cross_task_value_probe.py)。

同一对齐协议下，旧 global-minmax scorer 的 Clean regret 为 **0.0200 m**，反而高于随机基线 **0.0131 m**，所以它只能作为 scorer-specific ranking probe，不能支撑一个普遍的“价值损害” Claim。完整 step-3 结果和协议见 [`worldagen_aligned_planner_v1`](../runs/worldagen_aligned_planner_v1/summary.json)、[`statistics.json`](../runs/worldagen_aligned_planner_v1/statistics.json) 和 [`pilot_worldagen_aligned_planner.py`](../scripts/pilot_worldagen_aligned_planner.py)。

### Planner 敏感性与负结果

上表不能被解读为“Goal-Swap 必然降低真实效用”。global-minmax 是一个固定、可复现且不使用未来观测的独立 planner，但其把归一化 WorldAgen patch 重建为伪图像。为检查结论是否依赖这个选择，增加了 `reference_patch` scorer：它只用当前观测 patch 的均值和方差反归一化预测 patch，再计算同一个 Clean Goal 的 CLIP 相似度；候选集合、随机种子、回放状态和真实效用定义全部不变。

在 6-task 的相邻 Goal-Swap 复核中，`reference_patch` 的 EEF 接近进展为 Clean **−0.0620 m**、Goal-Swap **−0.0472 m**、Bus-Cut **−0.0239 m**，对应 Goal-Swap 相对 Clean **+0.0149 m**、Bus-Cut **+0.0381 m**；而 ranking flip 仍为非零（Goal-Swap 均值 **0.4286**）。因此稳健成立的是 Goal-Swap 会改变排序和被执行的真实轨迹，不能声称效用损害方向与 planner 无关。

我们还把机制级 activation patch 作为另一种修复消融：在全部 6 个任务的每个候选动作上，仅替换 Goal-Swap forward 中对应 Clean forward 的 action-token hidden states。它确实降低了 forecast 到 Clean 的相对 L2：native / Bus-Cut / layer-4 / layer-8 的候选均值分别为 **0.5295 / 0.4750 / 0.1349 / 0.1323**。但 ranking flip 仍为 **0.3333**（layer 4）和 **0.3690**（layer 8），Clean selected action 只在 **2/6** 任务恢复；EEF 差值分别为 **+0.0877 m** 和 **+0.0372 m**。task 0 的 layer-8 patch 确实恢复了 Clean action，task 2 则没有，说明这不是稳定的任务无关修复。这是一项直接的预测—决策分离证据：forecast residual 下降并不保证 clean decision recovery。结果见 [`summary_activation_patch_tasks6_v2.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_activation_patch_tasks6_v2.json) 和 [`summary_activation_patch_tasks6_v2_statistics.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_activation_patch_tasks6_v2_statistics.json)，代码见 [`pilot_worldagen_activation_patch_action_ranking.py`](../scripts/pilot_worldagen_activation_patch_action_ranking.py)。

两种 scorer 的原始 30-step ranking/EEF 文件分别是 [`summary_fixedscore_tasks6.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6.json) 和 [`summary_referencepatch_tasks6.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_referencepatch_tasks6.json)；自动对齐比较见 [`scorer_sensitivity_tasks6.json`](../runs/worldagen_counterfactual_action_ranking_v0/scorer_sensitivity_tasks6.json)，代码是 [`compare_worldagen_counterfactual_scorers.py`](../scripts/compare_worldagen_counterfactual_scorers.py)。这些文件的 ranking/EEF 结果仍可复用，但旧 Forecast MSE 列不再作为主结果；canonical 的 Forecast Error 使用上面的 v2 primary-view/step-3 protocol。合法 Goal 全 sweep 的 reference-patch candidate-utility 统计见 [`referencepatch_all_goal_evalall_statistics.json`](../runs/worldagen_counterfactual_action_ranking_v0/referencepatch_all_goal_evalall_statistics.json)。该 sweep 中 native EEF 差值既有正也有负（offset 5、7 的两项配对为负，offset 3 的均值为正），进一步排除了“普遍损害”解释。

### 所有合法 Goal offset sweep（global-minmax planner）

为避免只挑一个相邻目标，task 0 和 task 2 各评估其余 9 个合法 Goal offset，共 18 个 native counterfactuals。结果见 [`fixedscore_all_goal_offset_statistics.json`](../runs/worldagen_counterfactual_action_ranking_v0/fixedscore_all_goal_offset_statistics.json)；全候选真实回放结果见 [`fixedscore_all_goal_evalall_statistics.json`](../runs/worldagen_counterfactual_action_ranking_v0/fixedscore_all_goal_evalall_statistics.json)。

- native Goal-Swap 的 EEF 进展相对 Clean 平均 **−0.0649 m**；18 个配对中 11 个为负，7 个为正。
- 18 个合法 Goal counterfactuals 的 native 选中动作 regret 描述性均值约 **0.226 m**，高于 Clean 的 **0.161 m**；这不是显著性检验。
- 最坏 offset 的描述性均值为 offset 3，差值 **−0.2071 m**；这是事后汇总，不能当作预注册攻击选择。
- Bus-Cut 后跨 9 个 offset 的 selected action 在每个任务内都稳定为一个候选，说明它确实抑制了 Goal 对 planner ranking 的变化；但它相对 Clean 的平均 EEF 差值仍为 **−0.0267 m**，不是恢复为 Clean。
- Control-Cut 在每个 offset 上保留 Goal-Swap 的 ranking，且分数逐项恒等；因此 Bus-Cut 的稳定化来自 World branch 干预，而不是删边本身。

这组结果支持一个更精确的决策级结论：**Goal-Bus 污染可以改变真实动作轨迹、候选 regret 和动作排序；在某个固定 planner 下它可以降低接近进展，但该效用方向不是跨 planner 的定律。Bus-Cut 可以抑制 Goal-dependent ranking，却可能引入新的 planner baseline，不能被称为已验证的决策修复。**

为检查这一结论是否能迁移到第二个载体，新增了一个明确隔离的 AlphaBrain forecast-driven planner 可行性门：固定 task 0/seed 17、8 条冻结动作候选、同一初始状态和 30-step LIBERO replay，用 future-primary forecast 的 CLIP 分数选动作，真实 EEF 进展只从环境回放读取。Clean native 选中候选 4，regret **0.0621 m**；Goal-Swap 选中候选 6，regret **0.4228 m**。但 fixed-action Bus-Cut 的 Clean 与 Swap 分数逐项相同，却都选中候选 6，regret 仍为 **0.4228 m**，没有恢复 Clean action。这个单任务 pilot 只证明“第二载体可以进入一个独立 planner 的决策回路”，不证明跨载体普遍任务损害，也不证明 Bus-Cut 是决策修复；CLIP 排序校准与样本规模不足以升级主结论。结果见 [`AlphaBrain planner feasibility`](../runs/alphabrain_forecast_planner_v0/summary.json) 和 [`script`](../scripts/pilot_alphabrain_forecast_planner.py)。

随后将同一协议扩展到 task **0/2/4**（同一 seed 17、每 task 8 条冻结候选、16-step forecast、30-step replay）。Goal-Swap 改变了 **2/3** 个 task 的选中动作；Clean native 的 mean regret 为 **0.1836 m**，Goal-Swap 为 **0.2251 m**，fixed-action Bus-Cut 为 **0.2796 m**，且 Clean/Swap cut 分数逐项恒等。task 4 的 Goal-Swap regret 反而从 **0.3765 m** 降到 **0.1402 m**，因此跨载体结果支持“forecast 可以进入真实 planner 决策回路”，但不支持普遍效用损害；Bus-Cut 只在 **1/3** task 恢复 Clean action。这是跨载体 feasibility evidence，不是 native policy 或 task-success 结论。结果见 [`AlphaBrain 3-task planner`](../runs/alphabrain_forecast_planner_multitask_v1/summary.json)。

为避免把三 task pilot 当成主结果，又扩展到与 WorldAgen 主 sweep 相同的六个 task（0/2/4/6/7/9），仍固定 seed 17 和同一候选/评分/replay 协议。Goal-Swap 改变 **5/6** 个 task 的选中动作；Clean/Goal-Swap/Bus-Cut regret 均值分别为 **0.1646/0.1911/0.2433 m**，但 task 4 和 6 的效用方向改善，task 0、7、9 变差，净差仅 **+0.0265 m**。Bus-Cut 的 Clean-action recovery 为 **2/6**，Clean 与 Swap 的 cut 分数仍逐项恒等。六 task 结果加强了“预测可进入独立 planner 决策回路”的结论，却明确反驳“Goal-Swap 普遍损害真实效用”与“Bus-Cut 完成决策修复”。结果见 [`AlphaBrain six-task planner`](../runs/alphabrain_forecast_planner_six_task_v1/summary.json)。

随后把同一六 task planner 在三个显式 LIBERO 初始状态（`init_index=0/1/2`）上复放，共 **18 个 task-state rows**；seed 只用于 RNG，物理初态由 `init_index` 明确选择。Goal-Swap 改变 **15/18** 个 selected action；Clean/Swap/Bus-Cut regret 均值为 **0.1511/0.1885/0.2607 m**，Swap−Clean 为 **+0.0374 m**。按初始状态聚类的 bootstrap CI 为 **[+0.0098,+0.0758] m**，按 task 聚类为 **[−0.1086,+0.1931] m**；因此跨状态有描述性上升，但跨 task 仍不能作普遍损害结论。Bus-Cut 只恢复 **3/18** 个 Clean action，Clean/Swap cut scores 逐项恒等；它切断了 Goal ranking 通路，却改变了 planner baseline。该结果支持跨状态的 planner reachability，不把它升级为 native policy attack 或 universal repair。结果见 [`replicate statistics`](../runs/alphabrain_forecast_planner_replicates_v1/statistics.json) 和 [`aggregation script`](../scripts/aggregate_alphabrain_forecast_planner_replicates.py)。

决策层的 ranking、forecast residual、真实 EEF 后果和 Clean-action recovery 汇总见 [decision evidence figure](figures/goal_bus_decision_evidence.svg)。

## 贡献

1. 提出 Goal-invariance 与 Decision-invariance 的分离，并把合法 Goal 当作控制平面输入而非普通 prompt 条件。
2. 用 edge cut、matched cut、alternate-action control 和 activation patch 定位共享 action-token carrier 的因果路径。
3. 构建固定候选、固定评分、真实 simulator consequence 的 counterfactual action-ranking protocol，避免把 forecast divergence 直接当成攻击成功。
4. 发现两个具有实际安全意义的边界：forecast invariance repair 不自动带来 decision recovery；真实效用的损害方向依赖 planner，不能从 Forecast Divergence 单独推出。

## 当前结论与限制

## Go/No-Go 判断

**Go：**继续把 Goal-Bus 作为“预测级控制平面泄漏 + 决策不变性单独验证”的论文主线。现有证据已经足以支持：合法 Goal 能通过共享 action-token carrier 因果改变 World Forecast；经过 Clean 校准的独立 forecast-driven planner 在 6 个任务、3 个 seed、162 个合法 Goal counterfactual 的 30-step 真实回放中会稳定改变候选排序并增加候选 regret、降低 EEF 接近进展；Bus-Cut 能恢复 World 的 Goal-invariance，并在部分 task-seed 对降低 regret，但不保证恢复 Clean selected action。

**No-Go：**暂不把结果表述为原生 WorldAgen/AlphaBrain 已遭受决策级攻击，也不把 Goal-Swap 的 Forecast Divergence 直接等同于普遍的真实效用损害。原生 policy 没有消费 World Forecast；LIBERO 短回放没有产生 sparse success；而 global-minmax 与 reference-patch 两个固定 scorer 给出的 EEF 效用方向相反。

因此当前项目的决策级结论是“可重复的、在一个经过 Clean 校准的 planner 上可造成长 horizon 真实后果损害的 planner-dependent decision perturbation”，而不是“已证明的 universal decision harm”。下一阶段若要升级为强 Go，需要加入真正使用 forecast 的原生或公开 planner、更多任务和更多独立 seed，并预先固定物理效用与 scorer。

当前最强、可以写进主文的 Claim 是：

> **Goal-Bus 是一个可被因果切断的预测级控制平面泄漏；当独立 planner 消费该 forecast 时，它能改变动作排序、候选 regret 和真实 simulator 轨迹，但“恢复 forecast invariance”与“恢复 clean decision”是两个必须分别验证的安全目标，效用损害方向还取决于 planner。**

仍不能声称：

- 原生 WorldAgen 或 AlphaBrain 已发生决策级攻击，因为它们的原生 policy 不使用 World Forecast。
- 当前 checkpoint 已证明任务成功率损害；LIBERO sparse reward 和 clean baseline 均未在本实验短 horizon 内成功。
- Goal-Swap 在所有合理 planner 下都会造成效用损害；reference-patch 复核显示效用方向会反转。
- Bus-Cut 已经是完整修复；它稳定了排序，却没有在所有任务中回到 Clean action 或 Clean EEF utility。
- activation patch 已经是稳健的决策修复；六个任务中它降低了 forecast residual，但 layer-4 和 layer-8 都只恢复了 2/6 个 Clean selected action。
- 6 个任务和 18 个 offset 足以作显著性结论；当前统计是配对描述性结果。

## 可复现入口

```bash
LIBERO_CONFIG_PATH=/tmp/libero_cfg \\
PYTHONPATH=vendor/LIBERO:vendor/WorldAgen \\
  WORLDAGEN_RANKING_OUT=runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_v2_tasks6.json \\
.venv-worldagen/bin/python scripts/pilot_worldagen_counterfactual_action_ranking.py \\
  --suite libero_10 --task-ids 0,2,4,6,7,9 --candidates 8 --horizon 6 \\
  --seed 17 --candidate-source recorded \\
  --candidate-data-root /tmp/worldagen_actions_merged3 \\
  --candidate-map 0:5:8,2:3:6,4:0:0,6:1:1,7:4:7,9:2:2 \\
  --recorded-replay-horizon 30 --goal-offsets 1
```

重复回放可在命令末尾加入 `--replay-repeats 3`；脚本会为每次重复创建 fresh LIBERO 环境并在结果中保留 `repeat_results`。

本地校验已通过：ranking、分析和审计脚本均可 `py_compile`；所有模型参数保持冻结，没有调用外部 API 或重新训练大模型。

将评分器切换为当前观测 patch 重建版本时，在上述命令末尾加入 `--clip-reconstruction reference_patch`，并使用不同的 `WORLDAGEN_RANKING_OUT` 文件名保存结果。
