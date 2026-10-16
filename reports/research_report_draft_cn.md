# 共享世界模型中的有序交互与规划时放大

## 摘要

本项目研究一个受限但具有安全意义的威胁：攻击者只执行环境允许的动作，提交真实观测和真实奖励，不修改样本、标签、模型权重或规划器；这些交互随后被用于共享动作条件世界模型的在线更新。当前最强候选是：**在具有共享表示和短视规划的多任务世界模型中，合法且可验证的源任务轨迹可能通过状态—动作支持塑形，重写目标任务未执行动作序列的反事实组合关系；因此已执行的一步预测、物理真实性和 actor 行为可保持正常，而 source 退出后的原生规划仍偏向错误分支。**

这不是最终广泛主张。原生 TD-MPC2 证据在 walker-run 上通过了双留出种子、时间反序、动作集合乱序和 actor 对照，但在 cup-catch、walker-stand、walker-walk 及 walker-run-backwards 上没有形成稳定跨任务族复制。最新的多时域 latent 漂移与更新算子不交换诊断也没有给出选择性内部机制。因此当前稿件把它标为候选现象，不把新意分数或单一任务族结果升级为 ICLR 级安全结论。

## 威胁模型与载体

攻击者在 cheetah-run 中执行固定的单调、分相 actuator cascade，收集 300 个真实转移；受害者使用官方 TD-MPC2 mt30-48M 检查点的原生 `model.next`、原生 MPC 和 dynamics-only native update。对照包括相同预算的随机合法轨迹，以及保持 cascade 动作向量完整多重集但打乱时间顺序后重新运行模拟器的 shuffled-cascade。

该设置不允许攻击者伪造观测、奖励、终止标签或模型输出。所有受害任务从同一官方检查点重新开始，规划器保持原生 horizon=3、512 个候选和官方 MPPI/CEM 参数。时间反序控制重新运行真实模拟器，因而不是对已记录样本做离线重排。

## 主要证据

| 目标任务 | 证据 | 结果 |
| --- | --- | --- |
| walker-run，seed31/32 | 300 步 source、300 次 dynamics-only update、500 步原生 MPC | cascade 相对 random 回报下降 19.1%/19.3%，相对 shuffled 下降 19.3%/18.2%；一步误差为 random 的 0.922 倍 |
| cup-catch，seed31/32 | 相同 source、update、规划器 | cascade 没有特异损害，故未作为正证据 |
| walker-run-backwards，seed31/32 | 相同 source、update、规划器 | 一个种子约 12.4% 下降，另一个反向；未通过复制门槛 |
| hopper-hop，seed31/32 | 相同 source、update、规划器 | 一个种子 cascade 高于 random，另一个低于 random；未通过独立任务族门槛 |
| walker-stand/walker-walk | 相同 cheetah-run source 与 300 update | cascade 相对 random 仅下降 6.43/0.91，未支持跨 walker-like 迁移 |

动作顺序的专门对照进一步把现象限定在正向 cascade：在 walker-run 的 500 步回合中，cascade 为 495.22/547.53，时间反序为 608.43/621.95，保持完整动作多重集的乱序为 613.76/669.45；四个条件的 actor 回报完全相同，因为只更新 dynamics。一个 4×4 source-target 矩阵显示，cheetah-run、walker-walk 和部分 hopper 源都能影响 walker-run，而 walker-walk/stand 目标基本不变；只有 cheetah-run 源在两个种子上同时满足一步误差不超过 random 的 1.05 倍。

walker-run 的结果说明，单步均值误差并不能替代规划时的动作效果验证；但它尚未说明具体内部因果机制。此前尝试的非正规 Jacobian 解释、SimNorm 温度防御、固定标准差地板和选择性 horizon 干预没有形成有效因果隔离。actor-only 介入可以改变结果，但同时改变了规划协议，不能作为机制防御证据。

作为强基线，使用相同 source 但改为 upstream native full update（80 次，包含 encoder、reward/value、task embedding 和 policy）后，walker-run cascade 相对 random 仍下降 19.0%/13.0%，而时间反序仅下降 6.0%/0.5%，乱序为持平或上升；cascade 一步误差为 random 的 0.993 倍。这个结果说明顺序效应不只依赖 dynamics-only 冻结设置，但 full update 同时改变 actor，因此不能把它解释为纯 MPC 失真。full update 的 cup-catch seed32 也出现 27.9% 下降，故 cup-catch 只能作为 dynamics-only 协议下的负对照，不能作为所有适配范围的安全控制。 随后将每个 full-update 条件的 actor 恢复为官方原始参数，仅保留更新后的模型做 300 步 MPC：cascade 相对 random 在 seed31/32 为 −4.6%/+8.2%，时间反序为 +0.2%/+17.1%，一步误差比为 0.993。因此先前 full-update 的双种子损失主要依赖 actor/value 等联合更新；actor 恢复后只剩弱的单种子方向，不能作为 planner-only 因果证据。

一个较弱但一致的因果定位是规划时域：在 300 步短回合中，cascade 的原生 horizon=3 回报为 300.63/306.34，而将同一适配模型改为 horizon=1 后为 376.16/374.34；random 对照分别为 362.88/372.49 和 364.83/381.70。这个干预消除了 cascade 相对 random 的方向性损害，但它改变了规划器协议，属于 planner-specific 定位和任务局部缓解，不是可直接推广的防御。

为检验“规划排序”而不是单纯 latent 漂移，固定 walker-run 初始状态和 512 条合法三步候选序列，比较模型价值排序与 simulator 三步回报。cascade 在 seed32 的 Spearman 相关为 0.071，random 为 0.291、时间反序为 0.155、乱序为 0.354；seed31 的 cascade 为 0.589，和对照接近。因此该排序诊断只作为一个种子上的支持性证据，不能单独承担机制结论。

## 轨迹条件反事实完整性：新增控制与边界

为区分“多步训练窗口”与更宽的轨迹条件效应，使用完全相同的 source 和 dose，将 dynamics-only consistency window 从原生 3 步改为 1 步，部署仍恢复原生 horizon=3 MPC。cascade 相对 random 在 walker-run 两个 seed 上下降 12.4%/16.0%，时间反序下降 6.9%/3.6%，动作多重集乱序为基本持平；cascade 一步误差比为 1.011，actor 保持不变。结果排除了“必须依赖多步 consistency loss”的解释，但没有单独证明反事实支持机制。

三个机制隔离没有通过。只更新共享 Q 头并冻结 encoder、dynamics、reward、task embedding 和 actor 后，cascade 相对 random 为 −6.5%/+4.0%，反序为 −0.1%/+3.9%，乱序为 +0.1%/+0.7%，所以没有顺序选择性的终端价值污染。Round20 提出的 shared-latent projection 短门中，cascade 的 walker-run 回报高于 random，目标五步 latent power norm 几乎不变；Round21 严格评议把更宽的 trajectory-manifold/support 重写评为 4.0/10，认为它仍可能是普通非 IID occupancy 或共享 trunk 干扰。 Round22 提出的 Cross-morphology Latent Contraction Invariant（CLCI）虽被 Gemini 评为 7.5/10，但其原生短门直接否证了中介：cascade 的 source→target top-3 action-sensitivity 子空间对齐为 0.489，random 为 0.769；三步 target branch dispersion 为 2.110，random 为 2.098。 Round23 对 CLCI 的反向“子空间排斥”解释也返回 REJECT_ALL、4.0/10：cascade、reverse、shuffle 的回报没有系统排序，branch dispersion 基本不变，不能把对齐差异重新命名为新安全机制。报告因此停止在 TD-MPC2 上继续换名扩展几何候选。反序与乱序也没有预测的高对齐/收缩模式，因此不运行 PNDR 防御，也不把 7.5 的 idea 分数写成实验新意。

独立任务族和 source-seed 对照也限制了范围。walker-walk-backwards→hopper-hop 的 cascade 损失为 18.0%/24.8%，但 shuffled 在 seed31 也损失 17.2%，反序在 seed31 更差而 seed32 反转；不能通过顺序选择性门。把 cheetah cascade 换到 source seed22 后，walker-run cascade 相对 random 反而提升 10.7%/9.4%，反序和乱序也不稳定。因此当前名称是条件性的“轨迹条件反事实完整性候选”，不写成 source-independent 或 cross-carrier 攻击。

## 新诊断与候选淘汰

近期工作将“当前状态可读但动作后果传播错误”定义为 planning-time intervention fidelity（[原始论文](https://arxiv.org/abs/2608.29998)）。本项目把它改写成攻击候选，并在 64 个固定 walker-run 根状态上执行相同物理动作对。冻结 encoder 后，三臂 root-capture MSE 都是 18.4023；干预余弦为 clean 0.7858、random 0.7603、cascade 0.7771；干预对比 MSE 为 clean 9.7611、random 10.8504、cascade 10.2094。cascade 没有显示相对 random 的特异算子偏差，所以不能用该诊断解释 walker-run 回报损失。

严格的 Gemini 机制审查还拒绝了 prototype-router hijacking、counterfactual-branch consistency、action-operator rotation 和跨反事实分支耦合四个方向；这些意见只用于候选筛选，不作为实验事实。最后一项只得到 4.5/10：现有 TD-MPC2/PETS 载体按点或独立边缘建模，没有跨分支联合头，无法在保持所有分支边缘时单独制造可验证的联合耦合差异。随后的 Round12 又审查了 recurrent latent deadlock、PETS disagreement quenching 和 SimNorm directional depletion；最高只有 5.4/10。Round13 对当前有序印刻主张的严格复核为 5.5/10，指出它仍可能是非 IID continual interference 与规划误差累积的组合，尚未达到目标新意。

两个机制诊断进一步收窄了结论。固定 walker-run 真实轨迹的 horizon=1/3/5/10 latent drift 中，反序和乱序条件有时比 cascade 更大，却没有同等回报损失；同一批 native dynamics windows 的 A→B 与 B→A 小步更新差异，cascade 的归一化均值为 0.000323，random 为 0.000395，因而没有 cascade 特异的更新非交换性。当前最稳妥的术语是“任务选择性的有序交互现象”，而不是非正规、半群缺陷或更新算子攻击。

Round14 还提出了两个达到目标新意区间的不同候选。C2“正交任务 dynamics action-Jacobian desensitization”在原生 TD-MPC2 上做了 20 次 walker-stand 高频合法 dither 与 random 对照；reacher-easy 两个种子的回报比为 1.0，action-Jacobian Frobenius 比为 0.991，一步误差比为 1.016，6 维有效秩没有变化，故立即否决。C1“TTT context-rank parasitism”评分 7.4，随后完成了真实依赖安装和 MuJoCo EGL 冒烟测试：使用释放的 WorldAgen checkpoint、LIBERO_10 task0、10 个合法振荡探测步和 20 次 causal-backbone 自监督更新。一步损失比为 0.711，但隐藏表示谱熵只下降 28.1%（门槛是超过 50%），且六步 baseline 没有成功，因此完成率下降门槛也没有定义成功。上游评测代码引用的 `test_time_training/ttt_libero.py` 在当前提交中缺失；这次是透明的 native-style 短筛选，不把它升级为完整复现或正证据。Round15 官方 Gemini 又给出两个表面上可运行的方向：dynamics-nullspace gradient inversion（7.3）直接落入已否决的 gradient-integrity 族且需要默认不存在的梯度规划器；Dreamer RSSM predictive-horizon contraction（7.2）依赖本地没有的 Dreamer 载体，并与普通 mode collapse/rollout compounding 重叠。两者都没有启动 GPU。Round17 针对 TD-MPC2 的 SimNorm、min-Q head、task embedding 和 MPPI 组件做了架构特异搜索，官方 Gemini 返回 REJECT_ALL：剩余交互都退化为已否决的负迁移、潜空间/UQ 漂移或 CEM 分散。Round19 将 actor-restore 结果作为硬约束交给官方 Gemini 严格复核，返回 REJECT_ALL、评分 2.1/10：cascade 在两个 seed 上变号，评审认为它与默认 MPC 采样方差相容，不能独立于普通迁移抖动。评审建议的五 seed 显著性测试只是终止确认门，不构成新候选，因此不启动扩展。Round18 对 ScienceWorld 的“预测权威倒置”复核给出 4.5/10；它与 BadWAM、TRAP、合法交互投毒、检索偏置和 automation bias 重叠，现有单实例任务损失也不足以证明自确认吸引子。因此报告保持条件性 TD-MPC2 现象，不虚构 novelty 7 的最终机制。Round16 将方向改为防御候选。两步 latent shooting commutator penalty 在 20 次更新后确实把 cascade 的局部交换子降到约 7.3×10^-9，但 walker-run seed31 仍为 219.18 对 random 的 237.65，反序也下降到 212.93，未形成选择性恢复。action-Jacobian 条件数防御在 kappa=5 时完全没有触发；降低到 kappa=3 后虽然介入，一步误差比为 1.031，但 clean/random 比例上升到 1.13/1.11，cascade 也出现 11.7% 的 seed32 损失，故不能作为防御或主机制。

## 相关工作边界

该候选需要和已有的隐蔽世界模型投毒（[SWAAP](https://arxiv.org/abs/2606.18697)）、尾部轨迹排序后门（[TRAP](https://arxiv.org/abs/2605.01950)）、跨动作物理一致性（[OneWorld](https://arxiv.org/abs/2609.30946)）以及动作条件反事实一致性（[CoCo](https://arxiv.org/abs/2608.04653)）区分。当前证据只显示真实有序交互与规划时任务损害的条件关联，尚未证明一种独立于普通跨任务负迁移的内部安全原语。

## 当前结论与升级条件

当前可以保留的结论是：在官方 TD-MPC2 的 walker-run 任务族、固定原生短视 MPC 和真实有序 source curriculum 下，存在一个低一步误差但有明显闭环回报损失的候选现象。不能写成“所有共享世界模型都会受到该攻击”，也不能写成“非正规性已被因果证明”。

要升级为 ICLR 级主张，仍需在第二个独立 carrier 或任务族上完成双种子、随机、时间反序和 shuffled 对照；证明一个不改变 clean/random 性能的机制干预能够修复 cascade 损失；并报告强基线、跨种子统计和至少一个部署时防御。当前没有满足这些条件，因此不把稿件标为完成，也不启动长 GPU 扩展。

## 可复核证据

- [官方 TD-MPC2 walker-run/cascade 结果](../runs/tdmpc_cascade_heldout_matched_v1/status.json)
- [级联候选总配置与边界](../configs/world_model_non_normal_screen.json)
- [walker-like 独立门槛](../runs/tdmpc_non_normal_walker_like_gate_v0/status.json)
- [规划时干预算子诊断](../runs/tdmpc_intervention_operator_v1/summary.json)
- [动作顺序与 actor 对照](../runs/tdmpc_order_controls_v0/status.json)
- [单步训练窗口控制](../runs/tdmpc_window_composition_h1_v0/status.json)
- [Q-only value 头控制](../runs/tdmpc_value_provenance_v0/status.json)
- [hopper 跨任务顺序控制](../runs/tdmpc_order_controls_walkerback_hopper_v0/status.json)
- [cheetah source seed22 复测](../runs/tdmpc_order_controls_cheetah_seed22_v0/status.json)
- [source-target 交叉矩阵](../runs/tdmpc_cascade_matrix_v0/status.json)
- [多时域 latent 几何诊断](../runs/tdmpc_horizon_geometry_v0/status.json)
- [固定候选的规划排序诊断](../runs/tdmpc_planner_ranking_v3/status.json)
- [更新算子非交换性诊断](../runs/tdmpc_update_noncommutativity_v0/status.json)
- [native full-update 强基线](../runs/tdmpc_full_update_order_controls_v0/status.json)
- [actor-restore 因果强基线](../runs/tdmpc_full_update_actor_restore_v0/status.json)
- [有序主张严格复核](../runs/gemini_claim_review_round13_20261010T0000Z/review.json)
- [Round14 候选检索](../runs/gemini_mechanism_search_round14_20261010T0000Z/review.json)
- [C2 action-Jacobian 原生门槛](../runs/tdmpc_action_jacobian_desensitization_v0/status.json)
- [C2 筛选配置](../configs/world_model_action_jacobian_desensitization_screen.json)
- [C1 WorldAgen/LIBERO TTT context-rank 短筛选](../runs/worldagen_ttt_context_rank_v0/ttt20.json)
- [C1 短筛选脚本](../scripts/pilot_worldagen_ttt_context_rank.py)
- [跨反事实分支耦合严格评议](../runs/gemini_counterfactual_coupling_search_20261010T0000Z/review.json)
- [Round15 严格候选搜索](../runs/gemini_mechanism_search_round15_20261010T0000Z/review.json)
- [Round16 防御候选复核](../runs/gemini_defense_search_round16_20261010T0000Z/review.json)
- [Round17 TD-MPC2 搜索](../runs/gemini_mechanism_search_round17_20261010T0000Z/review.json)
- [Round18 预测权威复核](../runs/gemini_mechanism_review_round18_20261010T0000Z/review.json)
- [Round19 actor-restore 严格复核](../runs/gemini_mechanism_search_round19_20261010T0000Z/review.json)
- [Round20 provenance projection 复核](../runs/gemini_mechanism_search_round20_20261010T0000Z/review.json)
- [Round21 trajectory support 复核](../runs/gemini_mechanism_search_round21_20261010T0000Z/review.json)
- [Round22 CLCI 复核](../runs/gemini_mechanism_search_round22_20261010T0000Z/review.json)
- [CLCI 原生短门](../runs/tdmpc_clci_gate_v0/status.json)
- [Round23 CLCI 排斥复核](../runs/gemini_mechanism_search_round23_20261010T0000Z/review.json)
- [Round16 commutator gate](../runs/tdmpc_commutator_defense_v0/status.json)
- [Round16 spectrum gate](../runs/tdmpc_spectrum_defense_kappa3_v0/status.json)
- [候选路线图](../configs/research_roadmap.json)
- [自动生成研究进展报告](research_progress.html)
