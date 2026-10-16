# 指令条件世界模型的共享动作因果总线：研究报告

> **历史研究报告，正文中的旧主张已被后续专项审计限制或替代。** 下文原样保留研究过程，其中“当前主张”、旧预测/规划结果及 novelty 7/10 自评不代表当前结论，也不能直接复用于论文。当前支持范围以[证据索引](research_archive/evidence.json)、[物理不变性规范](../configs/goal_bus_physical_invariance_spec.json)和对应专项审计为准；从[研究知识库](research_archive/index.html)查找可复用代码、结果及失败记录。归档告示不构成实验复现或旧结论重新验证。

## 摘要

论文式主稿草案见[goal-bus paper draft](goal_bus_paper_draft_cn.md)。

本项目研究一个可直接落到多任务机器人世界模型接口的威胁：攻击者只切换环境允许的任务指令，不修改图像、状态、已执行动作、奖励、模型权重或标签。当前主张是：**指令条件世界模型会把任务文本写入共享的动作 token 因果总线，使 goal 在直接 goal→world 掩码存在时仍能改变固定状态与固定动作下的物理未来预测；切分该总线可以恢复 goal-invariant world prediction，同时保留目标条件策略输出，暴露控制条件与动力学保真之间的安全权衡。** 这是一个自评 novelty 7/10 的机制主张，证据已经覆盖结构性 mask 路径、冻结发布权重、WorldAgen 30 个真实窗口、AlphaBrain 21 个固定动作 sampler 窗口、跨 offset 矩阵、任务语义分层和 held-out forecast tradeoff。

我们保留此前 TD-MPC2 轨迹条件反事实完整性的全部负结果，但不再把它当作主线。新的主线不依赖数据投毒、跨任务 online update 或 rollout compounding；它检验一个更基础的因果安全不变量：在物理状态和已执行动作固定时，改变任务描述不应改变世界模型对下一物理观测的预测。WorldAgen 的 30 个真实 LIBERO 窗口覆盖公开 LIBERO-10 数据集的 10 个 episode；换 goal 使原生 world prediction 相对 L2 平均变化 0.300，换为空指令为 0.270。切断 9 条原生 goal→input-action 边后，world 输出在所有窗口精确不变，而 policy 输出仍保持目标敏感（relative L2 平均 0.176）。

这个不变量的理由来自环境转移而不是攻击叙事：若 \(S_t\) 是完整物理状态、\(A_{t:t+H}\) 是实际执行动作、\(G\) 只表示任务偏好/奖励/终止条件，则在 `do(A=a)` 下应有 \(Y^{physical}\perp G\mid S_t,A_{t:t+H}\)。Goal 可以合法地改变 policy、value、reward、termination 和 goal-relative semantics。若模型只有部分观测，Goal 可能是隐藏状态的线索；若 Goal 改变 reset 或场景定义，上述 estimand 也不成立。因此必须先验证同状态同动作的真实 future 和物理 target，再讨论 Forecast Divergence、ranking flip 或攻击成功率。当前 120/120 同场景 crossing 通过了环境端控制，但 partial-observability sufficiency 与 state/contact/proprio target 仍是开放门槛。

跨 backbone 的机制和 forecast-level 复现已经通过，但任务级成功率仍未建立。WorldAgen 扩展窗口中的记录目标、换目标和 edge-cut primary patch MSE 平均为 1.423、1.496、1.484，修复带来约 4.3% 的记录目标预测代价；30 个配对窗口中换目标的 primary MSE 全部上升。相同 WorldAgen 图像、state 和 goal 下替换同 episode 六步真实动作 chunk，30 个窗口的 cut action→world relative L2 均值为 0.1655，且 30/30 非零。独立的 AlphaBrain CosmosPolicy 的扩展 held-out matrix 覆盖七个真实 episode、三个时间点共 21 个固定图像/state/action 窗口；native world 差异均非零，均值 0.05317，episode-cluster bootstrap 95% CI 为 0.04512–0.06180，fixed-action world cut 在 21/21 窗口精确恢复 world invariance，并在三窗口 action control 中保留非零 action→world sensitivity（均值 0.08715）；策略输出由并行 native policy branch 保留。seed 1 的 21 窗口 repair delta 为 +0.04899（episode-cluster CI 0.03321–0.06798），独立 seed 2 和 seed 3 的三个窗口重复仍保持 exact invariance，但 repair delta 分别为 −0.00079 和 −0.03820，因此 forecast cost 是条件性结果，稳定主张是 invariance 与 action sensitivity。另有一个冻结 DiT/VAE、只学习共享无指令 context 的 disjoint-episode 对照，两个随机种子将 one-step held-out MSE 从 0.58519 降到 0.46215 和 0.46219；它是受控 forecast comparator，不是完整 goal-free dynamics head。上述结果支持跨载体的预测层机制，但不把短闭环 0/3 结果写成任务攻击成功。

| 载体与证据 | 物理窗口／重复行 | native world 敏感性 | edge-cut world | edge-cut 后 policy | 预测代价 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WorldAgen 真实输入 goal swap | 30 | rel. L2 0.2999 | 0 | rel. L2 0.1760 | swapped-recorded MSE +0.0732 |
| AlphaBrain 真实输入单步 | 7 | 0.03105 | 0 | 0.01629 | 未测 held-out target |
| AlphaBrain sampler offset 1/5/9 | 3 / 9 行 | 0.07452 | 0 | native policy branch | 未测 held-out target |
| AlphaBrain expanded held-out forecast | 21 / 7 episodes | native/cut MSE matrix | fixed-action invariance 21/21 | native policy branch | repair +0.04899，episode CI [0.03321, 0.06798] |

完整的机器可读聚合见[goal-bus 主结果](../runs/goal_bus_main_results_v0/summary.json)；逐项命题、判据和命令见[复现与证据索引](goal_bus_reproduction_manifest_cn.md)。重复 offset 的区间按物理 episode 聚类；zero-context MSE 0.60833 只作 prompt-independent 诊断，不当作训练过的 goal-free dynamics 基线。旧的 full edge-cut 曾同时切掉 world→action/value self-attention，保留为过强 legacy comparator；fixed-action world cut 才是本文主修复。

当前主线的威胁模型只涉及冻结模型的预测层：在相同图像、状态和已执行动作下切换合法 task text，测量 `Δ_g^W=d(F_θ(g),F_θ(g'))`；再用 matched fixed-action cut 检查该效应是否恢复到零，并用 policy goal sensitivity 或 alternate-action `Δ_u^W` 检查控制条件是否仍保留。WorldAgen 的同 cardinality policy-only edge cut 仍保留 world relative L2 0.2999，而 input-action carrier cut 为精确零；同 episode alternate-action control 在 cut 后仍有 0.1655 的 world relative L2，且 30/30 非零；activation patch 额外报告 `ρ_l^W=d(F_{B←A;l},F_A)/d(F_B,F_A)` 及 policy residual，用于定位中间 action carrier。native `Δ_g^W>0`、cut 后 `Δ_g^W≈0`、action sensitivity 非零是主判据；held-out MSE repair 只作为 sampler 条件下的次级 fidelity 量。下面保留的 TD-MPC2、轨迹条件和其他候选记录属于历史筛选档案，不改变当前主线的威胁模型或结论。

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
| reacher-easy，seed31/32 | 相同 cheetah-run source、300 update、原生 MPC 与顺序对照 | clean/random/cascade/reverse/shuffle 的 MPC 回报均为 572/579；cascade 一步误差为 random 的 0.828 倍，但没有任务损害或顺序选择性 |

动作顺序的专门对照进一步把现象限定在正向 cascade：在 walker-run 的 500 步回合中，cascade 为 495.22/547.53，时间反序为 608.43/621.95，保持完整动作多重集的乱序为 613.76/669.45；四个条件的 actor 回报完全相同，因为只更新 dynamics。一个 4×4 source-target 矩阵显示，cheetah-run、walker-walk 和部分 hopper 源都能影响 walker-run，而 walker-walk/stand 目标基本不变；只有 cheetah-run 源在两个种子上同时满足一步误差不超过 random 的 1.05 倍。

walker-run 的结果说明，单步均值误差并不能替代规划时的动作效果验证；但它尚未说明具体内部因果机制。此前尝试的非正规 Jacobian 解释、SimNorm 温度防御、固定标准差地板和选择性 horizon 干预没有形成有效因果隔离。actor-only 介入可以改变结果，但同时改变了规划协议，不能作为机制防御证据。

作为强基线，使用相同 source 但改为 upstream native full update（80 次，包含 encoder、reward/value、task embedding 和 policy）后，walker-run cascade 相对 random 仍下降 19.0%/13.0%，而时间反序仅下降 6.0%/0.5%，乱序为持平或上升；cascade 一步误差为 random 的 0.993 倍。这个结果说明顺序效应不只依赖 dynamics-only 冻结设置，但 full update 同时改变 actor，因此不能把它解释为纯 MPC 失真。full update 的 cup-catch seed32 也出现 27.9% 下降，故 cup-catch 只能作为 dynamics-only 协议下的负对照，不能作为所有适配范围的安全控制。 随后将每个 full-update 条件的 actor 恢复为官方原始参数，仅保留更新后的模型做 300 步 MPC：cascade 相对 random 在 seed31/32 为 −4.6%/+8.2%，时间反序为 +0.2%/+17.1%，一步误差比为 0.993。因此先前 full-update 的双种子损失主要依赖 actor/value 等联合更新；actor 恢复后只剩弱的单种子方向，不能作为 planner-only 因果证据。

一个较弱但一致的因果定位是规划时域：在 300 步短回合中，cascade 的原生 horizon=3 回报为 300.63/306.34，而将同一适配模型改为 horizon=1 后为 376.16/374.34；random 对照分别为 362.88/372.49 和 364.83/381.70。这个干预消除了 cascade 相对 random 的方向性损害，但它改变了规划器协议，属于 planner-specific 定位和任务局部缓解，不是可直接推广的防御。

为检验“规划排序”而不是单纯 latent 漂移，固定 walker-run 初始状态和 512 条合法三步候选序列，比较模型价值排序与 simulator 三步回报。cascade 在 seed32 的 Spearman 相关为 0.071，random 为 0.291、时间反序为 0.155、乱序为 0.354；seed31 的 cascade 为 0.589，和对照接近。因此该排序诊断只作为一个种子上的支持性证据，不能单独承担机制结论。

## 轨迹条件反事实完整性：新增控制与边界

为区分“多步训练窗口”与更宽的轨迹条件效应，使用完全相同的 source 和 dose，将 dynamics-only consistency window 从原生 3 步改为 1 步，部署仍恢复原生 horizon=3 MPC。cascade 相对 random 在 walker-run 两个 seed 上下降 12.4%/16.0%，时间反序下降 6.9%/3.6%，动作多重集乱序为基本持平；cascade 一步误差比为 1.011，actor 保持不变。结果排除了“必须依赖多步 consistency loss”的解释，但没有单独证明反事实支持机制。

三个机制隔离没有通过。只更新共享 Q 头并冻结 encoder、dynamics、reward、task embedding 和 actor 后，cascade 相对 random 为 −6.5%/+4.0%，反序为 −0.1%/+3.9%，乱序为 +0.1%/+0.7%，所以没有顺序选择性的终端价值污染。Round20 提出的 shared-latent projection 短门中，cascade 的 walker-run 回报高于 random，目标五步 latent power norm 几乎不变；Round21 严格评议把更宽的 trajectory-manifold/support 重写评为 4.0/10，认为它仍可能是普通非 IID occupancy 或共享 trunk 干扰。 Round22 提出的 Cross-morphology Latent Contraction Invariant（CLCI）虽被 Gemini 评为 7.5/10，但其原生短门直接否证了中介：cascade 的 source→target top-3 action-sensitivity 子空间对齐为 0.489，random 为 0.769；三步 target branch dispersion 为 2.110，random 为 2.098。 Round23 对 CLCI 的反向“子空间排斥”解释也返回 REJECT_ALL、4.0/10：cascade、reverse、shuffle 的回报没有系统排序，branch dispersion 基本不变，不能把对齐差异重新命名为新安全机制。报告因此停止在 TD-MPC2 上继续换名扩展几何候选。反序与乱序也没有预测的高对齐/收缩模式，因此不运行 PNDR 防御，也不把 7.5 的 idea 分数写成实验新意。

另有一条独立但范围更窄的采样器路线：在 MBRL-Lib 的连续 CartPole 中，rank-coupling sampler 对每个候选动作保留完整的 native 预测边缘分布，只在候选之间改变 bank index 的联合耦合。旧的两批 seed 共四个配对回合中，targeted 回报均低于 independent（均值 97 对 180）；更强的 HalfCheetah、Pusher 和 TD-MPC2 门槛没有复现损害，所以这不是当前主张的跨任务证据。作为一个可检验的防御，对每个 CEM 评估把同一预算的八个 native forecast bank 全部取均值；在未使用的 CartPole seed706/707 上，两回合都达到 200 步上限，模型 hash 保持不变，共 144,000,000 个预测转移。这个结果支持“对候选独立化或做 bank averaging 可缓解该 CartPole sampler 操作”的局部结论，不能外推为通用防御或 ICLR 级主机制。
为寻找第二个随机连续控制载体，我补上了 MBRL-Lib 原生 Reacher3D 的 reward、模型维度和 CEM 适配，使用 3000 条真实随机交互训练五成员 GaussianMLP。强设置（20 particles、8-bank、100 候选、horizon15、50 步）下，Reacher3D 两个 seed 的 independent 回报为 −60.25/−60.89，targeted 为 −59.49/−54.56；targeted 两个 seed 都改善，未出现攻击损害。该结果把 rank-coupling 限定为 CartPole 条件现象，并关闭当前路线的进一步扩展。

Round26 在 reacher-easy 负门槛之后再次要求 Gemini 只保留一个可在公开载体上两分钟内证伪、具有机制观测量和修复方案的新操作，结果为 REJECT_ALL、novelty 1/10。评议把剩余 cheetah-run→walker-run 单点现象归为 source-seed-specific negative transfer、表示错配或规划随机性，不能再把它包装成稳定安全原语。因此 TD-MPC2 载体路线在这里封存，保留所有反例和脚本供后续真正不同的载体或操作使用。

随后两轮严格机制检索没有找到可直接升级的第三条路线。Round24 提出的 normalizer 统计量失同步要求当前载体不存在的在线统计量更新；MPPI warm-start 和温度/ESS 候选分别落入已测的热启动状态干扰、奖励尺度或采样退化。Round25 的 bootstrap-correlation planning poisoning 只得到 5.0/10：它在概念上等价于 bootstrap overlap、ensemble diversity collapse 或跨任务负迁移，除非能同时证明目标边缘方差不变、反事实区域的跨成员残差相关性增加、且只有非线性 CEM 排序受损。按停止规则没有为这些候选启动 GPU。

WorldAgen 的另一个结构性观察也被单独收窄：原生 mask 虽然禁止 goal 直接连到 world 输出，但三段真实 LIBERO 片段中，换 goal 会使冻结模型的 world prediction 相对变化 0.142–0.381；切断 goal 到 input-action token 的三条边后，world prediction 对 goal 完全不变，而 policy 输出仍可依赖 goal。由于当前释放版 LIBERO 基线没有成功回合，严格评议将其定为单 checkpoint 的注意力掩码缺陷（4.5/10），不是已建立的世界模型安全攻击。

进一步检查 wrapper 的任务切换边界：释放版 `WorldAgenLiberoModelWrapper.step_agent` 只在 `text_queue` 为空时写入目标，后续调用即使传入新任务也继续使用首次目标，除非调用未接入主评测流程的 `reset_text()`。在三个真实 LIBERO 片段上，模拟“首次目标 A、第二次请求目标 B”的 stale-queue 条件，stale 输出与 fresh-A 输出逐元素相同；相对 fresh-B，world prediction 的相对 L2 为 0.142–0.318，policy 输出为 0.073–0.255。该结果确认了一个可复现的 wrapper 级任务切换残留，但没有环境步、成功率或任务损害证据，因此只作为实现边界记录，不升级为安全攻击或论文主机制（[短门结果](../runs/worldagen_stale_goal_queue_v0/summary.json)）。

独立任务族和 source-seed 对照也限制了范围。walker-walk-backwards→hopper-hop 的 cascade 损失为 18.0%/24.8%，但 shuffled 在 seed31 也损失 17.2%，反序在 seed31 更差而 seed32 反转；不能通过顺序选择性门。把 cheetah cascade 换到 source seed22 后，walker-run cascade 相对 random 反而提升 10.7%/9.4%，反序和乱序也不稳定。因此当前名称是条件性的“轨迹条件反事实完整性候选”，不写成 source-independent 或 cross-carrier 攻击。
随后把同一 source seed23 的 cascade 迁移到官方 TD-MPC2 的 reacher-easy，保持 300 步 source、300 次 dynamics-only update、原生 horizon=3 MPC，并加入 clean/random/reverse/shuffled 对照。两个评估种子上五个条件的回报都严格为 572/579；cascade 一步误差反而是 random 的 0.828 倍，但没有任何闭环损害。这个独立任务族门槛关闭了当前 TD-MPC2 的跨任务扩展：现象只能保留为 cheetah-run→walker-run 的条件性候选，不能声称共享世界模型的通用攻击。

## 新诊断与候选淘汰
Round27 在封存 TD-MPC2 后提出跨载体“反事实来源阴影”：合法边界动作前缀把 Pusher 状态推入训练分布外区域，理论上应出现低 ensemble 分歧而高真实 rollout 误差，并损害原生 CEM。实际短门使用随机训练的五成员 GaussianMLP、原生 ModelEnv/CEM、两个种子和 clean/random/interior/boundary 对照。boundary 的 rollout RMSE 均值为 1.213，确实高于 clean 的 0.716，但一步 ensemble 分歧也从 0.00496 升到 0.00798，且 25 步回报为 −46.29，高于 clean 的 −46.82。它没有满足“低不确定性掩盖高风险”或任务损害门槛，归类为普通 OOD/model error，候选不升级。
Round28 的动作间反事实可分性与证据来源候选也被严格否决（3.5/10）：在现有载体上，它们分别退化为协变量偏移、遗忘、Jacobian desensitization 或动作归因探针。Round30 对“action-null twin/gap”做了独立复核，评分4.1/10，结论同样是条件互信息/动作敏感性测量，不构成新的安全原语。备用 gateway 的 Round29 请求返回 HTTP 200 但只有隐藏 reasoning、没有可用正文，因此不计为候选证据。

Round31 将剩余的时间有序机制迁移到独立的 MBRL-Lib 五成员 GaussianMLP/CEM。Pusher 上使用 120 条合法 phase-locked source、相同动作多重集的 shuffled source、random source 和 clean，对每个条件做 5 次顺序更新。15 步 native CEM 回报为 phase −35.25、shuffled −36.71、random −35.13；phase 一步 RMSE 为 0.0698，低于 clean 的 0.0970，没有 phase-specific 损害。非递归 ensemble 不复制 TD-MPC2 的时间效应，因此不扩展到更多任务。

Round32 尝试用备用模型接口做独立的新信任边界构思，但接口返回 HTTP 200 的 reasoning-only 响应，没有可用正文；该请求已记录为 API 输出失败，不作为研究证据。



近期工作将“当前状态可读但动作后果传播错误”定义为 planning-time intervention fidelity（[原始论文](https://arxiv.org/abs/2608.29998)）。本项目把它改写成攻击候选，并在 64 个固定 walker-run 根状态上执行相同物理动作对。冻结 encoder 后，三臂 root-capture MSE 都是 18.4023；干预余弦为 clean 0.7858、random 0.7603、cascade 0.7771；干预对比 MSE 为 clean 9.7611、random 10.8504、cascade 10.2094。cascade 没有显示相对 random 的特异算子偏差，所以不能用该诊断解释 walker-run 回报损失。

严格的 Gemini 机制审查还拒绝了 prototype-router hijacking、counterfactual-branch consistency、action-operator rotation 和跨反事实分支耦合四个方向；这些意见只用于候选筛选，不作为实验事实。最后一项只得到 4.5/10：现有 TD-MPC2/PETS 载体按点或独立边缘建模，没有跨分支联合头，无法在保持所有分支边缘时单独制造可验证的联合耦合差异。随后的 Round12 又审查了 recurrent latent deadlock、PETS disagreement quenching 和 SimNorm directional depletion；最高只有 5.4/10。Round13 对当前有序印刻主张的严格复核为 5.5/10，指出它仍可能是非 IID continual interference 与规划误差累积的组合，尚未达到目标新意。

两个机制诊断进一步收窄了结论。固定 walker-run 真实轨迹的 horizon=1/3/5/10 latent drift 中，反序和乱序条件有时比 cascade 更大，却没有同等回报损失；同一批 native dynamics windows 的 A→B 与 B→A 小步更新差异，cascade 的归一化均值为 0.000323，random 为 0.000395，因而没有 cascade 特异的更新非交换性。当前最稳妥的术语是“任务选择性的有序交互现象”，而不是非正规、半群缺陷或更新算子攻击。

Round14 还提出了两个达到目标新意区间的不同候选。C2“正交任务 dynamics action-Jacobian desensitization”在原生 TD-MPC2 上做了 20 次 walker-stand 高频合法 dither 与 random 对照；reacher-easy 两个种子的回报比为 1.0，action-Jacobian Frobenius 比为 0.991，一步误差比为 1.016，6 维有效秩没有变化，故立即否决。C1“TTT context-rank parasitism”评分 7.4，随后完成了真实依赖安装和 MuJoCo EGL 冒烟测试：使用释放的 WorldAgen checkpoint、LIBERO_10 task0、10 个合法振荡探测步和 20 次 causal-backbone 自监督更新。一步损失比为 0.711，但隐藏表示谱熵只下降 28.1%（门槛是超过 50%），且六步 baseline 没有成功，因此完成率下降门槛也没有定义成功。上游评测代码引用的 `test_time_training/ttt_libero.py` 在当前提交中缺失；这次是透明的 native-style 短筛选，不把它升级为完整复现或正证据。Round15 官方 Gemini 又给出两个表面上可运行的方向：dynamics-nullspace gradient inversion（7.3）直接落入已否决的 gradient-integrity 族且需要默认不存在的梯度规划器；Dreamer RSSM predictive-horizon contraction（7.2）依赖本地没有的 Dreamer 载体，并与普通 mode collapse/rollout compounding 重叠。两者都没有启动 GPU。Round17 针对 TD-MPC2 的 SimNorm、min-Q head、task embedding 和 MPPI 组件做了架构特异搜索，官方 Gemini 返回 REJECT_ALL：剩余交互都退化为已否决的负迁移、潜空间/UQ 漂移或 CEM 分散。Round19 将 actor-restore 结果作为硬约束交给官方 Gemini 严格复核，返回 REJECT_ALL、评分 2.1/10：cascade 在两个 seed 上变号，评审认为它与默认 MPC 采样方差相容，不能独立于普通迁移抖动。评审建议的五 seed 显著性测试只是终止确认门，不构成新候选，因此不启动扩展。Round18 对 ScienceWorld 的“预测权威倒置”复核给出 4.5/10；它与 BadWAM、TRAP、合法交互投毒、检索偏置和 automation bias 重叠，现有单实例任务损失也不足以证明自确认吸引子。因此报告保持条件性 TD-MPC2 现象，不虚构 novelty 7 的最终机制。Round16 将方向改为防御候选。两步 latent shooting commutator penalty 在 20 次更新后确实把 cascade 的局部交换子降到约 7.3×10^-9，但 walker-run seed31 仍为 219.18 对 random 的 237.65，反序也下降到 212.93，未形成选择性恢复。action-Jacobian 条件数防御在 kappa=5 时完全没有触发；降低到 kappa=3 后虽然介入，一步误差比为 1.031，但 clean/random 比例上升到 1.13/1.11，cascade 也出现 11.7% 的 seed32 损失，故不能作为防御或主机制。

## Round33：联合支持空洞候选的合成通过与 native 否证

为寻找与时间顺序、采样 copula、OOD 和普通遗忘不同的载体，Round33 定义了一个更严格的联合支持候选：攻击者只根据真实可见上下文选择合法动作，使上下文边缘和动作边缘保持平衡，但让部分上下文×动作组合从未出现。合成上下文动力学的三 seed 短门通过了预设条件：支持空洞臂的上下文/动作均值仍平衡，已观测组合的误差为 clean 的 0.835 倍，缺失组合误差约为已观测误差的 3663 倍，规划目标误差比约为 9.05；随机置换联合配对对照没有这种差异。这说明“保留边缘统计而删除干预联合支持”在一个可控系统中确实能构成可检验的机制。

随后将同一操作放入原生 MBRL-Lib 的 GaussianMLP、ModelEnv 和 CEM，并加入两个上下文位与交替相位抵消物理状态漂移。三 seed 的动作边缘、状态幅度和缺失联合支持门均通过，但 one-step 缺失/已观测误差比只有 1.063，已观测误差本身为 clean 的 1.208 倍；support-hole CEM 回报低于 clean，但与拟合差异和原生小型上下文载体耦合，不能通过“局部预测保持不变、反事实规划单独受损”的门槛。因此该候选保留为合成机制正例和 native 载体否证，不升级主 claim，也不启动 GPU 扩展（[合成短门](../runs/joint_support_hole_gate_v1/summary.json)，[native MBRL 短门](../runs/mbrl_joint_support_hole_gate_v3/summary.json)，[脚本](../scripts/pilot_joint_support_hole_gate.py)、[native 脚本](../scripts/pilot_mbrl_joint_support_hole.py)）。

## Round34：合法作用范围泄漏候选的 WAV 原生短门

Round34 改用带公开 checkpoint 和 MiniGridPhysicsOracle 的 WAV 原生载体，测试“合法交互能否让世界模型在动作真实作用范围之外产生变化”。以相同 16 条更新预算比较两臂：一臂集中于真实 toggle 交互，另一臂按动作混合随机抽样；两臂都只使用数据集中的真实状态、动作和下一状态。评估在共同 103 条留出转移上进行，保护区域定义为 oracle 对该动作没有改变的格子。

基础 checkpoint 的保护区预测变化率为 0.1301、格子错误率为 0.1448；toggle 集中更新后分别为 0.1382 和 0.1537；动作混合对照为 0.1276 和 0.1427。集中更新相对对照只增加 0.0106 的保护区变化和 0.0111 的格子错误，且没有独立的任务规划回报门。因此“因果作用范围防火墙”在当前公开 WAV 载体上未通过 native 机制门，不能解释为选择性越界影响，也不启动 GPU 扩展（[结果](../runs/wav_footprint_gate_v1/summary.json)，[脚本](../scripts/pilot_wav_footprint_gate.py)）。

## Round38：ScienceWorld 反事实覆盖短门未复现选择性改写

为把“联合支持空洞”改写成可执行的原生测量门，使用 ScienceWorld 的未知摩擦任务。模型先接收一次真实 H 平面测量作为校准，然后给三臂各做 8 次真实合法更新：重复 H 测量、重复 C 测量、以及重复检查无关秒表。最后三臂都回到完全相同的 H 平面首测状态，查询同一个测量动作；因此评估状态和真实 H 结果不随 source arm 改变。

三臂均预测 H 平面首个读数为 1%，与模拟器真实结果一致；共同评估状态 hash 相同，source 更新预算均为 8，C 与无关秒表臂的预测差异门未触发。这说明在当前 WMEpisodicSemantic/Qwen3 原生载体和该剂量下，真实 C 轨迹没有把已校准的 H 反事实测量改写。该短门没有 actor 任务成功率或 GPU 结果，因此只作为对“分离动作覆盖”候选的否证，不启动剂量扩展（[结果](../runs/scienceworld_counterfactual_coverage_v1/summary.json)，[脚本](../scripts/pilot_scienceworld_counterfactual_coverage.py)）。

## Round41：A2World 原生去噪器暴露序列级动作非局部性

前面的 MoSim 和 WorldAgen 短门分别给出了两个重要边界：严格的一步模型对未来动作不敏感，带因果块掩码的 WorldAgen 首个世界块也不读取后缀动作。为检查这是否只是载体限制，本轮转向公开的 A2World 2B 全序列动作条件去噪器，并直接加载发布权重进行固定噪声潜变量前向。

在相同初始潜变量、时间步、相机布局和零上下文下，保持动作第 1--4 步不变，只把第 5--20 步反转。输出潜变量形状为 `[1,16,18,4,4]`；输出时间槽 1--6 的平均绝对差为 `0.1779199`，时间槽 7--18 为 `0.1754707`，因此早期槽与后期槽都发生变化，前缀不再是精确不变量。更严格地只翻转第 20 步动作，早期槽仍有 `0.0701187` 的平均绝对差，后期槽为 `0.0626189`。动作条件头的载体探针也显示，后缀变化会改变全局展平动作嵌入、AdaLN 条件和 cross-attention 条件（相对差分别为 `0.0269`、`0.9056`、`1.2513`）。结果见[固定噪声前向](../runs/a2world_fixed_noise_suffix_v0/summary.json)、[条件载体探针](../runs/a2world_global_action_condition_v0/summary.json)和[脚本](../scripts/pilot_a2world_fixed_noise_suffix.py)。

这支持一个暂定的机制表述：带全局动作广播或双向跨模态注意力的序列世界模型，可能学习的是“整段动作到整段未来”的条件场，而不是按时间索引的动作到状态映射；所以只改晚动作即可改写早时刻的预测。它与前两种载体的否证边界不同，也比单纯的规划时采样耦合更接近可定位的内部机制。OpenWAM 的公开掩码实现提供了可操作的比较轴：`mutual`/`video_sees_action` 允许后续视频行读取全部动作，而 `isolated`/`action_sees_video` 则切断视频行到动作的读取；本轮的[掩码语义审计](../runs/openwam_temporal_mask_audit_v0/summary.json)复现了这一差异。

这仍不是最终安全主张。当前证据只到固定噪声潜变量级，A2World 的 Cosmos tokenizer 在本环境需要 gated 权限，因此还没有解码视频、核心任务损害、修复收益或跨任务复现。路线图中的下一道门是获得可用解码器或建立可审计的决策代理，并在至少两个核心任务上比较全局条件、前缀局部条件和 OpenWAM 因果掩码；在该门通过前，不扩展长实验，也不把本轮结果写成 ICLR 级广泛攻击。

Gemini 的严格复核将该候选评为 `2/10`：它认为目前的现象是双向 cross-attention、全局动作条件和未加时间因果掩码的直接结果，不能单独构成新机制或安全漏洞。这个判断与本报告的停止条件一致。因此 Round41 只保留为一个可复核的载体观察和比较轴；若要继续，必须通过闭环 MPC/策略短门，证明后缀扰动会改变早期安全关键决策并导致失败，而且严格因果基线不会出现同样现象。复核记录见[Round41 Gemini 评议](../runs/gemini_round41_broadcast_temporal_nonlocality_20261010/review.json)。

## Round52–63：放宽威胁模型后的候选仍未达到新意门

Round51 的“合法输入、无梯度/参数/标签访问、单步近正常、必须有任务损害”的约束过紧，因此 Round52 暂时允许少量数据投毒、恶意可执行输入或共享服务攻击，并要求仍然有 world-model 内部机制和公开载体。候选“latent curvature poisoning 导致 CEM/MPPI 协方差拓扑坍缩”在提案阶段得到 8/10，但严格复核为 3/10：它等价于动力学对抗投毒、采样器协方差 pinching 和标准谱正则，未形成独立机制（[提案](../runs/gemini_round52_relaxed_frontier_search_20261010/review.txt)，[复核](../runs/gemini_round53_latent_churn_strict_20261010/review.json)）。

随后几轮对不同信任边界做了独立严格筛选：固定时间抽象下的动作时序/相位攻击为 4/10，属于异步控制和已有时间抽象失配；保持 imagined future 在自然流形上却越过安全 oracle 为 4/10，属于流形约束分类器攻击；CEGIS 的 revision-feedback privilege 为 4/10，属于规范/数据投毒；单调加法式 revision hysteresis 为 6.5/10，但依赖人为禁止回撤的脆弱提示约束，范围窄且标准回归测试即可消除（[Round54](../runs/gemini_round54_timing_safety_strict_20261010/review.json)，[Round55](../runs/gemini_round55_on_manifold_oracle_strict_20261010/review.json)，[Round56](../runs/gemini_round56_revision_feedback_strict_20261010/review.json)，[Round57](../runs/gemini_round57_revision_hysteresis_strict_20261010/review.json)）。

Round58 以“可恢复性/干预证据”开放搜索没有在输出限额内形成可审查候选；Round59 的时序残差相消被评为 5/10，核心是非完整观测器面对多步扰动的经典积分漂移；Round60 的适配器合并完整性为 4.5/10，属于通用 LoRA 参数干扰；Round61 的双线性 transition cloaking 为 4/10，属于组合式后门；Round62 的反事实动作效果隐私侧信道为 5/10，属于主动系统辨识；Round63 的持久 latent spatial cache 写回为 5/10，接近动态 SLAM 的地图污染与状态滞后（[Round59](../runs/gemini_round59_temporal_residual_laundering_20261010/review.json)，[Round60](../runs/gemini_round60_adapter_composition_integrity_20261010/review.json)，[Round61](../runs/gemini_round61_bilinear_transition_cloaking_20261010/review.json)，[Round62](../runs/gemini_round62_counterfactual_privacy_sidechannel_20261010/review.json)，[Round63](../runs/gemini_round63_persistent_latent_cache_writeback_20261010/review.json)）。

这些结果不是把分数当实验结论，而是收紧了研究边界。近期综述明确把数据投毒、轨迹排序、想象完整性、反馈来源和终止/滚动安全都列为已有威胁族；近期工作还直接研究了动作写法不变性、动作一致性、反事实崩溃、跨动作物理一致性和世界模型/策略失败归因。因此继续换名会降低论文可信度。下一条允许进入短门的候选必须同时满足：不是已有攻击族的变体；机制能在一个公开载体中被因果干预；至少有两个核心任务的闭环损害；修复不是简单的输入过滤、权重正则、缓存清空或回归测试。当前不启动长 GPU 实验。

Round64 由开放式搜索提出了“扩散反演时序陷阱”，声称晚期去噪中的 temporal cross-attention 会被高频触发器重排，并给出 DIAMOND 作为公开载体，提案自评 8/10。但载体先审计即否证：DIAMOND 的官方实现是逐帧条件化的二维 U-Net，使用上一观测和动作的 AdaGroupNorm 以及可选空间 SelfAttention2d；采样器虽遍历噪声级别，却没有 temporal cross-attention、时序块注意力或时序位置基。因而提案的核心中介在该载体不存在，把它改成普通高频输入攻击又会落入 BadWorld/通用去噪破坏族。该审计没有启动 GPU（[源码审计](../runs/diamond_source_audit_round64/summary.json)，[原始提案](../runs/gemini_round64_unrestricted_architecture_gap_20261010/review.json)）。这次先审计载体再判断新意，避免把模型自动生成的架构细节当成事实。

Round65 对 Round64 的候选做了真实载体纠正：DIAMOND 被排除后，改用 A2World 的 Cosmos/DiT 源码和公开动作条件管线复核。严格评议仍给 4/10，认为“高频观测扰动只在晚期去噪破坏时序动作条件”没有可依据的优化机制，且与已有输入/想象攻击相同；提案要求的 late-step temporal reversal 尚无证据（[复核](../runs/gemini_round65_a2world_schedule_attack_strict_20261010/review.json)）。Round66 又把 A2World 的时间压缩改写为“压缩窗口内瞬时危险别名”，严格评议给 4/10，认定是标准帧跳过、动作分块和连续碰撞检测问题；端点相同不能把表示层丢失的信息变成新安全原语（[复核](../runs/gemini_round66_temporal_compression_hazard_alias_20261010/review.json)）。


Round67–68：latent CBF 提案在源码核对后降为普通启发式缺陷

Gemini Round67 将“RSSM latent contraction 使 HJ/CBF Lie 导数失真”自评为 9/10，但官方 `latent_cbf` 源码不走这条路径：部署过滤器从五步历史得到一个 Dreamer latent，给原始动作和 25 个均匀动作调用 `critic_old`，计算 `qvals / qvals.max()`，再按相对阈值保留动作；没有 Lie 导数或 CBF-QP。世界模型虽有 margin/continuation 头，但在该部署路径中未被用于过滤。源码审计因此否定了原机制和 pilot 的载体对应关系（[审计](../runs/latent_cbf_source_audit_round67/summary.json)）。

把候选纠正为“批内最大 Q 基准可被动作候选集合操纵”后，Round68 严格评为 `1/10`：这是除以最大值的直接算术性质，与普通 critic 标定、动作采样敏感性和代码级阈值 bug 相同；绝对阈值即可修复，不是 ICLR 级 world-model 安全原语（[复核](../runs/gemini_round68_relative_q_threshold_strict_20261010/review.json)）。本轮没有启动训练或 GPU。


## Round69：FACT 的动作权限不对称仍是已知 teacher-forcing 失配

为避免 Round50 合成 toy 把 FACT 的现象泛化过度，本轮先审计公开 FACT 源码，再评议候选“failure-only action-authority asymmetry under teacher-forced future supervision”。源码事实是：训练 token 布局为 `[state | ref | pred_action | gt_action | future_state | value | future_image]`；`pred_action` 段是 sink，future/value/image 不能读取它，而这些输出读取的是干净 `gt_action`。失败 episode 将 action loss mask 置零，但 future-state/value 仍训练，`failure_active` 还给 value 加惩罚。推理时 Stage 1 先由 state/ref/text 生成动作，Stage 2 再把这个生成动作当作 `gt_action` 去预测 future/value。审计记录见 [FACT 源码审计](../runs/fact_action_authority_source_audit_round69/summary.json)。

这确实形成了一个可测的训练—部署动作条件差异：训练中的未来/价值监督从未使用动作头实际生成的动作，失败样本还没有动作模仿梯度；部署时生成动作却成为未来/价值预测的条件。但是 Gemini Round69 严格评为 novelty `4.5/10`、机制特异性 `7/10`、范围 `4/10`、可操作性 `5/10`，判定 `REJECT`（[严格复核](../runs/fact_action_authority_source_audit_round69/review.json)）。评议认为这仍是两阶段模型中的 teacher-forcing exposure bias、失败样本动作过滤和多任务表征漂移的组合，不足以独立于已知训练—推理失配、离线策略分布偏移或数据投毒。

因此本轮不启动 FACT 权重下载或 GPU 训练。若未来继续，只能先证明共享表征中存在能由合法 failure 轨迹触发、且区别于直接移除动作监督和普通 OOD 动作的可重复后门式负迁移；否则不能把动作 sink 或 failure-only mask 升格为新的安全原语。

## Round70：开放搜索的两个高分提案均在载体审计后被否决

在 Round69 之后做了一次不限定单一载体的开放搜索，要求候选先绑定公开源码，并排除已有的投毒、OOD、teacher forcing、负迁移、rollout compounding、采样耦合、动作编码、时序压缩和普通实现缺陷等路线。Gemini 初始提出两个候选：TD-MPC2 的“forward-inverse consistency 几何正交脱节”和 DIAMOND 的“扩散去噪 timestep/action 频率混叠”。

对 TD-MPC2 官方源码的审计显示，`WorldModel` 只有 encoder、dynamics/reward/termination MLP、Gaussian policy prior 和 ensemble Q；训练是 latent consistency、reward/value loss 加单独的 policy update，没有 inverse model、forward-inverse 双路径目标、潜变量奇异向量投影或该提案所需的联合梯度路径。对 DIAMOND 的审计显示，动作只是离散 embedding 与 Fourier 噪声 embedding 相加后送入二维 UNet；采样器是 EDM sigma 的 Euler/Heun 步进，没有 temporal cross-attention、动作频率基底或时间步—动作频率混叠模块。源码事实见[Round70 载体审计](../runs/gemini_round70_open_frontier_search_20261010/source_audit.json)。

因此严格复核将两项分别评为载体匹配 `1/10`、`2/10`，新意 `3/10`，均为 `REJECT`（[Round70 严格复核](../runs/gemini_round70_open_frontier_search_20261010/strict_review.json)）。本轮没有 GPU 实验；这再次说明自动生成的高新意叙述必须先经过源码路径核对。

## Round71：ActSWM 的单零动作锚点盲区仍是普通对比学习缺陷

近期检索发现公开 ActSWM 仓库把“Context Collapse”作为动作敏感世界模型的核心失效，并在默认配置中用真实动作 rollout 与全零动作 rollout 的单一 cosine hinge（margin `0.3`）约束动作效果；额外的 batch 内错误动作 contrastive loss 默认关闭。源码审计还确认 CEM 最终按 terminal latent goal cost 比较动作序列，IDM 只从 latent transition 读出动作。审计记录见[ActSWM 源码审计](../runs/actswm_anchor_contrast_source_audit_round71/summary.json)。

据此提出的“single-anchor action-effect blind spot”是：非零动作之间可能坍缩到同一 latent 等价类，同时与零动作保持距离，令单锚点评估正常但 CEM 无法排序真实候选。严格复核判为 `REJECT`，并指出它就是不足负样本、普通 contrastive 几何退化和 ActSWM 已明确讨论的 Context Collapse；下游规划失败也只是动作不敏感的直接表现（[Round71 严格复核](../runs/actswm_anchor_contrast_source_audit_round71/review.json)）。本轮没有训练或 GPU 实验。

## Round72–73：反事实拓扑投毒暂时只能作为评测候选

Round72 的两个新提案先经过原生载体审计即被否决。TD-MPC2 实际没有提案所需的反事实可达性度量；所给欧氏三角 hinge 由三角不等式恒为零。DreamerV3 代码也不在当前工作区，且动作对易子已在前轮被排除。两项严格复核的新意分别为 `3/10` 和 `2/10`，均为 `REJECT`（[Round72 审计与复核](../runs/gemini_round72_security_property_search_20261010/)）。

Round73 将问题收紧为一个可以短门检验的评测主张：**在基于采样的隐空间世界模型规划中，合法数据注入可能保持 on-policy 单步拟合与执行策略表面行为，同时改变未执行分支的局部价值排序，诱发 CEM/MPPI 选择错误分支。** Gemini 严格评议给出 novelty `7.0`、mechanism specificity `6.5`、breadth `6.0`、actionability `7.5`，但结论是 `ACCEPT_AS_EVALUATION`，不是主机制；它明确要求用 oracle/因果干预排除多步累积误差，证明表征没有普通 action-insensitivity collapse，扩展到多个载体，并用非平凡修复恢复排序（[主张评议与最低证据矩阵](../runs/gemini_round73_counterfactual_contract_claim_20261010/review.json)）。

在固定合法三步动作库和真实模拟器回报上，已有 cheetah-run→walker-run cascade 更新的 seed32 确实出现局部排序损害：walker-run 的 Spearman 从 clean `0.225` 降到 `0.092`，预测 top-10 的真实回报从 `0.843` 降到 `0.820`；seed31 则为 `0.687→0.656`，没有复制。独立 walker-walk 目标也只在 seed32 出现较弱方向（Spearman `0.468→0.384`，top-10 `0.955→0.940`），seed31 不变；hopper-hop 的三步动作库真实回报全部为零，不能作为排序证据。因此这是一条条件性短门现象，不是跨任务攻击结果（[walker-run 排序](../runs/tdmpc_counterfactual_topology_gate_round73_v2/status.json)，[walker-walk 排序](../runs/tdmpc_counterfactual_topology_walk_round73/status.json)，[hopper 无效对照](../runs/tdmpc_counterfactual_topology_hopper_round73/status.json)）。

随后比较原生均值 Q、五个 Q head 的保守最小值和“均值减标准差”分数。seed32 cascade 的排序没有被修复：walker-run Spearman 分别为 `0.069`、`0.011`、`0.039`，而 clean 为 `0.185`、`0.156`、`0.180`；top-10 真实回报三种分数都保持 `0.821`。seed31 也没有选择性恢复。说明简单 ensemble 保守化既不能承担 Branch Support Audit 的修复，也没有达到 clean/random 变化小于 3% 的门槛（[分支支持审计](../runs/tdmpc_branch_support_audit_round73/status.json)，[脚本](../scripts/pilot_tdmpc_branch_support_audit.py)）。

为直接压低 rollout 累积解释，又运行相同 source、动作库和 seed 的 horizon=1 评分。seed32 的 native Spearman 为 clean `0.141`、random `0.166`、cascade `0.068`，但 top-10 真实回报为 `0.2690/0.2679/0.2682`，差异不到 0.5%；seed31 也没有 cascade 特异的 top-k 损害。因而 horizon=1 只显示排序相关系数的局部波动，尚未形成“单步就能造成任务选择错误”的因果证据（[horizon=1 审计](../runs/tdmpc_branch_support_h1_round73/status.json)）。

当前允许保留的名称是“反事实拓扑投毒评测候选”，适用范围暂限于采样式隐空间规划器和已观察到的条件性 source-target 组合。下一步只有在固定 oracle 转移的 do-intervention、动作敏感性/隐空间秩、至少两个不同载体，以及能选择性恢复排序的分支支持修复同时通过后，才可把它升级为主线；在此之前不启动 ICLR 规模训练，也不把 proposal 分数当作实验新意。

文献边界也进一步收紧：近期 Counterfactual Latent World Model 已把 counterfactual collapse、separability audit 和 hard counterfactual negatives 作为训练目标，并报告规划收益；安全综述也把 trajectory ranking 和 imagined-rollout integrity 列为现有威胁面。因此本项目不能把“反事实排序会坏”本身包装成新攻击，若继续，必须贡献合法交互触发的独立中介和原生任务级证据（[CLWM](https://arxiv.org/abs/2609.05834)，[世界模型安全综述](https://arxiv.org/abs/2607.28226)）。

## Round74：六个 walker-run seed 未形成总体效应

为避免把 seed32 的排序异常误当成机制，固定同一 cheetah-run source、同一 96 条合法三步动作库和同一更新剂量，扩展到 walker-run seed31–36。native Q 的 cascade 相对 clean 的 Spearman 差值为 `−0.045/−0.116/−0.220/+0.040/−0.041/+0.017`，平均 `−0.061`；预测 top-10 真实回报差值为 `−0.0025/−0.0146/−0.0128/+0.0028/+0.0003/−0.0009`，平均仅 `−0.0046`。时间反序的平均排序更低，乱序条件反而更高，因而没有稳定的 source-order 选择性或任务级效应（[六 seed 结果](../runs/tdmpc_branch_support_audit_round74_6seeds/status.json)）。这条线停止扩大训练，只保留为负门槛和条件性诊断。

## Round75：新候选在载体核验前被拒绝

Gemini 在 Round74 后提出两个高分方向，但源码核验否决了两者。ActSWM 的公开代码没有 `SlotAttention`、slot assignment 或对象槽位模块，实际是标准 causal/cross-attention Transformer；因此“槽位置换”机制在该载体不存在。TD-MPC2 的 `WorldModel.task_emb` 确实使用任务 embedding，但实现是 `nn.Embedding` 加 `task.long()` 的整数查表，原生路径没有可被外部连续插值的任务流形；内部直接改 embedding 属于模型/潜变量篡改，外部改 task ID 则是任务 spoof/OOD。两项都没有启动 GPU（[Round75 提案](../runs/gemini_round75_post_gate_search_20261010_v1/review.json)，[源码审计](../runs/gemini_round75_source_audit_20261010/summary.json)）。

## Round76：反馈世界模型暂不可复现

近期的 Feedback World Model 提供了一个潜在的“在线观测残差—隐状态校正”载体，但其官方 GitHub 仓库当前只有项目网页、图片、视频和 README，没有模型源码、权重、环境、数据加载器或推理实现。它不能承担安全机制短门；网页中的 observer 公式和成功率只作为相关工作背景，不计入本项目实验（[载体审计](../runs/feedback_world_model_source_audit_20261010/summary.json)）。

同一轮检查了 World Action Planner。它确实公开了 diffusion video predictor、pose/image condition 管线、WebSocket 推理服务和 LIBERO/Robomimic 环境代码，但 checkout 不含本地 checkpoint，Wan 基础权重和独立模型服务也未提供；下载这些大权重之前尚无明确安全 claim。因此只记录为后续载体，不把代码存在误写成可复现实验（[WAP 载体审计](../runs/world_action_planner_source_audit_20261010/summary.json)）。

## Round78：MoSim 的“Lie 生成元坍缩”候选是源码幻觉

为避免继续复用 TD-MPC2 的离散规划异常，本轮只允许 Gemini 在本地已有 MoSim checkpoint 和源码中寻找候选。它提出在合法有界动作的高频交变下，MoSim 的 Lie 生成元、Magnus 展开和伴随表示乘积会发生 nilpotent contraction，并建议用 Killing 型保积约束修复（[候选记录](../runs/gemini_round78_native_candidate_search_20261010/review.json)）。

直接核对 `vendor/MoSim/MoSim/src/models/dynamics.py` 与 `bijectors.py` 后，实际路径只有 `B(x)+action_module(action)`、可选 learned correctors，以及 `torchdiffeq.odeint_adjoint` 的 `dopri5` 积分。源码没有 Lie generator、Magnus/`exp(ad)`、Jacobian determinant、体积保持损失或所提修复；发布的模型接口每次只接收一个当前 action，也没有未来动作序列或频率载体。此前的 MoSim 非前瞻性探针也显示改变未传入的未来 suffix 时输出完全相同。因此该候选在原生载体门槛前被拒绝，没有启动 GPU，也不把高 proposal novelty 当作证据（[源码审计](../runs/gemini_round78_native_candidate_search_20261010/source_audit.json)）。

## 相关工作边界

## Round50：失败监督下的角色共享机制门控被严格复核否决

为检验 FACT 风格的训练不对称是否会形成可利用的内部通道，新增了一个小型可复现实验：[causal role privilege toy](../runs/causal_role_privilege_toy_v0/summary.json)。模型用共享状态干路同时预测动作、未来状态和价值；干净阶段监督三者，失败阶段只提供真实的未来状态/价值，不提供动作模仿标签。这对应 FACT 公开训练规则中失败样本跳过动作模仿、仍监督未来/价值的结构，但本实验没有加载 FACT 的 10.4 GB 权重，也没有把合成结果当作公开模型证据。

三次独立种子中，共享干路的失败批次未来误差比例为 `0.00687`、价值误差比例为 `0.0917`，而动作误差比例升至 `117.1`，闭环终端距离比例为 `18.4`。分离 outcome trunk 后，动作误差比例为 `1.0`；停止 outcome 分支向共享干路传梯度后也为 `1.0`，同时两种修复仍把失败批次未来误差压到约 `0.0056--0.0066`。因此合成门控复现了“结果预测保持、动作角色漂移、结构性解耦修复”的方向性模式。

该结果只证明了共享多任务网络中的梯度干扰可以被构造出来。Gemini Round50 严格评议给出 `3/10` 并判为 `REJECT`：它等价于普通 negative transfer / shared-trunk gradient interference，停止梯度和分离干路是标准多任务消融，尚不足以构成独立的世界模型安全机制。要升级，必须在官方 FACT checkpoint 上用现实的 replay 混合比例和学习率复现，并且证明动作/闭环任务损害相对普通多任务微调有可重复的隐蔽优势；当前没有下载权重，也没有启动该长实验。

## Round51：在排除既有攻击族后未找到可接受的新主张

把当前硬约束（攻击者只能提供合法物理轨迹或接口输入、不得改梯度/参数/标签、单步指标近似正常、必须有任务级损害和可定位修复）交给 Gemini 做开放式前沿检索，并显式排除投毒/后门、OOD、遗忘、共享干路梯度干扰、不确定性坍缩、视界效应、rollout compounding、采样耦合、分支不一致、动作编码不变性和 CoCo 已覆盖的动作捷径。Round51 返回 `REJECT_ALL`：剩余候选要么是栅格扫描顺序这种普通架构偏置，要么是 latent norm 漂移这种已被 SimNorm/漂移与 rollout 族覆盖的现象，评分分别为 `6.5` 和 `6.0`。

检索还发现近期工作已直接研究“绝对关节目标与相对增量写法”的等价动作编码崩溃，并给出跨数据集任务损害和一致性修复（[Robot World Models Are Not Invariant to How the Actions Are Written](https://arxiv.org/abs/2609.23252)）。因此不能把动作参数化切换重新包装成新的安全漏洞。当前证据支持的结论是：在上述全部约束同时成立时，尚未存在可进入 ICLR 级扩展的候选；若继续推进，必须放宽一个明确的攻击能力/载体约束，或把目标改为系统性否证与评测基准，而不是继续用同一威胁模型寻找“第一个”新攻击。

该候选需要和已有的隐蔽世界模型投毒（[SWAAP](https://arxiv.org/abs/2606.18697)）、尾部轨迹排序后门（[TRAP](https://arxiv.org/abs/2605.01950)）、跨动作物理一致性（[OneWorld](https://arxiv.org/abs/2609.30946)）以及动作条件反事实一致性（[CoCo](https://arxiv.org/abs/2608.04653)）区分。当前证据只显示真实有序交互与规划时任务损害的条件关联，尚未证明一种独立于普通跨任务负迁移的内部安全原语。

近期综述和实证工作也报告了基于 adversarial fine-tuning 的 trajectory-persistent attack 以及 DreamerV3 的 action drift 探针（[Safety, Security, and Cognitive Risks in World Models](https://arxiv.org/abs/2604.01346)）。因此本报告不把“轨迹能持久影响世界模型”本身算作新意；剩余候选必须在只执行合法动作、不给攻击者改数据或梯度的条件下，额外证明可重复的规划损害和可干预的中介。

## 当前结论与升级条件

TD-MPC2 的 Round74 六 seed 复测仍只是条件性诊断；Round75 的两个高分提案缺少原生机制；Round78 的 MoSim 提案则把源码不存在的 Lie/Magnus 结构当成事实。这些结果关闭了旧主线，但不否定新的 goal-bus 机制。当前主张已有一个真实发布 checkpoint、WorldAgen 30 个真实窗口、10 个 episode、结构性 reachability、冻结权重因果 edge-cut、下一帧 forecast 证据，以及 AlphaBrain 的 3 组固定 latent、7 个真实图像/state 窗口、7 个固定动作 sampler 窗口、9 行合法 offset 矩阵、因果阶梯和独立 noise-seed invariance 复现；forecast cost 的符号依赖 noise seed，因此只作为条件性分析。它进入显式 no-goal baseline 与闭环验证阶段，暂不声称已完成最终 ICLR 任务规模。

要完成论文级验证，仍需要显式 goal-free dynamics head 和普通 no-goal baseline 的多任务 forecast error 与成功率。AlphaBrain 的 fixed-action world cut 已在七个真实 sampler 窗口上恢复 goal invariance，并在三窗口 action control 中保留动作条件；native policy branch 并行保留。held-out 16 帧 forecast 的修复 MSE 代价为 +0.06737，bootstrap 区间不跨零。下一步是建立能成功完成任务的 clean baseline；只有修复代价可控且任务层表现不崩溃，才启动长训练和完整核心任务矩阵。

新增第一组任务对通路消融：仅切断观察槽的直接文本 cross-attention 时，future proprio/wrist/primary 的平均绝对差异仍为 0.00448；进一步阻断它们读取 action/value 槽后差异才归零。零 T5 context 诊断下，原生 world 槽平均差异为 0.01535，修复后为 0，action/value 平均差异仍为 0.01403。零 context 不是已验证的训练分布条件。AlphaBrain 当前输入是随机 latent，动作槽也参与联合预测，尚未固定真实已执行动作；因此这里只支持间接文本通路与局部隔离结果，不把它算作主张中固定物理动作不变量的完整跨载体复现。随后用备用环境的原生 transformer-engine 运行时重放同一门：三组交换仍改变全部预测槽，第一组修复后的 future world 差异精确为 0，action/value 平均差异为 0.00692；这仍是随机 latent、联合动作预测，不能替代固定真实已执行动作。

真实输入的因果阶梯进一步确认中间载体：只切断 world rows 的直接文本 cross-attention，三个窗口的 world 差异仍非零，均值 0.04235；同时清除 action/value carrier 的文本后才得到 fixed-action world cut 的精确零差异。详见[world-text-only 消融](../runs/alphabrain_cosmos_policy_real_input_sampler_refined_cut_v0/summary.json)与[聚合脚本](../scripts/analyze_alphabrain_refined_cut.py)。

为进一步定位该中间载体，新增了 WorldAgen 21 窗口 activation mediation patch：在 swapped-goal 前向中，把中间层的 input-action hidden states 替换为 recorded-goal states。layer 4/8 的 world residual 均值降至 0.2062/0.1998，21 个窗口全部向 recorded goal 移动，而 policy residual 为 0.9555/0.9643；该内部 patch 是机制诊断，不是攻击者输入能力。详见[逐层聚合](../runs/worldagen_goal_bus_activation_patch_v1/summary.json)和[图](figures/worldagen_activation_mediation.svg)。

为推进第二载体的真实输入门，我新增了 `scripts/pilot_alphabrain_cosmos_policy_real_input_goal_bus.py`，并在三个公开 LIBERO parquet 文件的七个 episode 上运行。公开 Wan2.1 VAE 的 16 通道、时间压缩和 latent mean/std 配置与 AlphaBrain wrapper 一致；固定记录的16步执行动作槽、状态槽、条件 mask 和 timestep 后，七窗口 native future-world 差异平均 0.03105，覆盖五种任务对，bootstrap 95% CI 为 0.02896–0.03339，edge-cut 后精确为 0，action/value 仍平均变化 0.01629（bootstrap 95% CI 0.01423–0.01865）。该 VAE 文件来自公开 [Wan2.1 Diffusers checkpoint](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)，SHA256 为 `d6e524b3…ed19e793`；因此这是与 gated NVIDIA tokenizer 配置一致的可复核公开 VAE 路径，但仍是单步冻结 DiT forward，没有环境步或任务成功率。详见[真实输入聚合结果](../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/summary.json)、[逐窗口结果](../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/window_3.json)和[门脚本](../scripts/pilot_alphabrain_cosmos_policy_real_input_goal_bus.py)。
同一真实输入门还计算了零 T5 context 诊断：native future-world 与任务嵌入的平均差异为 0.05517，matched cut 后仍精确为 0；这只是 no-goal 诊断，不把零 context 当作发布模型的训练分布。

为减少单次 DiT 噪声输出对结论的依赖，我又使用仓库内的 CosmosPolicy 两步 reverse-ODE sampler，在 episode 3–9 的七个真实窗口中同时固定 observation 槽和记录的 16 步 action 槽。七个窗口的 native future-world goal-swap 差异均非零，平均为 0.05589，bootstrap 95% CI 为 0.03660–0.07801；matched action-bus cut 后七个窗口的 world 差异都精确为 0，action/value policy 差异平均为 0.06373（bootstrap 95% CI 0.02571–0.10887）。这是 sampler confirmation，没有环境步或任务成功率结论。详见[sampler 门脚本](../scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_goal_bus.py)、[sampler 聚合结果](../runs/alphabrain_cosmos_policy_real_input_sampler_goal_bus_v0/summary.json)。

又在三个物理窗口上补做合法 task offset 1、5、9 的跨语义距离矩阵，共 9 个重复配对行。每行 native world 差异均非零，均值为 0.07452，按 episode 聚类的 bootstrap 95% CI 为 0.02845–0.10800；三个 offset 的 edge-cut world 差异全部精确为 0，policy 仍保持敏感。重复 offset 按物理 episode 聚类，不把 9 行当作 9 个独立世界。详见[offset sampler 聚合](../runs/alphabrain_cosmos_policy_real_input_sampler_offsets_v0/summary.json)和[聚合脚本](../scripts/analyze_alphabrain_sampler_offsets.py)。

随后检查 repair 是否误删动作条件：在三个窗口中保持当前图像/state 和 goal 不变，只替换为同一 episode 后续记录的另一段 16 步动作。fixed-action world cut 后 action→world 差异仍为正，均值 0.08715，3/3 非零；因此该修复与旧的 full edge-cut 不同，保留了物理动作对 world branch 的影响。详见[动作敏感性 control](../runs/alphabrain_cosmos_policy_real_input_sampler_action_control_v0/summary.json)和[fixed-action cut 聚合](../runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_cut_v0/summary.json)。

独立 noise seed 2 和 seed 3 的复现进一步区分稳定机制和偶然指标：seed 2 在 episode 3、6、8 的 native world 差异均非零，fixed-action cut 仍精确为零，repair delta 为 −0.00079；seed 3 在 episode 3、4、5 的三个窗口也全部保持非零 native 和精确 cut，repair delta 为 −0.03820。故把 exact invariance 与 action sensitivity 作为主证据，把 seed-1 的正 forecast cost 仅作条件性 fidelity 结果。详见[noise-seed 2 复现](../runs/alphabrain_cosmos_policy_real_input_sampler_noise2_v0/summary.json)与[noise-seed 3 矩阵](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/aggregate.json)。

为了把“修复代价”从槽位差异推进到预测指标，我又把每个窗口当前时刻之后 16 帧的真实 LIBERO 图像/state 用同一公开 VAE 编码为 held-out latent target；target 没有输入 denoiser。扩展 matrix 的 21 个窗口 native goal-swap world 差异均非零，均值 0.05317（episode-cluster bootstrap 95% CI 0.04512–0.06180），fixed-action world cut 在 21/21 窗口精确为零。对应 held-out repair MSE delta 平均 +0.04899（episode-cluster CI 0.03321–0.06798，7/7 episode 聚类均值为正）；独立 seed 2 的三个窗口仍保持 fixed-action exact invariance，但 MSE repair delta 为 −0.00079。native 两个合法 goal 的 MSE 差只有 0.00011（CI −0.02109–0.02052），而 fixed-action cut 后两个 goal 的预测完全相同。作为 prompt-independent 诊断，zero T5 context 的 MSE 也是 0.60833；它没有经过 goal-free dynamics 训练，不能当作强基线。这说明 forecast fidelity 方向依赖 sampler noise，不能写成修复必然增加误差；它仍是 latent forecast metric，没有环境 rollout 或任务成功率。详见[修正 forecast 聚合结果](../runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_forecast_v0/summary.json)、[扩展 21 窗口 matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json)、[matrix 脚本](../scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast_matrix.py)和[聚合脚本](../scripts/analyze_alphabrain_forecast_matrix.py)。

逐窗口曲线见[forecast tradeoff SVG](figures/alphabrain_forecast_tradeoff.svg)；区间估计仍以七个物理窗口为单位，而不是把 latent 坐标当作独立样本。

为补强 no-goal 对照，我又在冻结 2B DiT/VAE 上只训练一个共享的无指令 context。除原先 frame-30 的 3 train/4 held-out episode、seed 17/23 对照外，新增 frame 30/60/90 的 9 个训练窗口和 12 个不重叠 held-out 窗口；seed 29 的 held-out MSE 从 0.60197 降到 0.47852，按四个 episode 聚类的 delta 95% CI 为 −0.18414–−0.01299。它说明对照不依赖单一时间点，但仍不是完整 dynamics-head fine-tune，也没有任务成功率或策略微调。详见 [multi-timepoint learned-context 结果](../runs/alphabrain_goal_free_context_v3/summary.json)。

随后将同一官方 sampler 接入 LIBERO 环境，分别在 spatial、object、goal 三个任务族各跑一个 task0 初始状态，native 与 corrected fixed-action world cut 各执行 32 步。两种条件均为 0/3 成功；动作范数均值分别为 0.415 和 0.653。该结果是修正后控制保留的短时域 sanity check，不能支持任务损害归因，也不能替代官方长时域 clean baseline。详见[修正 AlphaBrain LIBERO 短闭环聚合](../runs/alphabrain_cosmos_libero_short_control_fixed_action_v1/aggregate_summary.json)、[运行脚本](../scripts/pilot_alphabrain_cosmos_libero_short_control.py)和[聚合脚本](../scripts/analyze_alphabrain_libero_short_control_fixed.py)。

进一步扫过全部 9 个合法任务 offset：同一 30 个物理窗口形成 270 个重复配对行。每个 offset 的 native world 差异都非零，edge-cut world 差异均精确为 0，policy 保持敏感；按物理窗口聚类的 bootstrap 95% 区间为 world relative L2 0.286–0.361、edge-cut policy 0.163–0.240、换目标减记录目标 MSE 0.0626–0.0940。重复 offset 没有被当作独立物理样本。详见[offset 扫描](../runs/worldagen_goal_bus_offset_sweep_v1/summary.json)与[聚类统计](../runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json)。

该扫描的逐 offset 曲线见[SVG 图](figures/worldagen_goal_bus_offset_sweep.svg)；它只展示描述性均值，区间估计仍以物理窗口为聚类单位。

为排除相邻任务文本的特殊性，又把替换目标改成任务表中相隔 5 位的合法指令。30 个窗口中 native world relative L2 平均为 0.333，edge-cut 仍精确归零，policy relative L2 为 0.181；换目标减记录目标的 primary MSE 平均差 0.0812，bootstrap 95% 区间 0.0622–0.1012，29/30 为正。该对照支持总线效应不依赖相邻任务语义，但仍是单一 WorldAgen checkpoint。详见[远距 goal 结果](../runs/worldagen_goal_bus_far_goal_v3/summary.json)与[配对统计](../runs/worldagen_goal_bus_far_goal_v3/statistics.json)。

随后补跑了官方 600 步时域的闭环控制 pilot：LIBERO spatial/object/goal 各一个固定初始状态，native 与 edge-cut 共 6 回合，双方均为 0/3 成功，动作范数接近。AlphaBrain 又在 spatial 的一个固定初态上跑完 608 步，native 与 corrected fixed-action world cut 都为 0/1；使用官方 JPEG/center-crop、proprio min-max 归一化和 5-step sampler 的修正版仍为 0/1，动作范数均值 0.25785/0.29310。由于该 checkpoint 配置声明 `dataset_mix: libero_goal`，又在匹配的 `libero_goal` task0–2 上按同一官方管线各跑 304 步，三个 task 的 native 与 corrected cut 均为 0/3，动作范数均值 0.36723/0.32465。两个 checkpoint/configuration 都没有通过 clean task gate，所以不能把 cut 解释成任务损害；这些结果只说明当前模型不适合作为任务成功基线。这个负结果也使主张范围保持在 forecast-level 因果不变量，不声称已有闭环攻击成功。详见[WorldAgen 600 步控制 pilot](../runs/worldagen_libero_closed_loop_full_pilot_v0/combined_summary.json)、[AlphaBrain 官方一致预处理修正版](../runs/alphabrain_cosmos_libero_full_control_v2/spatial_600.json)和[匹配 Goal suite task0–2 聚合](../runs/alphabrain_cosmos_libero_goal_control_v0/aggregate_summary.json)。

任务层短基线也已跑通：使用发布的 WorldAgen 权重，在 LIBERO spatial/object/goal 三个任务族各取一个固定初始状态，native、九边 edge-cut 和空指令各一回合，每回合 36 个环境步，共 9 回合。三组条件均为 0/3 成功；edge-cut 与 native 的动作范数接近。由于官方评测上限为 600 步，这个短时域只能说明修复没有立即让控制输出崩溃，不能当作任务成功率或攻击成功证据。完整多 seed 闭环评测仍未完成。详见[短闭环结果](../runs/worldagen_libero_closed_loop_v0/summary.json)与[运行脚本](../scripts/pilot_worldagen_libero_closed_loop.py)。

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
- [Round78 MoSim 候选记录](../runs/gemini_round78_native_candidate_search_20261010/review.json)
- [Round78 MoSim 源码审计](../runs/gemini_round78_native_candidate_search_20261010/source_audit.json)
- [Goal-bus 主张与短门设定](../configs/world_model_goal_bus_security.json)
- [Goal-bus 扩展三十窗口实验](../runs/worldagen_goal_bus_replicate_v2/summary.json)
- [WorldAgen 同 episode alternate-action control](../runs/worldagen_goal_bus_action_control_v1/summary.json)
- [WorldAgen matched policy-only edge-cut control](../runs/worldagen_goal_bus_edge_cut_controls_v0/summary.json)
- [WorldAgen 六任务语义分层](../runs/worldagen_goal_bus_task_groups_v0/summary.json)
- [WorldAgen 动作保真与任务分层图](figures/worldagen_action_control.svg)
- [扩展 LIBERO-10 数据 provenance](../data/worldagen_goal_probe_extra/provenance.json)
- [三十窗口配对统计](../runs/worldagen_goal_bus_replicate_v2/statistics.json)
- [Goal-bus 先前九窗口复现实验](../runs/worldagen_goal_bus_replicate_v1/summary.json)
- [AlphaBrain CosmosPolicy 第二载体源码审计](../runs/alphabrain_cosmos_policy_goal_bus_audit_v0/summary.json)
- [AlphaBrain CosmosPolicy 固定 latent goal-swap 短门](../runs/alphabrain_cosmos_policy_goal_swap_v0/summary.json)
- [AlphaBrain CosmosPolicy 七窗口真实输入 goal-bus 短门](../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/summary.json)
- [AlphaBrain CosmosPolicy 七窗口固定动作 sampler 确认](../runs/alphabrain_cosmos_policy_real_input_sampler_goal_bus_v0/summary.json)
- [AlphaBrain CosmosPolicy sampler 聚合脚本](../scripts/analyze_alphabrain_real_input_sampler_goal_bus.py)
- [AlphaBrain CosmosPolicy 合法 task-offset sampler 矩阵](../runs/alphabrain_cosmos_policy_real_input_sampler_offsets_v0/summary.json)
- [AlphaBrain CosmosPolicy 七窗口 fixed-action held-out forecast 误差](../runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_forecast_v0/summary.json)
- [AlphaBrain CosmosPolicy 扩展 21 窗口 forecast matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json)
- [AlphaBrain bounded learned goal-free context baseline](../runs/alphabrain_goal_free_context_v1/summary.json)
- [AlphaBrain learned context seed-23 replicate](../runs/alphabrain_goal_free_context_v2/summary.json)
- [AlphaBrain learned context multi-timepoint check](../runs/alphabrain_goal_free_context_v3/summary.json)
- [AlphaBrain corrected official-preprocess 608-step pilot](../runs/alphabrain_cosmos_libero_full_control_v2/spatial_600.json)
- [AlphaBrain matching LIBERO-Goal task0–2 aggregate](../runs/alphabrain_cosmos_libero_goal_control_v0/aggregate_summary.json)
- [AlphaBrain training-resource audit](../runs/alphabrain_training_resource_audit_v0/summary.json)
- [AlphaBrain CosmosPolicy independent noise-seed 复现](../runs/alphabrain_cosmos_policy_real_input_sampler_noise2_v0/summary.json)
- [AlphaBrain CosmosPolicy independent seed-3 forecast matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/aggregate.json)
- [Goal-bus 复现与证据索引](goal_bus_reproduction_manifest_cn.md)
- [AlphaBrain CosmosPolicy forecast probe](../scripts/pilot_alphabrain_cosmos_policy_real_input_sampler_forecast.py)
- [公开 Wan2.1 VAE provenance 与 hash](../runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/vae_provenance.json)
- [AlphaBrain CosmosPolicy 真实输入聚合脚本](../scripts/analyze_alphabrain_real_input_goal_bus.py)
- [真实输入 goal-bus 门脚本](../scripts/pilot_alphabrain_cosmos_policy_real_input_goal_bus.py)
- [reacher-easy 独立任务族顺序门槛](../runs/tdmpc_order_controls_reacher_gate_v0/status.json)
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
- [CartPole rank-coupling 配置与跨任务边界](../configs/world_model_sampling_security.json)
- [CartPole bank-mean 防御复核 seed706/707](../runs/mbrl_rank_bank_mean_confirm_v1/bank_mean/summary.json)
- [Reacher3D rank-coupling 独立负对照](../runs/mbrl_rank_reacher_v0/targeted/summary.json)
- [Rank-coupling 主张严格评议](../runs/gemini_rank_coupling_claim_review_20261010T0000Z/review.json)
- [Round24 新载体机制检索](../runs/gemini_mechanism_search_round24_20261010T0000Z/review.json)
- [Round25 bootstrap correlation 严格评议](../runs/gemini_mechanism_search_round25_20261010T0000Z/review.json)
- [Round26 严格机制封存评议](../runs/gemini_mechanism_search_round26_20261010T0000Z/review.json)
- [Round27 跨载体候选评议](../runs/gemini_cross_carrier_search_round27_20261010T0000Z/review.json)
- [Pusher provenance shadow 短门](../runs/mbrl_provenance_shadow_pusher_gate_v1/summary.json)
- [Round28 动作间反事实候选评议](../runs/gemini_cross_carrier_search_round28_20261010T0000Z/review.json)
- [Round30 action-null gap 评议](../runs/gemini_action_null_gap_review_round30_20261010T0000Z/review.json)
- [Round29 gateway 请求记录（无可用正文）](../runs/gateway_pivot_search_round29_20261010T0000Z/calls.jsonl)
- [Pusher ensemble synchronization dose20](../runs/mbrl_ensemble_sync_pusher_gate_v2/summary.json)
- [Pusher ensemble synchronization dose5](../runs/mbrl_ensemble_sync_pusher_dose5_v0/summary.json)
- [Round31 MBRL-Lib 时间顺序独立门槛](../runs/mbrl_temporal_source_pusher_gate_v0/summary.json)
- [Round32 gateway 输出失败记录](../runs/gateway_novel_pivot_round32_20261010T0000Z/calls.jsonl)
- [WorldAgen goal leakage 严格评议](../runs/gemini_worldagen_goal_leak_review_20261010T0000Z/review.json)
- [WorldAgen stale goal queue 短门](../runs/worldagen_stale_goal_queue_v0/summary.json)
- [Round33 合成联合支持空洞短门](../runs/joint_support_hole_gate_v1/summary.json)
- [Round33 native MBRL 联合支持空洞短门](../runs/mbrl_joint_support_hole_gate_v3/summary.json)
- [Round34 WAV 合法作用范围短门](../runs/wav_footprint_gate_v1/summary.json)
- [Round34 WAV 短门脚本](../scripts/pilot_wav_footprint_gate.py)
- [Round41 A2World 固定噪声序列非局部性](../runs/a2world_fixed_noise_suffix_v0/summary.json)
- [Round41 A2World 全局动作条件探针](../runs/a2world_global_action_condition_v0/summary.json)
- [Round41 OpenWAM 掩码语义审计](../runs/openwam_temporal_mask_audit_v0/summary.json)
- [Round50 共享角色机制合成门控](../runs/causal_role_privilege_toy_v0/summary.json)
- [Round50 共享角色严格复核](../runs/gemini_round50_causal_role_strict_20261010/review.json)
- [Round51 开放前沿候选检索](../runs/gemini_round51_open_frontier_search_20261010/review.json)
- [共享角色机制门控脚本](../scripts/pilot_causal_role_privilege_toy.py)
- [A2World 官方仓库](https://github.com/LogosRoboticsGroup/A2World)
- [OpenWAM 官方仓库](https://github.com/OpenWAM-Official/OpenWAM)
- [Round16 commutator gate](../runs/tdmpc_commutator_defense_v0/status.json)
- [Round16 spectrum gate](../runs/tdmpc_spectrum_defense_kappa3_v0/status.json)
- [候选路线图](../configs/research_roadmap.json)
- [自动生成研究进展报告](research_progress.html)
