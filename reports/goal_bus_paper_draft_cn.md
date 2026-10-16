# 共享动作 token 因果总线：指令条件世界模型中的预测层安全边界

> **历史材料，不能作为当前投稿或完成证明（2026-10-15）。** 下文保留旧协议和旧结论；部分 codec、原生决策路径、Clean 判断及不变性前提已被后续审计修正。当前证据见[研究进度](research_progress.html)与[现行英文工作稿](world_model_decision_audit_working.pdf)。完整安全机制、有效防御、独立泛化和论文竞争力尚未建立。原样版本保存在 [historical_goal_bus_package_20261015/](historical_goal_bus_package_20261015/)。历史复现命令不代表当前实验建议。

## 摘要

多任务机器人世界模型通常把任务指令、观测和动作放进同一个时空 Transformer。我们研究一种只需要切换合法任务指令的攻击：图像、proprioception、实际执行的动作、奖励、标签、参数和梯度全部固定，但任务文本仍可能改变物理未来预测。本文的核心主张是：**指令条件世界模型会把任务文本写入共享的动作 token 因果总线，使 goal 在直接 goal→world 掩码存在时仍能改变固定状态与固定动作下的物理未来预测；切分该总线可以恢复 goal-invariant world prediction，同时保留目标条件策略输出，暴露控制条件与动力学保真之间的安全权衡。**

我们在两个公开发布载体上验证这一主张。WorldAgen 的 30 个真实 LIBERO-10 窗口中，固定图像、状态和动作换 goal 的 world relative L2 为 0.2999；切断原生九条 goal→input-action 边后 world 输出在 30/30 窗口精确不变，policy relative L2 仍为 0.1760。同一载体的 alternate-action control 在 30 个窗口中只替换同 episode 的六步真实动作块，修复后 world 差异均非零，均值 0.1655，episode-cluster CI 为 [0.1469, 0.1855]。AlphaBrain CosmosPolicy 的扩展 held-out matrix 覆盖七个真实 episode、三个时间点共 21 个固定图像/state/action 窗口；native world 差异均非零，均值 0.05317，按 episode 聚类 bootstrap 95% CI 为 [0.04512, 0.06180]，fixed-action world cut 在 21/21 窗口精确恢复 invariance。合法 task offset 1、5、9 矩阵另有 9/9 非零 native 和 9/9 精确 invariance。seed 1 的 21 窗口 forecast repair delta 为 +0.04899，episode-cluster CI [0.03321, 0.06798]；独立 seed 2 和 seed 3 的各三个窗口仍保持 exact invariance，但 MSE 修复差分别为 −0.00079 和 −0.03820，因此 forecast cost 是条件性结果，不是主 claim。三窗口 action control 仍显示修复后的 action→world 差异均非零，均值 0.08715。
额外的 WorldAgen activation mediation 在 7 个 episode、frame 30/60/90 共 21 个窗口中把中层 action-token hidden state 从记录 goal 注入 swapped-goal 前向；layer 4/8 的 world residual 降至 0.2062/0.1998（21/21 向记录 goal 移动，cluster CI 分别为 [0.1678, 0.2482] 和 [0.1796, 0.2202]），而 policy residual 仍为 0.9555/0.9643，为共享中间总线提供了跨时间点的层级定位证据。

任务级结果被单独报告：当前释放的 WorldAgen 和 AlphaBrain checkpoint 都没有通过 LIBERO clean-task gate，因此本文不声称任务损害或闭环攻击成功。研究结论目前属于跨载体、预测层的安全机制证据；正式的 goal-free dynamics head 和能成功完成任务的 clean baseline 是下一阶段要求。

## 1. 威胁模型与安全不变量

攻击者只能在模型支持的多任务接口上选择另一条合法任务指令。以下量保持不变：当前图像、proprioception、固定的已执行动作 chunk、奖励、终止标签、模型权重、训练数据和梯度。攻击者不能伪造 observation、修改 reward、注入 OOD action 或改变推理参数。

安全不变量是：当物理状态和已执行动作固定时，任务文本不应改变世界模型对下一物理观测的预测，除非任务文本代表一个明确的隐藏物理变量。策略输出可以保留任务条件，因为不同任务可以合法地产生不同控制。本文的干预只作用于冻结模型的 attention path，不更新权重。

本文贡献有三点：第一，把合法 task-text 切换下的 fixed-state/fixed-action world invariance 明确化为可检验的安全不变量；第二，在 WorldAgen 与 AlphaBrain 两个载体上用 native swap、matched edge cut、alternate-action control 和 held-out forecast 共同定位共享 action-token 因果总线；第三，提供冻结模型的机制修复与边界审计，区分稳定的 invariance/action sensitivity 证据、依赖 sampler noise 的 forecast-fidelity 结果，以及尚未建立的 task-level harm。

### 1.1 可证伪的估计量与判据

令冻结模型在输入图像、状态、已执行动作和指令 `g` 下的 world 预测为 `F_θ(x,s,u,g;m)`，policy 输出为 `P_θ(x,s,u,g;m)`，其中 `m` 是 attention mask 或冻结的路径干预。对一对合法指令 `g,g'`，定义固定物理输入下的目标效应 `Δ_g^W(m) = d(F_θ(x,s,u,g;m), F_θ(x,s,u,g';m))`；对同一指令下的一对真实动作 chunk `u,u'`，定义动作敏感度 `Δ_u^W(m) = d(F_θ(x,s,u,g;m), F_θ(x,s,u',g;m))`。`d` 使用各实验记录的 relative L2 或 latent distance，不能把不同载体的数值直接混为同一尺度。

对 WorldAgen 的第 `l` 层 activation patch，令 `B←A;l` 表示把记录 goal `A` 的 action-token hidden state 注入 swapped-goal `B` 的前向。我们报告归一化残差 `ρ_l^W = d(F_{B←A;l},F_A)/d(F_B,F_A)`，并以同样方式定义 `ρ_l^P`。`ρ_l^W<1` 且 `ρ_l^P≈1` 是“中间 action state 介导 world、但没有把 policy 一起替换”的定位证据，不等同于攻击者可执行的干预。

主判据是：native `Δ_g^W>0`，matched fixed-action cut 使 `Δ_g^W≈0`，同时 policy goal sensitivity 或 alternate-action `Δ_u^W` 保持非零。held-out forecast MSE repair 是次级 fidelity 量，只在固定 sampler/noise 条件下解释。以下结果会被视为反证：native world 已经对 goal 不敏感；切断总线后仍不恢复 invariance；cut 同时抹掉真实 action sensitivity；或 activation patch 没有在中间 action state 附近产生局部化恢复。

相关工作定位：WorldVLA 研究 action/image 联合建模和抑制历史 action 误差传播的 mask；LingBot-VA 研究共享 vision/action latent 与闭环执行；AffordanceWAM 用 masked joint attention 连接分离的 World/Action Expert；When to Trust Imagination 用真实观测校验 imagined future 并调整 action chunk。这些工作说明了 world-action 通路和 mask 设计的重要性，但没有把“固定物理状态与已执行动作时，合法 task text 改写 world forecast”定义为安全不变量，也没有用冻结权重的 edge split 恢复该不变量。详细链接见[related-work memo](related_work_goal_bus.md)。

## 2. 机制假设

在 WorldAgen 的原生 attention mask 中，goal token 没有直接读写 world-output token 的边，但可以先写入 contextualized input-action token，再由该 token 写入 future-world token。这个两跳路径构成共享动作 token 因果总线。切断九条 goal→input-action 边后，world 输出失去这条路径，而 action/policy 支路仍可读取 goal。

AlphaBrain CosmosPolicy 提供独立载体：一个 T5-conditioned DiT 同时生成 action、future proprio、future wrist、future primary 和 value latent slots。我们在不改权重的前提下，对 world rows 的 context/self-attention 读边执行 matched row-wise cut，并保持观测、state、记录动作、噪声、时间步和 condition mask 相同。

消融给出必要性链条：只移除 world rows 的直接文本 cross-attention 并不能修复问题，三个真实窗口的 future-world 差异仍全部非零，均值 0.04235；只有同时让 action/value carrier 不再接收文本，fixed-action world cut 才恢复精确 invariance。这把修复定位到“文本先写入 action carrier、再被 world branch 读取”的中间总线，而不是简单的 world-row 文本泄漏。

### 2.1 Activation mediation：中间 action state 的层级 patch

为把 edge cut 的必要性进一步推进到中介证据，我们在 WorldAgen 的 24 层 Transformer 上做 activation patch。对 7 个 episode 的 frame 30/60/90 共 21 个固定真实窗口，先记录 goal A 的 action-token 隐状态；再用相同图像、state、已执行 action 和 goal B 前向，只在指定层把两个 chunk 的 input-action hidden states 替换成 goal A 的对应 states。定义 world residual 为 patched-B 与 A 的相对距离除以 native B 与 A 的距离；policy residual 用同一归一化。该内部 patch 只用于定位机制，不是攻击者能力。

在 layer 4 和 8，world residual 均值分别降到 0.2062（cluster bootstrap CI [0.1678, 0.2482]）和 0.1998（[0.1796, 0.2202]），且 21 个窗口全部向 goal A 移动；对应 policy residual 为 0.9555 和 0.9643，说明策略支路仍主要保留 goal B。初始 7 窗口的完整层级 sweep 还显示 layer 0/12/16/20 的 world residual 为 0.8898/0.4851/0.6916/0.8723，最后一层 patch 不再产生下游变化。这种“中层 action carrier patch 强烈恢复 world、几乎不改 policy”的位置依赖，与共享 action-token 因果总线的预测相符，并补充了直接 edge cut 的证据。[21 窗口聚合结果](../runs/worldagen_goal_bus_activation_patch_v1/summary.json)和[图](figures/worldagen_activation_mediation.svg)。

## 3. 实验矩阵

| 载体与实验 | 物理窗口／重复行 | native world | edge-cut world | edge-cut policy | forecast 对照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WorldAgen 真实输入 goal swap | 30 / 10 episodes | rel. L2 0.2999 | 0 | rel. L2 0.1760 | swapped-recorded MSE +0.0732 |
| WorldAgen 9 个合法 goal offset | 270 / 30 physical windows | rel. L2 0.3247，CI [0.2860, 0.3613] | 270/270 exact zero | rel. L2 0.2000，CI [0.1632, 0.2401] | MSE delta +0.0779，CI [0.0626, 0.0940] |
| WorldAgen same-episode alternate action | 30 / 10 episodes | native action→world 0.1623 | cut 0.1655，30/30 nonzero | cut policy 0.2124 | fixed image/state/goal |
| AlphaBrain 真实输入单步 | 7 / 7 episodes | 0.03105 | 0 | 0.01629 | 单步 DiT forward |
| AlphaBrain official sampler, offsets 1/5/9 | 3 episodes / 9 rows | 0.07452，CI [0.02845, 0.10800] | 0 | native policy branch | 两步 reverse-ODE |
| AlphaBrain expanded held-out forecast (seed 1) | 21 / 7 episodes | native/cut MSE matrix | fixed-action invariance 21/21 | native policy branch | repair +0.04899，episode CI [0.03321, 0.06798]；seed 2/3 条件性 |

重复 task offset 按物理 episode 聚类，不把重复指令行当作独立世界。完整聚合见[主结果 JSON](../runs/goal_bus_main_results_v0/summary.json)。

### 3.1 WorldAgen：真实输入与 empty-goal comparator

实验使用公开 LIBERO-10 parquet 中 10 个 episode 的 30 个物理窗口。每个窗口固定当前图像、state 和记录动作，交换合法 goal，分别测量 native、empty-goal 和 edge-cut。native goal swap 的 world relative L2 平均 0.2999，empty-goal 为 0.2704；edge-cut world 输出在所有窗口精确不变，policy 仍保持敏感。记录目标、换目标、edge-cut 的 primary forecast MSE 分别为 1.4228、1.4960、1.4841；换目标减记录目标的 bootstrap 95% CI 为 [0.0465, 0.1030]。

作为匹配特异性对照，我们同样删除 9 条有限 goal 边，但把目标放在 future-action policy rows，而不是 input-action carrier。policy-only cut 在 30 个窗口中仍保持 world goal-swap relative L2 为 0.2999、30/30 非零；只有 proposed carrier cut 得到精确零，因此可以排除“删除边的数量”本身造成 invariance。详见[匹配删边聚合](../runs/worldagen_goal_bus_edge_cut_controls_v0/summary.json)。

将同一 30 个物理窗口与全部 9 个合法 goal offset 交叉后得到 270 个配对行。native world relative L2 均值为 0.3247，按物理窗口聚类 bootstrap 95% CI 为 [0.2860, 0.3613]；270/270 edge-cut world 输出精确不变，policy relative L2 均值为 0.2000。换目标减记录目标的 primary MSE 均值为 0.0779，聚类 CI 为 [0.0626, 0.0940]。重复 offset 只增加指令覆盖，不被当作独立物理样本。

在同一载体的物理动作对照中，保持当前图像、状态和记录指令不变，只把六步 action chunk 换成同 episode 的另一段真实动作。30 个窗口（10 个 episode、frame 30/60/90）中，native action→world relative L2 均值为 0.1623，episode-cluster CI [0.1426, 0.1839]；切断 goal bus 后仍为 0.1655，CI [0.1469, 0.1855]，30/30 个修复差异非零。因此修复移除了指令依赖，但没有冻结 WorldAgen 对已执行物理动作的响应。详见[动作对照聚合](../runs/worldagen_goal_bus_action_control_v1/summary.json)。

动作对照和六个任务组的结果见[汇总图](figures/worldagen_action_control.svg)。

留一 offset 稳定性分析中，删除任意一个 offset 后 native mean 仍为 0.31685–0.33044；9 个 offset 的 native mean 最小值为 0.27854，最大值为 0.38725，修复 world 在所有 offset 仍 exact zero。该结果排除了某一个语义距离单独驱动总线效应的解释。

将同一 30 个窗口按记录的合法 task text 分层得到 6 个任务组（每组 3–9 个窗口）。每个任务组的 native world 敏感性都非零，cut 输出都精确为零；组均值范围为 0.0972–0.5792。该结果是描述性的任务覆盖，重复时间点仍按所属物理 episode 解释。详见[任务组聚合](../runs/worldagen_goal_bus_task_groups_v0/summary.json)。

### 3.2 AlphaBrain：固定真实动作的跨载体复现

公开 Wan2.1 Diffusers VAE 将 LIBERO 图像编码为 `[1,16,9,28,28]` latent。每个窗口使用真实记录的 16 步执行动作 chunk，不用零动作替代。单步 DiT 门在七个 episode 上通过；随后使用仓库内 CosmosPolicySampler 的两步 reverse-ODE 实现，在 episode 3–9 上重复。native future-world goal-swap 差异 7/7 非零，edge-cut world 差异 7/7 精确为零，action/value policy rows 仍敏感。

为了检查任务语义距离，三个物理 episode 分别选择 task offset 1、5、9。9 行中 native world 差异均非零，按 episode 聚类 bootstrap 95% CI 为 [0.02845, 0.10800]；三个 offset 的 edge-cut world 差异全部为零。

在同一三个 episode 的 frame-30 配对 control 中，合法 goal-swap 的 world effect 均值为 0.07137，而修复后的 alternate-action effect 为 0.08715，前者约为后者的 0.819。这个比值是描述性效应量，不把三个 episode 当作显著性检验；它说明 goal effect 在量级上接近真实动作干预，而不是仅仅数值非零。[效应量聚合](../runs/alphabrain_goal_vs_action_effect_v0/summary.json)。

### 3.3 Held-out forecast tradeoff

对每个窗口，取当前时刻之后 16 帧的真实图像/state，用同一公开 VAE 编码成 target latent；target 不输入 denoiser。扩展 matrix 在七个 episode 的三个时间点共 21 个窗口上运行，native goal-swap world 差异均非零，均值 0.05317，episode-cluster bootstrap 95% CI 为 [0.04512, 0.06180]；fixed-action world cut 后 21/21 精确为零。held-out repair MSE delta 平均 +0.04899，episode-cluster CI [0.03321, 0.06798]，7/7 episode 的聚类均值为正。该 cut 保留 self-attention 对记录 action carrier 的读取，三窗口 action control 的 repaired world delta 均非零。

为检查 sampler 噪声依赖，在固定动作矩阵之外用 noise seed 2 重复 episode 3、6、8 三个窗口，并用 seed 3 重复 episode 3、4、5 三个窗口。两组中 native world 差异均为 3/3 非零，fixed-action cut 均为 3/3 精确为零；repair MSE delta 均值分别为 −0.00079 和 −0.03820。因而本文把 exact goal invariance 和 action sensitivity 作为稳定机制证据，把 +0.06737 forecast cost 只报告为 seed-1 的条件性 fidelity tradeoff。

zero T5 context 的 MSE 也是 0.60833。AlphaBrain 的预计算 embedding 表没有空指令项，因此 zero context 是 prompt-independent diagnostic，不是训练过的 goal-free dynamics head。WorldAgen 的 empty-goal 条件是本文正式的 no-goal comparator。旧的 full edge-cut 仍保留为过强 legacy comparator，但不作为 fixed-action repair 的主结果。

按 task-A 文本分层后，seed-1 矩阵覆盖五个合法任务语义组；每组 native world 差异仍全部非零，fixed-action cut 在每组都精确为零。合并 seed-3 的三个重复窗口后，五组的窗口数分别为 6、3、8、4、3，task-group native means 为 0.04878、0.04118、0.04450、0.12809、0.06945。该分层是核心任务覆盖的描述性证据，重复时间点仍按 episode 解释，不能当作额外独立物理世界。

基线和消融的职责是分开的：WorldAgen empty-goal 检验无目标条件；world-text-only cut 检验只切 world 入口是否足够；fixed-action cut 检验切开中间 action bus 后是否恢复安全不变量；same-episode alternate-action control 检验修复是否误删物理动作条件；native parallel policy branch 检验策略条件是否仍可保留。没有把 zero-context 或旧 full-cut 当作训练过的强 no-goal baseline。

### 3.4 一个受控的 learned goal-free context 对照

为了把 zero-context diagnostic 与可训练对照区分开，我们冻结发布的 2B DiT 和 VAE，只优化一个共享的 `512×1024` 无指令 context。最初的短对照使用 episode 3、4、5 的三个 frame-30 窗口训练，评估不重叠的 episode 6–9 四个窗口；两个独立随机种子（17、23）的 8 步、学习率 0.001 短优化都把 one-step future-world MSE 从 zero-context 的 0.58519 降到 0.46215 和 0.46219。为排除单一时间点的偶然性，随后保持相同 episode 划分和优化预算，改用 frame 30/60/90 的 9 个训练窗口和 12 个 held-out 窗口；seed 29 将 MSE 从 0.60197 降到 0.47852，四个 episode 的聚类 delta 均值为 −0.12346，episode-bootstrap 95% CI 为 [−0.18414, −0.01299]。学习率 0.05 的同一预算反而升到 1.47141，说明这个对照对优化设置敏感。

这个结果只说明一个低容量、跨任务共享的 learned context 可以作为 goal-free forecast comparator；它不是重新训练的 dynamics head，也没有 policy fine-tuning、环境 rollout 或任务成功率。多时间点扩展提高了对照的覆盖，但仍不能替代完整的匹配训练 goal-free dynamics baseline。

本地资源审计进一步确认了这个边界：发布的 AlphaBrain 2B 推理权重、统计文件、T5 embedding 以及官方 trainer/loader 都在，但兼容的 `success_only`/`all_episodes` LIBERO-Cosmos-Policy 训练轨迹目录不存在。因此当前 workspace 无法形成完整匹配架构的 fine-tuned dynamics head；learned context 只能作为受限的 forecast 对照。详见 [训练资源审计](../runs/alphabrain_training_resource_audit_v0/summary.json) 和 [审计脚本](../scripts/audit_alphabrain_training_resources.py)。

三次对照的 held-out MSE 见[可视化图](figures/goal_free_context_baseline.svg)；图中所有 DiT/VAE 权重都冻结，柱状差异只表示 one-step forecast comparator，不表示任务成功率。

为了让比较关系可直接审阅，主 baseline ladder 如下：

| 基线／干预 | 载体与规模 | world 结果 | 保留的控制条件 | 作用 |
| --- | --- | --- | --- | --- |
| Native mask | WorldAgen，30 窗口 | goal-swap rel. L2 0.2999 | native policy | 暴露原生泄漏 |
| Empty-goal | WorldAgen，同一 30 窗口 | rel. L2 0.2704 | 无 goal 条件 | 正式 no-goal comparator |
| Matched policy-only edge cut | WorldAgen，30 窗口 | 同样 9 边预算仍为 rel. L2 0.2999，30/30 非零 | 删除 future-action policy edges | 排除删边数量混淆 |
| Same-episode alternate-action control | WorldAgen，30 窗口 | cut action→world rel. L2 0.1655，30/30 非零 | 固定 goal、替换真实 action chunk | 检验修复保留物理动作敏感性 |
| World-text-only cut | AlphaBrain，3 窗口 | residual 0.04235，3/3 非零 | action/value 仍带文本 | 排除只切 world 入口 |
| Fixed-action world cut | AlphaBrain，21 窗口 | 0，21/21 exact | self-attention 与 action control 保留 | 主修复 |
| Alternate-action control | AlphaBrain，3 窗口 | repaired delta 0.08715 | 改变记录 action chunk | 检验物理动作敏感性 |
| Noise seeds 2/3 | AlphaBrain，各 3 窗口 | 两组均 exact zero | fixed image/state/action | 检验机制对采样噪声的稳定性 |
| Learned goal-free context | AlphaBrain，3 train / 4 held-out episode，seed 17/23 | one-step MSE 0.58519 → 0.46215/0.46219 | 只训练共享 context | 可训练 no-goal forecast comparator；非完整 dynamics head |
| Learned context multi-timepoint check | AlphaBrain，9 train / 12 held-out windows，seed 29 | MSE 0.60197 → 0.47852；episode CI [−0.18414, −0.01299] | 只训练同一共享 context | 检验对照不依赖单一 frame；仍非完整 dynamics head |

因此，empty-goal、direct-text cut 和 fixed-action cut 分别承担不同的对照责任；没有用一个会同时删除动作条件的 full cut 冒充强基线。

## 4. 任务级检查与强基线边界

我们在 LIBERO spatial、object、goal 三个任务族执行了冻结控制 pilot。WorldAgen 官方 600 步 native/edge-cut 均为 0/3；AlphaBrain corrected fixed-action world cut 的两个 sampler chunk 也均为 0/3。WorldAgen 的 native 与 edge-cut 动作范数均值约为 1.780/1.794；AlphaBrain 修正 cut 的均值为 0.653（native 为 0.415）。这些短时域结果只用于控制保留检查，不能解释为任务成功率或策略保持已证实；clean checkpoint 本身仍没有通过任务 gate。

为确认失败不是短 horizon 偶然现象，AlphaBrain spatial task0 的一个固定初态又运行到 608 步：native 和 corrected fixed-action world cut 都为 0/1。该长时域结果只强化 clean-gate 失败的判断，不构成任务损害证据。

随后将该 pilot 修正为官方 CosmosPolicy 的图像 JPEG/center-crop、proprio min-max 归一化和 5-step sampler 配置，重新运行同一 spatial 初态 608 步；native 与 fixed-action cut 仍为 0/1，动作范数均值为 0.25785 和 0.29310。由于该 checkpoint 的配置是 `dataset_mix: libero_goal`，又在匹配的 `libero_goal` task0–2 上运行同一官方管线，每个 task 304 步；三个 task 的 native 与 fixed-action cut 均为 0/3，平均动作范数为 0.36723 和 0.32465。由此 clean-task gate 的失败不是前一版本地预处理差异或仅仅 Spatial OOD 造成的；该结果仍只用于任务能力边界，不能解释为 task harm。

因此，本文不把这些结果写成任务损害成功率。它们是核心任务载体上的 negative control：当前 checkpoint 不支持把预测层变化外推为闭环任务影响。要完成任务级论文结论，需要能完成同一 LIBERO 任务的 clean baseline，以及匹配的训练 goal-free dynamics head。

## 5. 结论、限制与复现入口

现有证据支持一个跨载体的预测层机制：合法 goal 可以经共享 action token 改变固定物理输入下的 world forecast；切分该总线可以恢复 world invariance，同时保留 policy sensitivity；修复的 forecast fidelity 代价可能随 sampler noise 改变方向。证据不支持以下更强表述：修复必然增加预测误差、已经造成任务失败、已经证明所有 instruction-conditioned world models 都有该漏洞、或已经完成多 seed 的 ICLR 级任务成功率基准。

AlphaBrain 的额外 sampled activation patch（3 个窗口）没有复现 WorldAgen 的强层级回归：action/value carrier patch 的 world residual 约为 0.81–1.06，且 sampler 会把 action slot 作为固定输入重新写回，因此不能从该输出直接证明 policy 保留性。该结果不进入主阳性证据；AlphaBrain 的机制结论仍限定为 row-wise causal ladder、fixed-action cut 和 alternate-action control 所支持的 carrier-level mediation。

主要复现入口如下：

- [复现与证据索引](goal_bus_reproduction_manifest_cn.md)
- [提交包 manifest](goal_bus_submission_manifest.json)
- [交付审计 checklist](goal_bus_completion_audit_cn.md)
- [投稿包 README](README_goal_bus_submission.md)
- [统一证据图 SVG](figures/goal_bus_evidence_summary.svg)
- [统一证据图 PDF](figures/goal_bus_evidence_summary.pdf)
- [English submission draft](goal_bus_paper_draft_en.md)
- [English PDF](goal_bus_paper_draft_en.pdf)

- [WorldAgen 30 窗口结果](../runs/worldagen_goal_bus_replicate_v2/summary.json)
- [WorldAgen 270 行 offset stability](../runs/worldagen_goal_bus_offset_stability_v0/summary.json)
- [AlphaBrain 七窗口 sampler](../runs/alphabrain_cosmos_policy_real_input_sampler_goal_bus_v0/summary.json)
- [AlphaBrain offset 矩阵](../runs/alphabrain_cosmos_policy_real_input_sampler_offsets_v0/summary.json)
- [AlphaBrain legacy full-cut forecast](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_v0/summary.json)
- [AlphaBrain fixed-action world cut](../runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_cut_v0/summary.json)
- [AlphaBrain world-text-only ablation](../runs/alphabrain_cosmos_policy_real_input_sampler_refined_cut_v0/summary.json)
- [AlphaBrain action-sensitivity control](../runs/alphabrain_cosmos_policy_real_input_sampler_action_control_v0/summary.json)
- [AlphaBrain corrected held-out forecast](../runs/alphabrain_cosmos_policy_real_input_sampler_fixed_action_forecast_v0/summary.json)
- [AlphaBrain expanded 21-window forecast matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_v0/aggregate.json)
- [AlphaBrain independent noise-seed replicate](../runs/alphabrain_cosmos_policy_real_input_sampler_noise2_v0/summary.json)
- [AlphaBrain independent seed-3 forecast matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_matrix_seed3_v0/aggregate.json)
- [AlphaBrain task-stratified matrix](../runs/alphabrain_cosmos_policy_real_input_sampler_forecast_by_task_v0/aggregate.json)
- [Bounded learned goal-free context baseline](../runs/alphabrain_goal_free_context_v1/summary.json)
- [Learned context seed-23 replicate](../runs/alphabrain_goal_free_context_v2/summary.json)
- [AlphaBrain corrected three-family LIBERO control pilot](../runs/alphabrain_cosmos_libero_short_control_fixed_action_v1/aggregate_summary.json)
- [AlphaBrain 608-step LIBERO control pilot](../runs/alphabrain_cosmos_libero_full_control_v1/aggregate_summary.json)
- [AlphaBrain corrected official-preprocess 608-step pilot](../runs/alphabrain_cosmos_libero_full_control_v2/spatial_600.json)
- [AlphaBrain matching LIBERO-Goal task0–2 aggregate](../runs/alphabrain_cosmos_libero_goal_control_v0/aggregate_summary.json)
- [主结果聚合](../runs/goal_bus_main_results_v0/summary.json)
- [主线配置与威胁模型](../configs/world_model_goal_bus_security.json)

所有本地测试通过；本稿没有依赖外部模型 API。下一阶段只应在 clean task-capable checkpoint 或训练过的 goal-free dynamics head 可用后，扩展多 seed 长时域任务矩阵。
