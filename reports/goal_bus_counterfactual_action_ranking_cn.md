# Goal-Bus Counterfactual Action Ranking Test (superseded)

> **Correction notice.** The first version of this report used the condition-specific Goal as the CLIP scoring text, which violated the fixed-scoring control. Those ranking and utility numbers are retained only as a debugging record and must not be cited. The corrected experiment uses the original Clean goal as the scoring text for every condition; see [the corrected ICLR research report](goal_bus_iclr_research_report_cn.md) and [`summary_fixedscore_tasks6.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary_fixedscore_tasks6.json).

## 结论先行

当前证据对“原生系统中的决策级损害”给出 **No-Go**。代码审计显示，WorldAgen 和 AlphaBrain 都由 policy/action head 直接产生动作，World Forecast 在原生 rollout 中只在动作之后生成或被评分，没有进入 Action Selection。

把 WorldAgen 接入一个明确标注的独立 forecast-driven planner 后，三项 LIBERO 任务上 Goal-Swap 的候选动作排序相对 Clean 的平均 pairwise flip rate 为 **0.3929**（SD 0.1557）。这证明合法目标切换能够污染这个新增 planner 的排序输入，但不等于原生策略已被攻击。

四条件的 6 步真实 LIBERO 回放中，Clean、Goal-Swap、Goal-Swap + Bus-Cut、matched Control-Cut 的累计稀疏奖励全部为 0，成功率均为 0/3；因此本实验没有观测到 Selected-Action Utility 的下降或 Bus-Cut 带来的真实效用恢复。决策级损害仍 **未证明**。

## 可行性审计

审计结果见 [`runs/goal_bus_action_selection_audit_v0/summary.json`](../runs/goal_bus_action_selection_audit_v0/summary.json)。

| 项目 | 结果 |
| --- | --- |
| WorldAgen checkpoint | 存在，SHA256 已记录 |
| WorldAgen 原生动作路径 | `step_agent` 直接由 policy head 输出动作；随后才调用 `step_world_model` |
| AlphaBrain 原生动作路径 | `action_from_latent(denoised[:, :, 4])` 直接解码动作；future-world slot 只作为预测探针 |
| 原生 World Forecast 是否参与动作选择 | 两者均为否 |
| 结论 | 必须把新增 planner 与原生系统分开报告 |

## 最小实验设计

脚本为 [`scripts/pilot_worldagen_counterfactual_action_ranking.py`](../scripts/pilot_worldagen_counterfactual_action_ranking.py)。它使用 WorldAgen 和 LIBERO，固定：

- 同一真实初始状态、task、随机种子和 8 个候选动作轨迹；候选集合只由一次 Clean policy 输出加固定 CPU seed 的扰动生成，之后四个条件完全复用。
- 原始 Clean goal 作为所有候选评分的目标文本；WorldAgen 前向中的 goal 条件按四个条件切换，但 planner 只读取 world forecast、完全丢弃 policy head 输出，因此 policy goal 不会重新生成候选动作，也不改变真实回放任务。
- 四个条件：`Clean`、`Goal-Swap`、`Goal-Swap + Bus-Cut`、`Goal-Swap + matched Control-Cut`。
- 评分固定为 WorldAgen 预测 primary future image 的确定性 pseudo-image CLIP 与原始 goal 文本的相似度。未来真实 observation 不进入选择。
- 真实效用来自每个条件在同一 LIBERO 初始状态上的新鲜环境回放，记录累计稀疏奖励、最大奖励、成功和步数。

运行命令：

```bash
LIBERO_CONFIG_PATH=/tmp/libero_cfg \\
PYTHONPATH=vendor/LIBERO:vendor/WorldAgen \\
WORLDAGEN_RANKING_OUT=runs/worldagen_counterfactual_action_ranking_v0/summary.json \\
.venv-worldagen/bin/python scripts/pilot_worldagen_counterfactual_action_ranking.py \\
  --suite libero_goal --task-ids 0,1,2 --candidates 8 --horizon 6 --seed 17
python scripts/analyze_worldagen_counterfactual_action_ranking.py \\
  runs/worldagen_counterfactual_action_ranking_v0/summary.json \\
  runs/worldagen_counterfactual_action_ranking_v0/summary.json
```

## 结果

完整逐任务结果和聚合统计见 [`runs/worldagen_counterfactual_action_ranking_v0/summary.json`](../runs/worldagen_counterfactual_action_ranking_v0/summary.json)。误差是 WorldAgen patch space 的 MSE；每个条件的误差对应其被选中的候选动作，并与真实回放第 3、6 步图像比较。

| 条件 | Ranking flip rate 相对 Clean | Forecast Error MSE，均值 ± SD | 累计奖励，均值 ± SD | 成功率 |
| --- | ---: | ---: | ---: | ---: |
| Clean | — | 1.8708 ± 0.0404 | 0.0000 ± 0.0000 | 0/3 |
| Goal-Swap | 0.3929 ± 0.1557 | 1.8838 ± 0.0556 | 0.0000 ± 0.0000 | 0/3 |
| Goal-Swap + Bus-Cut | 0.3929 ± 0.0619 | 1.7815 ± 0.0388 | 0.0000 ± 0.0000 | 0/3 |
| Goal-Swap + Control-Cut | 0.3929 ± 0.1557 | 1.8838 ± 0.0556 | 0.0000 ± 0.0000 | 0/3 |

三项任务的 selected candidate 分别为：

| task id | Clean | Goal-Swap | Bus-Cut | Control-Cut |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 5 | 1 |
| 1 | 6 | 0 | 6 | 0 |
| 2 | 0 | 0 | 0 | 0 |

Goal-Swap 与 matched Control-Cut 的全部候选分数逐项完全相同（最大绝对差 0），说明这个 policy-only matched cut 没有改变新增 planner 的 World 分支评分。Bus-Cut 选中的动作与 Clean 在 2/3 任务匹配；但 matched Control-Cut 也在 2/3 任务匹配，且真实效用全为零，因此不能把 2/3 的动作匹配率解释成 Bus-Cut 的恢复成功。

## 统计与边界

聚合脚本为 [`scripts/analyze_worldagen_counterfactual_action_ranking.py`](../scripts/analyze_worldagen_counterfactual_action_ranking.py)。当前只有 3 个任务，统计仅作配对描述，不作显著性检验。6 步短回放对 LIBERO 稀疏奖励明显偏保守；它提供了真实后果观测，但没有覆盖完成任务所需的完整闭环时长。评分还使用了从 patch-normalized output 构造的确定性 pseudo-image，因此应把结果解释为一个可复现的独立 planner probe，而不是已验证的产品级动作价值函数。

## Go / No-Go 判断

| 判断对象 | 决策 | 已证明的内容 |
| --- | --- | --- |
| 原生 WorldAgen/AlphaBrain 是否因 World Forecast 污染而改变动作选择 | **No-Go** | 原生推理代码不把 World Forecast 接入 Action Selection |
| 新增 forecast-driven planner 是否出现预测级排序污染 | **Go（有限范围）** | 固定候选集合下，Goal-Swap 的排序 flip rate 非零，均值 0.3929 |
| 预测级污染是否已证明造成真实决策级损害 | **No-Go** | 四条件 LIBERO 回放奖励全为 0、成功率全为 0/3，没有效用差异证据 |
| Bus-Cut 是否已证明恢复真实决策 | **No-Go** | 仅有 2/3 selected-action 与 Clean 匹配，且 Control-Cut 同样 2/3；效用未恢复 |

所以当前 Goal-Bus 证据可以支持“合法 Goal-Swap 会改变 World Forecast，并能改变一个独立预测驱动 planner 的候选排序”，但不能支持“原生策略已遭受决策级攻击”，也不能把 Forecast Divergence 直接等同于攻击成功。下一步若要转为决策级 Go，需要把候选规划 horizon 扩展到足够产生非零真实奖励的闭环长度，并在成功的 Clean baseline 上预注册 utility loss、Bus-Cut recovery 和匹配控制的统计判据。
