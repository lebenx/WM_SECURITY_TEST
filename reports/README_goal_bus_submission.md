# 世界模型安全：当前工作稿与历史材料

**研究 Goal 尚未完成，当前没有可提交的完整安全论文。** 本入口已替换旧提交包说明；旧版连同原 TeX/PDF 保存在 [historical_goal_bus_package_20261015/](historical_goal_bus_package_20261015/)。

## 当前证据与工作稿

- [英文工作稿 PDF](world_model_decision_audit_working.pdf)及[可编辑 TeX](world_model_decision_audit_working.tex)：按研究论文结构整理已核实的前提、方法、结果、反证和局限；不是投稿就绪版本，不宣称已达到新颖性目标。
- [研究进度页](research_progress.html)与[研究资产索引](research_archive/index.html)。
- [物理不变性规范](../configs/goal_bus_physical_invariance_spec.json)：固定 belief 与完整执行控制的干预契约，不能直接推广成训练条件预测的 Goal 独立性。
- [DINO 30 步真值对照](dino_first_query_terminal_truth_decision_20261015.json)：局部排名翻转和真实回报损失成立，但两候选预测均更准，真实未来原生评分仍偏好低回报动作。
- [Cosmos 首选择真实后果](cosmos_first_selection_mediation_20261015.json)：既有开发面板中一个成功到失败案例，首动作块改选足以造成该失败；不是完整中介或泛化结论。
- [新增 Cosmos 真实未来替换检验](cosmos_first_query_truth_decision_20261015.json)：同一已知失败例中，Goal-Swap 增大两个候选的视觉预测误差并改选失败分支；真实未来替换恢复成功候选排名。仅一个事后诊断，续接效用来自历史环境回放，不是新防御恢复回合。上方工作稿 PDF 早于此补测，尚未收录。
- [当前机器可读索引](goal_bus_submission_manifest.json)：顶层纠正优先，继承的历史指标不可用作现行证据。

## 当前判断

WorldAgen 的预测分支依赖不能直接当作其原生策略受损；AlphaBrain 的联合输出结构也不自动证明隔离的 Forecast→Action 因果贡献。Cosmos 有局部真实失败证据，DINO 有条件性累计回报损失及同初态恢复，但尚无普适的新安全机制、经强对照验证的有效新防御及独立泛化。

DINO 的新固定库只有一个初态、两条事后选取的返回均值动作。30 步是开环真实回放，原生策略每次执行 15 步后再规划；不能把它写成完整原生候选集、99 步中介分析、总体 ASR 或成功率损失。预测误差的方向仍可影响排序，误差下降不等于误差完全没有中介作用。

## 可复现范围

从项目根目录，复算新增 Cosmos 真值检验的历史轨迹绑定、排名、效用和结果表：

```bash
python scripts/analyze_cosmos_first_query_truth.py --verify-only
```

此命令已通过，仅使用 Python 标准库读取缓存，不运行环境或模型。真实回放与推理代码及预算见[冻结协议](../configs/cosmos_first_query_truth_countercheck_20261015.json)；运行器拒绝覆盖既有输出。

从项目根目录，只读复算已保存的评分分量：

```bash
.venv-jepa-native/bin/python scripts/analyze_dino_first_query_objective_decomposition.py --verify-only
```

该命令已精确核对缓存数值，不加载模型或调用环境/API。完整回放与编码分别见[冻结回放协议](../configs/dino_first_query_full_horizon_replay_20261015.json)、[编码协议](../configs/dino_first_query_terminal_truth_20261015.json)及各报告。运行器拒绝覆盖已保存结果；不要为了文稿更新重跑实验。

编译当前工作稿：

```bash
cd reports
tectonic --only-cached --keep-logs world_model_decision_audit_working.tex
```

本机 `pdflatex` 缺少 format 文件，因此使用已有 Tectonic 缓存编译；该命令不会下载模板或宏包。

旧 `goal_bus_paper_draft_en.*`、中文稿、完成审计、复现清单和旧图表是历史材料。给旧 PDF 增加告示不修复其旧实验协议，也不把旧内容重新验证为正确。Gemini 只用于定义明确的实验；不用于 idea、评审或新颖性打分。
