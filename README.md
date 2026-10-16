# World Model Security Research

世界模型与 Agent 安全研究工作区，包含实验代码、冻结配置、环境回放结果、失败记录和研究资产索引。

**研究已按用户要求暂停，目前只做归档与发布；尚未形成可提交的完整安全论文。** 预测差异、局部动作翻转和单个开发案例不能直接解释为普适漏洞或总体攻击成功率。历史材料中的旧判断应结合后续纠正与反证阅读。

## 从哪里开始

- [供 GPT／Gemini 离线阅读的单文件研究包](reports/research_archive/ai_reader.txt)：仓库链接读取失败时，可下载这个文本文件提供给模型。
- [当前主张与证据边界](CLAIM.md)、[实验登记](IDEAS.md)、[关闭方向](CLOSED.md)和[可复用资产](ASSETS.md)：增量研究记录优先于历史快照。R004完整任务回报改善；R005否定候选生成主导的解释；R006/R007均未达到预设实质损害门槛。R008脚本已准备但未运行，不自动续跑。
- [决策杠杆与目标失配论文稿](reports/decision_leverage_paper/main.pdf)：当前英文正文；[阅读与复现说明](reports/decision_leverage_paper/READING_GUIDE.txt)区分已有结果、缓存复算和待完成实验。
- [研究进度](reports/research_progress.html)：下载后在浏览器打开 HTML。
- [研究知识库](reports/research_archive/index.html)：模型、论文、已验证现象、失败原因和开发资产。
- [当前实验入口](reports/research_archive/next_experiment_gate.json)：最新筛选结论和执行边界。
- [现有工作稿及证据局限](reports/README_goal_bus_submission.md)。
- `src/wm_security/`：复用模块；`scripts/`：实验、分析、资源获取脚本。
- `configs/`：实验协议；`runs/`：结果、轨迹、环境快照和分析输入；`tests/`：现有单元测试。

## 研究知识库的使用与维护

[离线知识库](reports/research_archive/index.html) 可搜索机制、失败原因、模型、论文与路径，并按分类和机制结果筛选。新增的 [模型复用配方](reports/research_archive/resource_recipes.json) 汇集模型接入、环境、权重、缓存、正常能力、Baseline 和恢复限制；[复用指南](reports/research_archive/reuse_guide.json) 说明实验流程、指标定义及归档规则。

```bash
# 搜索科学记录，而不只是文件名
python scripts/research_archive.py search "GenSafeNav" --registry resource_recipes
python scripts/research_archive.py search "query" --registry mechanisms --status no_harm
# 找到实际文件
python scripts/research_archive.py find WorldAgen --area runs --limit 10
# 有新资料或结论后，更新相应 JSON 卡片，再重建和检查索引
python scripts/research_archive.py build
python scripts/research_archive.py check
```

`build` 和 `check` 不加载模型、不运行环境、不调用 API。新增资料按原目录保留，不重复复制权重；目录索引的覆盖不等于独立验证了所有历史实验。[覆盖与待整理队列](reports/research_archive/coverage.json) 明列未关联运行、未关联文件和缺失路径提及。现有卡片优先引用具体修正分析；旧稿保留原文及历史告示。知识卡间的自动关联只表示共同引用了一个具体本地文件，不能当作因果证据。

## 克隆和依赖

```bash
git clone https://github.com/lebenx/WM_SECURITY_TEST.git
cd WM_SECURITY_TEST
# 仅初始化当前实验所需的依赖，例如：
git submodule update --init vendor/tdmpc2 vendor/jepa-wms
```

38 个已有第三方 Git checkout 以精确 commit 的子模块形式保存，详见 [.gitmodules](.gitmodules) 和 [来源清单](reproducibility/vendor_sources.json)。它们的许可证与使用条件由各上游项目规定。部分按文件获取的源码保留在 `vendor/`，来源见该目录的 provenance 文件或相应获取脚本。

不同实验依赖不同 Python、GPU、仿真环境及上游版本，**没有统一的一键安装环境**。请先阅读目标脚本、配置及运行记录，按需建立环境。原机器的 `.venv-*` 和绝对路径不会自动适配到新机器。

不默认递归初始化所有子模块：上游 HeXA 的 `discovery-agents` 和 wav_minigrid 的 `src/gym-minigrid` 缺少 `.gitmodules` 映射；OpenWAM 的 Cosmos、opine-world 的 ARC 嵌套依赖在原工作区也未初始化。dm_alchemy 的 protobuf 生成文件需按上游构建流程重建。

## 权重、数据和上传范围

本仓库保存项目代码、实验协议、报告、文本结果、图表，以及可纳入 Git 的 NumPy / 环境快照等实验记录。为避免把整个本地运行环境放进 Git，未上传：

- API 密钥、访问令牌、`.env*`、私有连接配置及凭据文件。
- `.venv-*`、安装缓存、编译产物和运行日志。
- `models/`、`checkpoints/` 下的完整模型 checkpoint、下载的 Parquet 数据集及安装压缩包。

上传范围核查（2026-10-16，已发布提交 `9a8448c`）：完整模型 checkpoint 已排除，但实验记录并非全部不含参数。`runs/tdmpc_latent_origin_confirmation_v0/encoder_ln.npz` 含提取的 LayerNorm 参数；`runs/jepa_affine_query_boundary_20261016/constraints.npz` 含 LayerNorm、预测器投影参数及派生量；`runs/alphabrain_official_forecast_contract_audit_v1/official_checkpoint_prefix.bin` 是含部分张量数据的 512 KiB checkpoint 前缀。此外，`runs/tdmpc_latent_origin_rbf_v1/probe.npz`、`runs/tdmpc_latent_origin_rbf_v2/probe.npz`、`runs/tdmpc_latent_origin_v0/probes.npz` 保存已拟合的分析探针。它们随实验记录进入了上述提交，不能把“没有上传完整 checkpoint”表述为“没有上传任何模型参数”。

本次增量保留预测特征、真实回放和分析缓存；`runs/forecast_residual_calibration_v1/calibrator.npz` 是 R004 拟合的残差修正数组，也属于研究资产，不能将本仓库描述为“完全不含拟合参数”。R005 四臂真实观测/动作/奖励另导出为 `evaluation_arrays.npz`，不包含模型权重。

大型资产的本地相对路径及大小见 [外部资产清单](reproducibility/external_artifacts.jsonl)，模型来源及已有验证状态见 [模型索引](reports/research_archive/models.json)。清单用于定位资产，不表示全部资产都已提供公共下载链接；本地生成而未上传的权重需要原工作区备份或重新执行对应实验。

已有资源获取入口包括 `scripts/fetch_gensafenav_resources.py`、`scripts/fetch_safedreamer_mode_gate_assets.py` 和 `scripts/download_cosmos_policy_checkpoint.py`。先查看各脚本的参数、对应配置及下载大小，再按需获取。报告中的某些链接指向未上传的权重、缓存或旧机器绝对路径，这是研究快照的已知复现边界。

## 不调用 API 的结果核对

从项目根目录可以只读核对已保存的 Cosmos 真值检验结果：

```bash
python scripts/analyze_cosmos_first_query_truth.py --verify-only
```

该命令读取历史缓存，不重新执行模型、环境或 API 实验。更多复算方法及证据限制见 [工作稿说明](reports/README_goal_bus_submission.md)。不要将已有输出覆盖为新的实验。

## API 使用

API 客户端代码会保留，真实凭据由本地私有配置或环境配置提供，不包含在此仓库中。需要 API 的实验应使用自己的凭据，并事先明确调用预算。按当前协作约定，Gemini 只用于定义明确的实验，不用于 idea 生成、科研评审或 novelty 打分。
