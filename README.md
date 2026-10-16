# World Model Security Research

世界模型与 Agent 安全研究工作区，包含实验代码、冻结配置、环境回放结果、失败记录和研究资产索引。

**研究仍在进行，尚未形成可提交的完整安全论文。** 预测差异、局部动作翻转和单个开发案例不能直接解释为普适漏洞或总体攻击成功率。历史材料中的旧判断应结合后续纠正与反证阅读。

## 从哪里开始

- [研究进度](reports/research_progress.html)：下载后在浏览器打开 HTML。
- [研究知识库](reports/research_archive/index.html)：模型、论文、已验证现象、失败原因和开发资产。
- [当前实验入口](reports/research_archive/next_experiment_gate.json)：最新筛选结论和执行边界。
- [现有工作稿及证据局限](reports/README_goal_bus_submission.md)。
- `src/wm_security/`：复用模块；`scripts/`：实验、分析、资源获取脚本。
- `configs/`：实验协议；`runs/`：结果、轨迹、环境快照和分析输入；`tests/`：现有单元测试。

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
- 模型权重、下载的 Parquet 数据集及安装压缩包。

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
