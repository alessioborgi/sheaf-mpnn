# Cora 超图实验：数据对齐与复现边界

## 审计结论（2026-11-09）

之前转换的 `cora-cocitation.npz` 来自官方仓库的 `hypergraph.pickle`，但这不是该论文 Cora 训练入口实际使用的拓扑。来源正确不代表实验协议正确。旧文件及其训练结果只能用于工程调试，不应与论文表格比较；新训练入口拒绝这种旧格式。

参照 [Directional Sheaf Hypergraph Networks 论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/83ccb398f3ce9c4d137011f36a03c7d4-Paper-Conference.pdf) 和 [官方代码固定版本](https://github.com/EmaMule/DirectionalSheafHypergraphs/tree/d94f81bc7d86bd8d41e416083063f3446c1ad796)，现使用 `cora.content`、`cora.cites` 重建数据。数据并非由 Alessio Borgi 的 sheaf-mpnn 仓库提供。

| 项目 | 旧转换文件 | 修正文件 |
| --- | ---: | ---: |
| 节点数 | 2708 | 2708 |
| 特征维数 | 1433 | 1433 |
| 类别数 | 7 | 7 |
| 超边数 | 1579 | 1565 |
| 节点—超边关联数 | 4786 | 6994 |
| 最大超边大小 | 5 | 167 |
| 平均超边大小 | 3.031 | 4.469 |

修正后的节点数、特征数、类别数、超边数及平均超边大小与论文附录表 8 对应。拓扑进一步通过官方加载器逐项核对，而非仅比较统计量。

## 构造及验证

`scripts/prepare_hsnn_cora.py` 固定上游提交及原始文本 SHA-256。按官方 `dataset_loaders/data_utils_direction.py` 的 `load_citation_dataset_direction`，以引用文件第一列节点为尾节点，将其对应的第二列节点集合与尾节点合为一条超边；保留首次出现的超边顺序，去重，忽略未知端点与自引用。没有额外添加单节点超边。

特征、标签、节点行顺序与该版本官方 `features.pickle`、`labels.pickle` 完全一致。正式准备脚本只读校验过的文本，不反序列化 pickle。NPZ 同时保存节点标识及头尾角色供追溯；当前无向 HSNN 使用关联支撑，不使用头尾符号。这对应移除方向信息的输入，不能称为已实现 DHSNN，也不意味着整个模型与论文 q=0 配置等价。

可选集成测试 `tests/integration/test_hsnn_cora_reference.py` 校验本地缓存的上游文件哈希，抽取并执行原样的官方加载和划分函数，比较全部关联及角色、特征、标签和十组划分。缺少缓存时测试会明确跳过，不访问网络。此次本地验证包含该测试，未跳过。

## 划分、模型选择与记录

- 冻结十组 50%/25%/25% 训练、验证、测试划分。使用种子 0 的连续 CPU `randperm` 流，与官方代码默认设置生成方式逐项一致；并非声称获得了作者论文运行时保存的划分。生成环境版本写入来源记录，服务器直接使用冻结文件。
- 默认最多 1500 轮，早停耐心值 200。严格按验证准确率提升保存检查点；相同准确率不以验证损失打破平局。最终从磁盘重新加载最佳检查点。
- 默认仅评估验证集。固定最终配置后才传 `--evaluate-test`，每个运行仅对选定检查点评估一次测试集。不得根据测试结果继续挑选配置。
- 汇总准确率以 0–1 表示；与论文百分数比较时乘 100。同时输出总体标准差 `ddof=0`（官方汇总口径）和样本标准差 `ddof=1`，不要混用。
- 每次运行记录配置、数据及代码哈希、实际划分、模型种子、环境版本、训练历史、最佳权重、耗时及 CUDA 峰值分配显存。计时包含本脚本自身的评估与记录开销，不直接与其他实现的计时比较。

## 当前尚未对齐的部分

这是在论文对应拓扑上的新增 HSNN 实验，不是完整 DSHN 复现。当前 HSNN 的映射学习器、激活、层数、隐藏维度、分类头、残差及归一化配置等与官方最佳 Cora 模型存在差异；默认配置也没有经过论文采用的贝叶斯搜索。模型初始化使用独立的 `seed + run`，与官方训练过程耦合的随机数流不同，且不保证跨硬件逐位一致。

因此，当前可以验证实现和实验流程，但不能据此宣称达到论文性能或优于基线。正式比较需让标准超图模型、固定映射消融和可学习映射 HSNN 使用同一冻结划分、相当调参预算及一致模型选择规则；完成验证集选型后再报告十次最终测试均值、标准差和配对差异。方向性实验属于后续 DHSNN 阶段。

## 本地准备与服务器运行

重新准备数据（需网络，输出目录必须尚无产物）：

```bash
python scripts/prepare_hsnn_cora.py --download --raw-dir data/cora-raw --output-dir data/cora-dshn
```

已准备的三个文件为 `data/cora-dshn/cora-dshn-q0.npz`、`cora-dshn-q0.splits.npz`、`cora-dshn-q0.provenance.json`，需一起上传。训练入口验证三者的绑定哈希，旧上传包不能替代它们。

已接入项目统一命令与配置、模型和预设注册表。使用项目虚拟环境里的命令，在项目目录运行：

```bash
sheaf run --preset hsnn_cora
```

等价方式为 `.venv/bin/python -m exp.cli run --preset hsnn_cora`，可避免 PATH 指向其他环境。默认使用 CUDA 0；以 `--hardware.cuda 4` 选择 GPU，以 `--hardware.cuda -1` 使用 CPU。`--optim.epochs`、`--cv.folds`、`--model.variant` 等沿用现有嵌套参数风格。输出默认位于新的 `runs/hsnn-cora-*` 目录，可通过 `--hypergraph.output` 指定；测试集开关为 `--hypergraph.evaluate-test`。

HSNN training uses `src/exp/hsnn_run.py` through `sheaf run --preset cora-hypergraph`. Graph presets retain their graph semantics. See [the unified sweep guide](user-guide/sweeps.md) for all hypergraph searches.

服务器辅助脚本同样调用上述新预设，并直接使用 `.venv/bin/python`，无需使用师兄的 Conda 环境。先通过 `nvidia-smi` 确认空闲 GPU，再替换示例编号 4：

```bash
cd /root/YYY/sheaf-mpnn
sheaf run --preset cora-hypergraph --hardware.cuda 4
```

这会运行十组完整的验证集实验。最终配置确定后，单独开启最终测试：

```bash
sheaf run --preset cora-hypergraph --hardware.cuda 4 --hypergraph.evaluate-test
```

不要重复运行测试集模式来选择超参数。脚本不会安装依赖或修改共享 Conda 环境。

## 此次验证范围

137 项相关测试通过，包括官方加载器与十组划分一致性、旧数据拒绝、数据哈希、错误划分拒绝、早停平局处理、测试集开关以及 HSNN 算子和模型测试。官方旧 pickle 加载产生两条弃用警告，不影响一致性结果。完整服务器性能实验尚未执行，短轮数本地训练仅用于确认运行链路，不能作为论文结果。

随后统一 CLI 接入验证：157 项针对性测试通过；`tests/exp` 回归测试 1439 项通过、2 项跳过（两组测试有重叠，不相加）。真实数据通过 `sheaf run --preset hsnn_cora --hardware.cuda -1 --optim.epochs 2 --cv.folds 1` 完成本地训练与检查点验证，测试集仍未评估。Ruff 与相关源文件的 ty 检查通过。
