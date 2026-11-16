# HSNN 第一阶段实施与审查说明

后续精简复核（2026-11-09）：每层 `forward` 仅校验一次关联结构，公开的映射生成与传播接口仍独立校验输入；共用内部计算路径，不缓存拓扑或映射。保留层级 `hidden_dim` 的调用兼容性，明确其不控制映射预测器宽度；移除变体参数字典的不必要复制。未改变传播公式、参数布局或已有图模型。

本轮新增 7 项前向与梯度等价检查，运行 HSNN 算子测试及整个 `tests/nsd`：335 项通过；Ruff、修改文件格式检查及全仓 ty 通过。下文 117 项专项、4379 项全仓及 wheel 结果是精简前的验证快照，旧审查 ZIP/补丁也不包含本轮修改；本轮未重建发行包或重跑全仓测试，以当前工作区为准。

本阶段交付无向、实数、单位超边权重的 Hypergraph Sheaf Neural Networks 库实现。代码位于 `feat/hsnn` 工作区，参考 Notebook 8 的数学定义，采用现有 NSD 的文件组织、变体工厂、参数初始化和模型封装方式。

实现包含对角、一般矩阵、正交和纯低秩四种限制映射。正式数据集复现、DHSNN、消融与论文效果判断属于后续阶段。当前合成训练仅证明实现可优化。英文 API、公式和环境修复细节见 [HSNN 使用文档](hsnn.md)。

## 1. 本次应包含的文件

| 文件 | 用途 |
|---|---|
| `src/sheaf_mpnn/hypergraph_sheaf_conv.py` | 独立超图基类、输入校验、度归一化和两阶段传播 |
| `src/sheaf_mpnn/nsd/hsnn_layers.py` | 四种关联映射参数化 |
| `src/sheaf_mpnn/nsd/hsnn_model.py` | `HSNNVariant` 与 encoder/layers/decoder 模型 |
| `src/sheaf_mpnn/__init__.py` | 增补顶层公开导出 |
| `src/sheaf_mpnn/nsd/__init__.py` | 增补 NSD 包内导出 |
| `tests/test_hypergraph_sheaf_conv.py` | 独立矩阵、能量、梯度和边界条件验证 |
| `tests/nsd/test_hsnn_layers.py` | 变体、置换性质和 CPU/CUDA 一致性验证 |
| `tests/nsd/test_hsnn_model.py` | 模型、序列化、参数梯度和短训练验证 |
| `scripts/smoke_hsnn.py` | 无下载的合成超图训练检查 |
| `docs/hsnn.md` | 英文 API、数学约定、验证快照及 Householder 修复说明 |
| `docs/hsnn-implementation-zh.md` | 本审查说明 |
| `docs/api.md` | 文档入口 |
| `notebooks/README.md` | Notebook 8 对应的库实现入口 |
| `CHANGELOG.md` | Unreleased 新增功能记录 |

旧 NSD 层、模型、归一化方法、实验数据管线和依赖文件均保持原实现。新增基类直接继承 PyG `MessagePassing`，因为旧 `BaseSheafConv` 的抽象接口依赖普通图中的两个节点端点；独立基类保留节点—超边的实际语义。

工作区原有的 `notebooks/0_GNNs_Basics.ipynb` 改动和 `docs/heterogeneous-implementation-zh.md` 不属于 HSNN 交付，不应纳入本次补丁或提交。

## 2. 输入、输出与编号空间

| 张量 | 形状 | 含义 |
|---|---|---|
| `x` | `[N, input_features]` | 原始节点特征 |
| `x_feat` | `[N, context_dim]` | 映射预测上下文 |
| `x_stalk` | `[N, d, f]` | 节点 stalk 状态 |
| `hyperedge_index` | `long[2, K]` | 第一行节点编号，第二行超边编号 |
| 对角映射 | `[K, d]` | 每条关联的对角元素 |
| 其他映射 | `[K, d, d]` | 每条关联的矩阵 |
| 模型输出 | `[N, out_channels]` | 原始 logits |

节点和超边独立编号，允许 `N != M`。同一列两个编号数值相等也属于合法关联，不能调用普通图的自环删除或 `to_undirected`。

`num_hyperedges=M` 保留尾部空超边；未指定时从非空索引推断，空索引推断为零。支持孤立节点、空超边、单节点超边和未排序关联。重复的 `(node, hyperedge)` 报错；成员相同但编号不同的超边合法。首版要求至少一个节点，支持 float32/float64 和单个超图的全批训练。

## 3. 需要优先审查的数学约定

每条关联只有一个映射 $F_{ve}$；节点向超边使用该映射，返回使用其转置。令概念上的块关联矩阵为 $B_{ev}=F_{ve}$：

$$
D_v=\sum_{e\ni v}F_{ve}^{\top}F_{ve},\quad
D=\operatorname{blockdiag}(D_v),\quad
D_E=\operatorname{blockdiag}(|e|I_d).
$$

$$
L=D-B^{\top}D_E^{-1}B,\qquad
R=(D^{\dagger})^{1/2},\qquad
P=RB^{\top}D_E^{-1}BR.
$$

修正后的对角块为 $L_{vv}=\sum_{e\ni v}(1-|e|^{-1})F_{ve}^{\top}F_{ve}$。

默认 `degree_regularization=0` 使用伪逆平方根；正数 $\tau$ 则使用 $(D+\tau I)^{-1/2}$。截断阈值为每个节点度块最大特征值乘以 `d * finfo(dtype).eps`；零块返回零。奇异情况下 $RDR$ 是支撑子空间投影，不能把 $RLR$ 无条件写成 $I-P$。

每层更新为：

$$
Z=W_1\operatorname{Dropout}(X)W_2,\qquad
X^+=\operatorname{ELU}(PZ+rX).
$$

默认 `residual=False`，即 $r=0$；启用后 $r=1$。这里沿用现有库的组织与初始化风格，更新公式采用 HSNN 传播形式，不能与旧 NSD 的 `X - alpha * ELU(LZ)` 当成相同架构。

## 4. 与 Notebook 8、参考代码的对应

- 第 3.2、5.1 节提供未归一化算子与稠密验证依据。
- 第 6 节演示使用 `diag(L)` 归一化；本实现选择 `sum(F.T @ F)` 的 sheaf degree，差异已记录。
- 第 7 节是带 TODO 的学生框架；库实现补齐相应职责，原 Notebook 保留练习形式。
- 算子参考 ICLR 2026 修正定义的无向 `q=0` 情形，未移植旧 `real_sheaf_layers.py` 的对角符号翻转。
- 超边大小均为 2 时，上述未归一化 $L$ 是通常图 sheaf Laplacian 的一半；测试明确保留该系数。
- 单位映射恢复选定的对称 HGNN 传播算子；结论针对算子，不代表不同激活、残差配置下完整网络输出相等。

参考代码固定到 `DirectionalSheafHypergraphs` 提交 `d94f81bc7d86bd8d41e416083063f3446c1ad796`。本阶段没有宣称逐项复现该仓库的完整架构或实验设置。

## 5. 参数化与传播

每层将当前状态展平为上下文，对超边成员求均值，再拼接节点与超边上下文。单个 bias-free Linear 加 Tanh 生成关联参数；映射与归一化均保留梯度。

| 变体 | 参数化 | 说明 |
|---|---|---|
| `DIAGONAL` | 对角向量 | 使用逐元素乘法 |
| `GENERAL` | 完整矩阵 | 支持 stalk 通道混合 |
| `ORTHOGONAL` | 现有 Cayley/Householder 系列工具 | 映射受正交约束 |
| `LOW_RANK` | $AB^{\top}$ | `1 <= rank <= d`，没有附加对角项 |

正交和低秩属于结构化选择，不构成表达能力严格递增的序列。低秩采用当前库的纯因子分解定义。

传播依次执行右归一化、节点到超边传输、按基数求均值、转置映射返回节点、左归一化。主路径不构造全局 Laplacian，也不枚举超边内部节点对。重复关联校验使用 `unique`，排序与设备同步成本应计入实际运行时间；本阶段不提供大规模效率结论。

## 6. 收尾复核发现与处理

原 GPU 对照只覆盖 `degree_regularization=0.1`。本次增加默认值 `0.0`，覆盖无正则伪逆路径，包括低秩度块。CPU/CUDA 前向与参数梯度对照均通过；新增测试共 117 项，其中 14 项为 CPU/CUDA 一致性配置。

发行 wheel 已独立验证：包含全部 HSNN 模块，不含环境、数据和临时缓存，并在解压出的 wheel 中导入四种模型完成前向与反向。检查显式确认模块来自 wheel，而非源码的 editable 安装。

本次静态复核与数值检查范围内，未发现需要修改已有算法实现的阻塞问题。低秩伪逆在数值秩变化处的敏感性、真实数据效果和更大规模训练，仍需后续实验评估。

## 7. 验证与复现

使用项目 `.venv`，Python 3.13.18、PyTorch 2.8.0+cu128、PyG 2.7.0，GPU 为 GTX 1650。Householder C++ 扩展已针对当前 PyTorch 重编译；旧二进制与安装记录备份在 `tmp/householder-torch28/backup/`，不属于代码交付。

| 检查 | 最终结果 |
|---|---|
| HSNN 专项测试 | 117 项通过，包含 14 项 CPU/CUDA float64 前向与梯度对照 |
| 全仓非 integration 回归 | 4379 项通过、7 项跳过、760 项 integration 测试未运行 |
| 全仓 statement coverage | 94% |
| Ruff、格式与全仓 ty | 全部通过 |
| wheel 内容与独立导入 | 157 个归档成员仅来自库包及发行元数据；四种变体前后向通过 |

7 项跳过中，6 项属于不适用的参数或 identity 配置，1 项因网络沙箱无法下载 Cornell 图参考数据；HSNN 测试没有跳过。全仓回归将 Householder 回退警告视为错误，验证使用的是已修复的 C++ 后端。

wheel 验证使用的是当前 Windows/Python 3.13 环境，不等于完成所有操作系统和 Python 版本的兼容矩阵。

PowerShell 复现命令：

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_hypergraph_sheaf_conv.py tests/nsd/test_hsnn_layers.py tests/nsd/test_hsnn_model.py --no-cov
.\.venv\Scripts\python.exe scripts/smoke_hsnn.py --variant all --steps 60 --seed 42 --device cpu
.\.venv\Scripts\python.exe scripts/smoke_hsnn.py --variant all --steps 60 --seed 42 --device cuda
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\ty.exe check .
```

CPU/GPU 合成损失与详细数学检查结果也记录在英文文档中。这些检查验证正确性与可训练性，不构成优于基线或足以纳入论文的实验证据。

## 8. 审查交付方式

审查包基于 `72434d18e616d3cd637b2d61c12c9537353205a0`，包含第 1 节列出的 14 个文件、补丁和 SHA-256 清单。它不包含环境、数据、编译二进制、用户原有 Notebook 改动或异构说明草稿。

在该基准提交的干净副本中，先对补丁执行 `git apply --check` 再审查。当前工作区已包含实现，不应重复正向应用补丁。审查包在当前工作区进行反向 dry-run 检查，确认补丁与交付一致；该检查不撤销工作区内容。

建议 PR 标题：`Add undirected hypergraph sheaf message passing and HSNN models`。

建议 PR 描述：

> Add incidence-wise, two-stage undirected HSNN propagation and encoder/layer/decoder models with diagonal, general, orthogonal and pure low-rank restriction maps. The implementation uses the corrected hypergraph Laplacian and explicit symmetric sheaf-degree normalization, without changing existing graph methods. Independent dense-operator and gradient checks, CPU/CUDA parity tests, synthetic optimization smoke tests and a wheel import check cover the new API. Formal benchmark reproduction and directed hypergraphs remain subsequent work.

优先审查数学约定、奇异度归一化、映射返回方向及与旧 NSD 的接口边界，再检查封装、测试和文档。是否进入论文实验阶段，应由后续真实数据结果决定。
