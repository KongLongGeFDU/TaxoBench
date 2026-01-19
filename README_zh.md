# TaxoBench: 一个面向科研文献的层级分类生成与评估基准

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red.svg)](https://arxiv.org/abs/2601.xxxxx)

---

## 📖 项目简介 (Introduction)

**TaxoBench** 是首个从**人类专家认知结构**角度，系统性评估 **Deep Research Agents** 与 **大语言模型（LLMs）** 在科研文献**组织、归纳与层级结构构建**能力上的基准测试框架。

本项目基于复旦大学 NLP 实验室论文：

> *Can Deep Research Agents Find and Organize?  
> Evaluating the Synthesis Gap with Expert Taxonomies*

### 📚 数据来源

- **72 篇高引用计算机科学综述论文（Survey Topics）**
- 专家人工构建的 **Taxonomy Trees**
- **3,815 篇被精确归类的引用文献**，作为 Ground Truth

### 🎯 评估模式

TaxoBench 实现了论文中定义的两种核心评估范式：

1. **Deep Research Mode**
   - 端到端评估：检索 → 筛选 → 组织 → 结构化总结

2. **Bottom-Up Mode（本仓库重点）**
   - 在给定文献集合的前提下  
   - 评估模型 **自下而上构建层级知识结构（Taxonomy）** 的能力

---

## 🌟 核心特性 (Key Features)

- **🧪 双层评估架构**
  - **Leaf-Level**：检索与聚类质量
  - **Hierarchy-Level**：分类树结构合理性

- **⚡ 高吞吐并发推理**
  - 基于 Python `multiprocessing`
  - 支持大规模并行

- **🧠 原生支持 Thinking / Reasoning 模式**
  - 适配推理增强模型：
    - DeepSeek-R1 / V3
    - Claude 4.5 Sonnet
    - Kimi-k2-Thinking 等

- **🔌 多模型统一接口**
  - OpenAI (GPT-5)
  - Anthropic (Claude 4.5)
  - Google (Gemini 3)
  - DeepSeek / Qwen / Moonshot (Kimi)

---

## 📂 项目结构 (Repository Structure)

```text
TaxoBench/
├── dataset/                  # 输入数据（72 个 Survey Topics + 3815 篇论文）
├── script/                   # 实验启动脚本（Bottom-Up Mode）
│   ├── eval_setting1.sh      # Setting 1: Title + Abstract
│   ├── eval_setting2.sh      # Setting 2: Title + Abstract + Summary
│   └── eval_setting3.sh      # Setting 3: Title + Abstract + Core-task & Contributions
├── setting_pipeline/         # 核心推理逻辑（Python）
│   ├── eval_setting1.py
│   ├── eval_setting2.py
│   └── eval_setting3.py
├── metric/                   # 评估指标
│   ├── get_clustering_result.py  # Get clustering alignment results
│   ├── get_clustering_metric.py  # Leaf-Level Metrics
│   ├── get_taxonomy_result.py    # Get hierarchical structure
│   ├── get_taxonomy_metric.py    # Hierarchy-Level (LLM Judge)
│   ├── ted.py                    # Tree Edit Distance
│   ├── nsp.py                    # soft nsp
│   ├── nsr.py                    # soft nsr
│   └── soft_f1.py                # Soft F1 (NSR / NSP)
└── results/                  # 实验结果输出
```

## 🧪 评测设定 (Evaluation Settings)

本仓库聚焦论文中的 **Bottom-Up Mode**，通过三种递进的信息粒度（Input Granularities）考察模型的组织能力。

### 🔹 Setting 1：基础评估
* **输入**：Title + Abstract
* **启动命令**：
    ```bash
    bash script/eval_setting1.sh
    ```
* **说明**：最基础设定，仅依赖表层语义信息，评估模型的初步组织能力。

### 🔹 Setting 2：基础评估 + 摘要
* **输入**：Title + Abstract + Summary
* **启动命令**：
    ```bash
    bash script/eval_setting2.sh
    ```
* **说明**：Summary 由 LLM 生成，包含研究问题、动机、方法等，评估更丰富语义是否提升分类质量。

### 🔹 Setting 3：基础评估 + 论文核心内容
* **输入**：Title + Abstract + Core-task & Contributions
* **启动命令**：
    ```bash
    bash script/eval_setting3.sh
    ```
* **说明**：
    * 使用专家抽取的 **核心任务与贡献**
    * 去除冗余描述，聚焦创新本质
    * 支持 Thinking 模式与自动纠错
    * → 最接近人类专家的认知组织方式

---

## 🚀 快速开始 (Getting Started)

### 1️⃣ 克隆仓库 & 安装依赖
```bash
git clone [https://github.com/KongLongGeFDU/TaxoBench.git](https://github.com/KongLongGeFDU/TaxoBench.git)
cd TaxoBench

pip install openai anthropic tqdm numpy pandas scikit-learn
```
### 2️⃣ 配置 API Key
在 `setting_pipeline/` 下的 Python 脚本中配置：

```python
from openai import OpenAI

client = OpenAI(
    base_url="[https://api.openai.com/v1](https://api.openai.com/v1)",
    api_key="sk-..."
)
```
### 3️⃣ 运行实验
修改 `script/eval_setting*.sh` 中的：

MODEL_PAIRS：模型列表

NUM_WORKERS：并发进程数

然后执行：

```bash

chmod +x script/eval_setting3.sh
./script/eval_setting3.sh
```

## 📊 评估指标 (Metrics)

本项目的 `metric/` 目录提供了论文中完整的评估工具 。

### 🧩 Leaf-Level Metrics（论文 / 聚类层级）

| 指标 | 说明 | 对应脚本 |
| :--- | :--- | :--- |
| **Recall** | (仅 Deep Research Mode) 衡量检索到的论文对专家选定核心文献的覆盖率。 | `get_clustering_result.py` |
| **ARI** | **Adjusted Rand Index**。衡量模型聚类结果与专家 Ground Truth 的一致性。 | `get_clustering_metric.py` |
| **V-Measure** | 包含 **Homogeneity** (纯度) 和 **Completeness** (完整性) 的加权平均。 | `get_clustering_metric.py` |

### 🌳 Hierarchy-Level Metrics（分类树结构）

| 指标 | 说明 | 对应脚本 |
| :--- | :--- | :--- |
| **TED** | **Tree Edit Distance**。计算模型树转换为专家树所需的最小编辑成本。 | `ted.py` |
| **Soft F1** | 基于 **NSR (Node Soft Recall)** 和 **NSP (Node Soft Precision)** 计算的软性 F1 分数。 | `soft_f1.py` |
| **LLM-as-a-Judge** | 调用 GPT-4o 从以下 4 个维度打分。 **Semantic Coverage** (语义覆盖). **Sibling Organization** (兄弟节点 MECE 原则). **Hierarchical Logic** (层级逻辑). **Structural Topology** (结构拓扑) | `get_taxonomy_metric.py` |

---

## 📝 引用 (Citation)

如果您在研究中使用了本代码或数据集，请引用我们的论文：

```bibtex
loading......
```
## 📄 License

本项目采用 Apache License 2.0 开源协议，详见 [LICENSE](LICENSE) 文件。
