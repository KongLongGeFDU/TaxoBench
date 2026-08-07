<h1 align="center">TaxoBench</h1>

<h3 align="center">
  深度研究智能体能否实现检索与组织？<br>
  利用专家分类体系评估知识综合能力差距
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2601.12369"><img src="https://img.shields.io/badge/Paper-arXiv-blue.svg?style=for-the-badge" alt="论文"></a>
  <a href="https://huggingface.co/datasets/konglongge/TaxoBench"><img src="https://img.shields.io/badge/Dataset-Hugging_Face-yellow.svg?style=for-the-badge" alt="数据集"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache--2.0-blue.svg?style=for-the-badge" alt="代码许可"></a>
  <a href="DATASET_LICENSE.md"><img src="https://img.shields.io/badge/Data-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge" alt="数据许可"></a>
</p>

<p align="center">
  <b>72 个专家综述分类体系</b> · <b>3,815 篇专家引用论文</b> · <b>可复现离线评测</b>
</p>

> **Note:** English version: [README.md](README.md).

## 🔔 动态

- 📦 **[2026-08]** TaxoBench 公开基准数据与离线评分工具已在 [GitHub](https://github.com/KongLongGeFDU/TaxoBench) 和 [Hugging Face](https://huggingface.co/datasets/konglongge/TaxoBench) 发布。
- 📄 **[2026-01]** TaxoBench 论文已发布于 [arXiv](https://arxiv.org/abs/2601.12369)。

## 📚 项目简介

**TaxoBench** 评测 Deep Research 系统和语言模型能否超越“找到相关论文”，进一步构建连贯、接近专家认知结构的综述分类体系。基准关注两项互补能力：

- **检索（Retrieval）：** 找回领域专家选择和引用的论文。
- **组织（Organization）：** 对论文进行聚类，并组织为有意义的层级知识结构。

TaxoBench 基于 **72 篇高引用计算机科学综述论文**及其专家构建的 taxonomy，覆盖 **3,815 篇被精确归类的引用论文**。公开版本提供标题与摘要输入、专家参考 taxonomy、预测格式说明以及确定性的离线评分工具。

## ✨ 核心特点

- **专家认知结构驱动** — 直接对齐综述作者构建的 taxonomy，而非合成分类体系。
- **覆盖“检索—综合”链路** — 同时评测论文覆盖率、叶节点组织质量与层级语义路径。
- **开箱即用的公开数据** — 包含 72 个基准实例及 Title + Abstract 条件下的 Bottom-Up prompts。
- **可复现离线评分** — 提供轻量 Python 包和统一命令行入口。
- **明确的公开边界** — 不发布模型输出、推理轨迹、人类基线原始标注及私有 provenance 数据。

## 🗂️ 项目结构

```text
TaxoBench/
├── dataset/
│   ├── data.jsonl                         # 72 个实例与专家 taxonomy
│   ├── prompts_title_abstract.jsonl       # 公开 Bottom-Up 输入 prompts
│   └── SCHEMA.md                          # 数据及预测格式
├── taxobench/
│   ├── score.py                           # 命令行评分入口
│   └── metrics/                           # 检索、聚类与 Sem-Path
├── examples/
│   ├── toy_reference.jsonl
│   └── toy_prediction.jsonl
├── tests/
│   └── test_smoke.py
├── DATASET_LICENSE.md
└── pyproject.toml
```

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/KongLongGeFDU/TaxoBench.git
cd TaxoBench
python -m pip install -e .
```

### Smoke Test

```bash
taxobench-score \
  --data examples/toy_reference.jsonl \
  --predictions examples/toy_prediction.jsonl
```

### 评测预测结果

```bash
taxobench-score \
  --data dataset/data.jsonl \
  --predictions your_predictions.jsonl \
  --output scores.jsonl
```

命令会将聚合指标输出到终端；指定 `--output` 后，还会将每个综述实例的详细结果保存为 JSONL。

## 💾 数据与预测格式

`dataset/data.jsonl` 中每一行对应一个综述基准实例：

- `id` — 整数形式的综述编号。
- `survey` / `survey_topic` — 来源综述元数据与主题。
- `pdfs` — 专家引用论文；公开字段为 `title` 和 `abs`。
- `gt_paper_count` — 专家引用论文数量。
- `gt` — 专家参考 taxonomy。

预测文件每行应包含一个 JSON 对象：

```json
{
  "id": 0,
  "hierarchy_tree": {
    "name": "Root",
    "subtopics": []
  },
  "retrieved_papers": []
}
```

`retrieved_papers` 为可选字段；若省略，评分器会从 `hierarchy_tree` 中提取论文标题。完整格式请参阅 [dataset/SCHEMA.md](dataset/SCHEMA.md)。

## 📊 评测指标

- **Retrieval Recall / Precision / F1** — 衡量专家引用论文的检索覆盖率与准确率。
- **ARI / V-Measure / Homogeneity / Completeness** — 在标题对齐后评估叶节点论文聚类质量。
- **Sem-Path** — 衡量已对齐论文是否遵循接近专家结构的根到叶语义路径。

默认标题对齐阈值为 `0.92`，可通过 `--threshold` 调整。

## 🔐 公开边界

本仓库用于公开基准访问与离线评测，明确**不包含**：

- 模型生成的 taxonomy 或 Deep Research 产品日志；
- 模型生成的摘要、核心任务或贡献抽取文本；
- 参与者级别的人类基线原始标注树；
- API 原始响应、推理过程、私有 endpoint 或密钥；
- 内部 provenance 归档及投稿中的聚合模型结果表。

公开的 `dataset/data.jsonl` 仅包含论文标题、摘要和专家参考 taxonomy。

## 📝 引用

如果 TaxoBench 对你的研究有所帮助，请引用：

```bibtex
@misc{zhang2026deepresearchagentsretrieve,
  title         = {Can Deep Research Agents Retrieve and Organize? Evaluating the Synthesis Gap with Expert Taxonomies},
  author        = {Ming Zhang and Jiabao Zhuang and Wenqing Jing and Kexin Tan and Ziyu Kong and Jingyi Deng and Yujiong Shen and Yuhui Wang and Zhenghao Xiang and Qiyuan Peng and Yuhang Zhao and Ning Luo and Renzhe Zheng and Jiahui Lin and Mingqi Wu and Long Ma and Zhangyue Yin and Shihan Dou and Maxm Pan and Tao Gui and Qi Zhang and Xuanjing Huang},
  year          = {2026},
  eprint        = {2601.12369},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.12369}
}
```

## 📄 许可

- **代码：** [Apache License 2.0](LICENSE)。
- **TaxoBench 原创标注：** 采用 [CC BY-NC 4.0](DATASET_LICENSE.md)，供非商业研究使用。
- **第三方论文元数据与摘要：** 保留其原始权利。

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request。如对基准或数据格式有疑问，请通过 GitHub Issue 联系我们。

---

<p align="center">
  <b>TaxoBench</b> · 复旦大学 NLP 实验室
</p>
