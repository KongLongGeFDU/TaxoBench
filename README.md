<h1 align="center">TaxoBench</h1>

<h3 align="center">
  Can Deep Research Agents Retrieve and Organize?<br>
  Evaluating the Synthesis Gap with Expert Taxonomies
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2601.12369"><img src="https://img.shields.io/badge/Paper-arXiv-blue.svg?style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/konglongge/TaxoBench"><img src="https://img.shields.io/badge/Dataset-Hugging_Face-yellow.svg?style=for-the-badge" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache--2.0-blue.svg?style=for-the-badge" alt="Code License"></a>
  <a href="DATASET_LICENSE.md"><img src="https://img.shields.io/badge/Data-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge" alt="Data License"></a>
</p>

<p align="center">
  <b>72 Expert Survey Taxonomies</b> · <b>3,815 Expert-Cited Papers</b> · <b>Reproducible Offline Scoring</b>
</p>

> **Note:** For the Chinese version, please refer to [README_zh.md](README_zh.md).

## 🔔 News

- 📦 **[2026-08]** The public benchmark dataset and offline scoring toolkit are released on [GitHub](https://github.com/KongLongGeFDU/TaxoBench) and [Hugging Face](https://huggingface.co/datasets/konglongge/TaxoBench).
- 📄 **[2026-01]** The TaxoBench paper is available on [arXiv](https://arxiv.org/abs/2601.12369).

## 📚 Overview

**TaxoBench** evaluates whether Deep Research systems and language models can move beyond finding relevant papers to constructing coherent, expert-like survey taxonomies. It measures two complementary capabilities:

- **Retrieval:** recovering the papers selected and cited by domain experts.
- **Organization:** clustering retrieved papers and arranging them into meaningful hierarchical knowledge structures.

The benchmark is grounded in **72 highly cited computer-science survey papers** and their expert-authored taxonomies, covering **3,815 precisely classified cited papers**. The public release provides title-and-abstract inputs, expert reference taxonomies, a documented prediction schema, and deterministic offline scoring utilities.

## ✨ Highlights

- **Expert-grounded evaluation** — compares model outputs against the cognitive structures used by survey authors rather than synthetic taxonomies.
- **Retrieval-to-synthesis coverage** — evaluates paper coverage, leaf-level organization, and hierarchy-aware semantic paths.
- **Ready-to-use public data** — includes 72 benchmark instances and corresponding Bottom-Up prompts for the Title + Abstract condition.
- **Reproducible scoring** — provides a lightweight Python package and a single CLI for local evaluation.
- **Clear release boundary** — excludes model outputs, reasoning traces, human-baseline raw annotations, and private provenance data.

## 🗂️ Repository Structure

```text
TaxoBench/
├── dataset/
│   ├── data.jsonl                         # 72 instances + expert taxonomies
│   ├── prompts_title_abstract.jsonl       # public Bottom-Up input prompts
│   └── SCHEMA.md                          # dataset and prediction formats
├── taxobench/
│   ├── score.py                           # command-line scoring entry point
│   └── metrics/                           # retrieval, clustering, Sem-Path
├── examples/
│   ├── toy_reference.jsonl
│   └── toy_prediction.jsonl
├── tests/
│   └── test_smoke.py
├── DATASET_LICENSE.md
└── pyproject.toml
```

## 🚀 Quick Start

### Installation

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

### Score Your Predictions

```bash
taxobench-score \
  --data dataset/data.jsonl \
  --predictions your_predictions.jsonl \
  --output scores.jsonl
```

The command prints aggregate scores to stdout. When `--output` is provided, it also writes per-survey results as JSONL.

## 💾 Data and Prediction Format

Each line in `dataset/data.jsonl` represents one survey benchmark instance:

- `id` — integer survey identifier.
- `survey` / `survey_topic` — source survey metadata and topic.
- `pdfs` — expert-cited papers with public `title` and `abs` fields.
- `gt_paper_count` — number of expert-cited papers.
- `gt` — expert reference taxonomy.

A prediction file should contain one JSON object per line:

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

`retrieved_papers` is optional; when omitted, the scorer extracts paper titles from `hierarchy_tree`. See [dataset/SCHEMA.md](dataset/SCHEMA.md) for the complete format.

## 📊 Evaluation Metrics

- **Retrieval Recall / Precision / F1** — measures coverage and precision against expert-cited papers.
- **ARI / V-Measure / Homogeneity / Completeness** — evaluates leaf-level paper clustering after title alignment.
- **Sem-Path** — measures whether aligned papers follow expert-like root-to-leaf semantic paths.

The default title-alignment threshold is `0.92` and can be changed with `--threshold`.

## 🔐 Public Release Boundary

This repository is intended for benchmark access and offline evaluation. It intentionally does **not** include:

- model-generated taxonomies or Deep Research product logs;
- generated summaries, core-task descriptions, or contribution extractions;
- participant-level human-baseline annotation trees;
- raw API responses, reasoning traces, private endpoints, or credentials;
- internal provenance archives or aggregate submission result tables.

The public `dataset/data.jsonl` contains only paper titles, abstracts, and expert reference taxonomies.

## 📝 Citation

If you find TaxoBench useful, please cite our work:

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

## 📄 License

- **Code:** [Apache License 2.0](LICENSE).
- **Original TaxoBench annotations:** [CC BY-NC 4.0](DATASET_LICENSE.md) for non-commercial research use.
- **Third-party paper metadata and abstracts:** retain their original rights.

## 🤝 Contributing

Issues and pull requests are welcome. For questions about the benchmark or data format, please open a GitHub issue.

---

<p align="center">
  <b>TaxoBench</b> · Fudan University NLP Lab
</p>
