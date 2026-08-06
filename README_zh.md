# TaxoBench

TaxoBench 用于评测 Deep Research 系统和语言模型是否能检索专家引用的论文，并将论文组织为接近专家综述的层级 taxonomy。当前公开仓库只包含基准数据、数据格式说明和离线评测脚本。模型生成的辅助摘要、预测树、人类基线原始标注、API 原始响应、reasoning traces、内部归档和密钥均不公开。

论文：[Can Deep Research Agents Retrieve and Organize? Evaluating the Synthesis Gap with Expert Taxonomies](https://arxiv.org/abs/2601.12369)

数据集：[Hugging Face — `konglongge/TaxoBench`](https://huggingface.co/datasets/konglongge/TaxoBench)

## 安装

```bash
pip install -e .
```

## 评测预测文件

预测文件为 JSONL，每行包含 `id` 和 `hierarchy_tree`。具体格式见 `dataset/SCHEMA.md`。

```bash
taxobench-score --data dataset/data.jsonl --predictions your_predictions.jsonl --output scores.jsonl
```

内置 smoke test：

```bash
taxobench-score --data examples/toy_reference.jsonl --predictions examples/toy_prediction.jsonl
```

## 公开边界

本仓库不包含模型预测树、模型生成的摘要/核心任务/贡献文本、人类基线原始标注树、Deep Research 产品日志、API 原始响应、推理过程、私有 endpoint、密钥或内部 provenance 归档。公开的 `dataset/data.jsonl` 仅保留论文标题、摘要和专家参考 taxonomy。

## 许可

代码采用 Apache-2.0。TaxoBench 原创标注采用 CC BY-NC 4.0 供非商业研究使用。论文元数据、摘要等第三方内容保留原始权利。
