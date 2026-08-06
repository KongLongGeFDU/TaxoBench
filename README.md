# TaxoBench

TaxoBench is a benchmark for evaluating whether deep-research systems and language models can retrieve expert-cited papers and organize them into expert-like survey taxonomies. This public repository contains only the benchmark data, schema, and offline scoring utilities. Raw model outputs, model-generated auxiliary summaries, human-baseline annotations, API responses, reasoning traces, and internal provenance archives are intentionally not released.

Paper: [Can Deep Research Agents Retrieve and Organize? Evaluating the Synthesis Gap with Expert Taxonomies](https://arxiv.org/abs/2601.12369)

Dataset: [Hugging Face — `konglongge/TaxoBench`](https://huggingface.co/datasets/konglongge/TaxoBench)

## Contents

```text
dataset/data.jsonl                 # 72 survey instances, title/abstract inputs, and expert taxonomies
dataset/prompts_title_abstract.jsonl
dataset/SCHEMA.md                  # data and prediction format
taxobench/                         # offline scoring package
examples/                          # tiny synthetic smoke-test files
DATASET_LICENSE.md                 # data-use terms
```

## Install

```bash
pip install -e .
```

## Score a Prediction File

A prediction file is JSONL with `id` and `hierarchy_tree` fields. See `dataset/SCHEMA.md`.

```bash
taxobench-score --data dataset/data.jsonl --predictions your_predictions.jsonl --output scores.jsonl
```

Run the built-in smoke test:

```bash
taxobench-score --data examples/toy_reference.jsonl --predictions examples/toy_prediction.jsonl
```

## Release Boundary

This repository is for dataset access and offline evaluation. It does not include:

- model-generated taxonomies or deep-research product logs;
- model-generated summaries, core-task descriptions, or contribution extractions;
- participant-level human-baseline annotation trees;
- raw API responses or reasoning traces;
- private credentials, endpoints, or local provenance archives;
- aggregate model result tables from the submission.

## Citation

```bibtex
@misc{zhang2026deepresearchagentsretrieve,
  title={Can Deep Research Agents Retrieve and Organize? Evaluating the Synthesis Gap with Expert Taxonomies},
  author={Ming Zhang and Jiabao Zhuang and Wenqing Jing and Kexin Tan and Ziyu Kong and Jingyi Deng and Yujiong Shen and Yuhui Wang and Zhenghao Xiang and Qiyuan Peng and Yuhang Zhao and Ning Luo and Renzhe Zheng and Jiahui Lin and Mingqi Wu and Long Ma and Zhangyue Yin and Shihan Dou and Maxm Pan and Tao Gui and Qi Zhang and Xuanjing Huang},
  year={2026},
  eprint={2601.12369},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.12369}
}
```

## License

Code: Apache-2.0. Original TaxoBench annotations: CC BY-NC 4.0 for non-commercial research use. Third-party paper metadata and abstracts retain their original rights.
