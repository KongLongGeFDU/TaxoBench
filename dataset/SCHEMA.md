# TaxoBench Dataset Schema

Each line in `data.jsonl` is one survey benchmark instance.

- `id`: integer survey id.
- `survey`: source survey metadata as used by the benchmark.
- `survey_topic`: survey title/topic.
- `pdfs`: expert-cited papers. Public fields are `title` and `abs`. Local file paths and model-generated summaries are intentionally omitted.
- `gt_paper_count`: number of expert-cited papers.
- `gt`: expert reference taxonomy. Internal nodes use `name` and `subtopics`; leaf nodes contain `papers`.

`prompts_title_abstract.jsonl` contains the corresponding Bottom-Up prompts for the
Title+Abstract condition. It contains inputs only, not model responses.

Prediction files for `taxobench-score` should contain one JSON object per line with:

- `id`: matching survey id.
- `hierarchy_tree`: predicted taxonomy in the same tree format.
- optional `retrieved_papers`: list of retrieved paper titles for retrieval scoring. If absent, the scorer uses papers appearing in the predicted tree.
