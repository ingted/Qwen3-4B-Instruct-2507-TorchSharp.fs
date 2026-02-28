# Local Model Metadata Copy

This folder stores non-`.dat` model metadata copied from:
- `/models/qwen3-4b-instruct-2507-torchsharp`

Included files:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`

Purpose:
- Keep runner and script dependencies reproducible inside this repository.
- Allow `--model-dir` to point to a repo-local path when needed.

Not included:
- Large `.dat` weights are intentionally excluded from git/repo packaging.
- Use `/models/qwen3-4b-instruct-2507-torchsharp/*.dat` for runtime weights.
