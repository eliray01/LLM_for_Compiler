# LLM for Compiler

Experiment code for evaluating language models on **LLVM IR peephole optimization**, with formal verification via [Alive2](https://github.com/AliveToolkit/alive2).

This repository contains the supporting materials for the dissertation *Enabling reasoning mechanisms in Small Language Models for compiler optimizations* (Emil Fakhretdinov).

Dissertation PDF: [`Emil_Fakhretdinov_240897244_Dissertation.pdf`](./Emil_Fakhretdinov_240897244_Dissertation.pdf)

## Overview

Models are prompted (or fine-tuned) to transform unoptimized LLVM IR into optimized IR. Outputs are checked for semantic equivalence with Alive2 (`alive-tv`). Experiments cover:

| Approach | Folder | Description |
|----------|--------|-------------|
| Few-shot / CoT | `single/` | Single-model baselines (Qwen Instruct/Coder, DeepSeek-R1 distill) |
| Dual-model | `dual/` | Reasoner + coder pipelines (Qwen–Qwen, GPT–Qwen) |
| APE | `ape/` | Automatic Prompt Engineering with Alive2 scoring |
| RL (exploratory) | `rl/` | Ad-hoc PPO / GRPO prototypes |

## Repository layout

```
.
├── alive2.py              # Alive2 wrapper (importable from repo root)
├── alive/                 # Same wrapper kept for reference
├── ape/                   # Automatic Prompt Engineering pipeline
├── dual/                  # Dual-model (reasoner + transformer) scripts
├── single/
│   ├── FS/                # Few-shot prompting
│   ├── CoT/               # Chain-of-Thought prompting
│   └── Distilled/         # Distilled reasoning models
├── rl/                    # Reinforcement learning prototypes
├── processed_data/        # HuggingFace dataset (unopt / opt IR pairs)
├── requirements.txt
└── README.md
```

## Requirements

### 1. Alive2

Install [`alive-tv`](https://github.com/AliveToolkit/alive2). On macOS with Homebrew:

```bash
brew install alive2
```

Update `ALIVE_TV_PATH` / `alive_tv_path` in each script to match your install (e.g. `/opt/homebrew/bin/alive-tv` or `/home/linuxbrew/.linuxbrew/bin/alive-tv`).

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

GPU (CUDA) is recommended for the larger Qwen models. Scripts use Hugging Face `transformers` with `device_map="auto"`.

### 3. OpenAI API (dual / APE only)

For `dual/gpt*.py` and `ape/ape_pipeline.py`, set your key before running:

```bash
export OPENAI_API_KEY="sk-..."
```

Replace placeholders such as `YOUR_OPENAI_API_KEY` in those scripts, or wire them to `os.environ["OPENAI_API_KEY"]`.

## Dataset

Scripts expect a Hugging Face dataset on disk at `processed_data/` (`load_from_disk("processed_data")`). Each example includes unoptimized and optimized LLVM IR.

The main Arrow shard (`data-00000-of-00001.arrow`, ~151 MB) exceeds GitHub’s file size limit and is **not** in this repo. Place it locally under `processed_data/` alongside the committed `dataset_info.json` and `state.json` (from your dissertation supporting materials zip or local copy), then:

```bash
python processed_data/see_dataset.py
```

## How to run

Run scripts from the **repository root** so `processed_data` and `alive2` resolve correctly:

```bash
# Few-shot baseline
python single/FS/Qwen2.5-Coder-3B.py

# Chain-of-Thought
python single/CoT/Qwen2.5-Coder-3B_cot.py

# Dual-model (local reasoner + coder)
python dual/Qwen3B_QwenCoder3B.py

# Automatic Prompt Engineering
python ape/ape_pipeline.py
```

Typical knobs at the top of each script:

- `NUM_EXAMPLES` / loop bounds — how many IR pairs to evaluate  
- `OUTPUT_FILE` — CSV of results and Alive2 status  
- model names, `batch_size`, device map  

RL notebooks under `rl/` are exploratory and incomplete.

## Citation

If you use this code, please cite the associated dissertation / course submission:

**Emil Fakhretdinov** — *Towards LLM-Based Compiler Optimization* (Supporting Materials).

## License

Research / academic use. Alive2 is subject to its own license. Model weights (Qwen, DeepSeek, etc.) follow their respective licenses and usage terms.
