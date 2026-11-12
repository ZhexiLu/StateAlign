# Third-party components

## Fetched by `third_party/fetch.sh` (not redistributed)

| Component | Source | License | Use |
|---|---|---|---|
| BFCL (Berkeley Function Calling Leaderboard) | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) @ `6ea5797` | Apache-2.0 | BFCL v4 multi-turn evaluation. `third_party/bfcl/bfcl.patch` adds one flag to the command that starts the local vLLM server (`--generation-config vllm`). The scorer, the data and the inference loop are unchanged. |

## Data

The task states in `data/statealign_example_100.jsonl` are constructed from trajectories of two public datasets.

| Dataset | Source | License |
|---|---|---|
| ToolMind | [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) | Apache-2.0 |
| LoopTool-23k | [zhangkangning/LoopTool-23k](https://huggingface.co/datasets/zhangkangning/LoopTool-23k) | Apache-2.0 |

## Models (downloaded by the user)

| Model | Source | License |
|---|---|---|
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 |

## Libraries

Training builds on [TRL](https://github.com/huggingface/trl) (`GRPOTrainer`), [PEFT](https://github.com/huggingface/peft)
and [vLLM](https://github.com/vllm-project/vllm). The pinned versions are listed in `env/`.
