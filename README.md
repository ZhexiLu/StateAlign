# StateAlign

Code for **StateAlign: Internalizing State-Aware Decision-Making for Reliable Multi-Turn Tool Agents via
Reinforcement Learning**.

Zhexi Lu<sup>1</sup>, Subhajit Chaudhury<sup>2</sup>, Tejaswini Pedapati<sup>2</sup>,
Keerthiram Murugesan<sup>2</sup>, Lei Yu<sup>1</sup>

<sup>1</sup>Rensselaer Polytechnic Institute, <sup>2</sup>IBM Research

**Model:** [ZhexiLu/StateAlign-8B](https://huggingface.co/ZhexiLu/StateAlign-8B) on Hugging Face.
**Paper:** coming soon.

> **Release status.** This is a partial release. It contains the core training code, a 100-state example of the
> training data, and the BFCL evaluation of StateAlign-8B. The full training set and the data construction pipeline
> are not included at this time.

## Overview

A tool-use agent has to decide what to do next from the current task state: which information is available, what
the listed tools can do, and how far the task has progressed. The right next action can be to execute a call, to ask
the user for a missing value, to retrieve it with a tool, to decline an infeasible request, or to stop calling tools
and answer.

StateAlign trains this decision with reinforcement learning on constructed task states. The data cover five types of
states (`EXEC`, `NEED_USER`, `NEED_FETCH`, `INFEASIBLE`, `STOP`), and a state is paired with a counterpart of the
same task in which one condition differs, so that a different next action is required. Training uses two signals.

**Outcome reward.** The policy acts in a short multi-step episode on each state. A question receives the stored user
information, a retrieval call receives the recorded tool result, and the policy acts again on the updated history.
For an episode $\tau$ that requires an execution,

```math
R_{\mathrm{out}}(\tau)=\gamma^{\,k-1}\,\alpha(\tau)\,c(\tau)
```

if the target calls are executed correctly, and 0 otherwise. Here $k$ is the step of the execution, $\gamma=0.8$,
$\alpha(\tau)=0.5$ if the executing step carries extra calls and 1 otherwise, and $c(\tau)$ indicates that the step
after the execution is also correct (1 when the state has no continuation). The rewards of the 8 rollouts sampled
per state give the advantages of a GRPO objective $\mathcal J_{\mathrm{out}}$ with a KL term to the initial model.

**State contrast.** The frozen initial model $\pi_0$ scores every generated token $y_t$ twice, once given the
current state and its reference path $(h, z)$ and once given the paired state and its reference path
$(\tilde h, \tilde z)$, with the same generated prefix $p_t$:

```math
r_t=\operatorname{clip}\bigl(\log\pi_0(y_t\mid h,z,p_t)-\log\pi_0(y_t\mid\tilde h,\tilde z,p_t),\,-1,\,1\bigr)
```

A positive score marks reasoning or action content that fits the current state, and a negative score marks content
that fits the counterpart. The scores weight a token-level policy-gradient term $\mathcal J_{\mathrm{pair}}$, and the
training objective is

```math
\mathcal J(\theta)=\mathcal J_{\mathrm{out}}(\theta)+\lambda\,\mathcal J_{\mathrm{pair}}(\theta),\qquad\lambda=0.1 .
```

The paired states and the reference paths are used for scoring only. The policy always acts from its tools and its
interaction history.

## Repository layout

```
configs/train/            training settings: the paper setting and a short run on the example data
data/                     100-state example of the training data, with a description of the row format
env/                      pinned package versions of the training and the evaluation environment
scripts/train/train.sh    launches one training run on a multi-GPU node
scripts/eval/             LoRA merge and the BFCL multi-turn evaluation
src/statealign/env/       episodes on task states: simulated feedback, continuation step, outcome reward
src/statealign/chat/      Qwen3 prompt rendering for the policy and for the state-contrast scorer
src/statealign/training/  the trainer (GRPO with multi-step rollouts and the state-contrast term) and its entry point
src/statealign/evaluation/  LoRA merge and the serving view used in the evaluation
third_party/              pinned BFCL commit, the fetch script and a one-flag patch
```

The package is used from the source tree. The scripts set `PYTHONPATH` to `src/` themselves.

## Setup

Training and evaluation use two separate environments. The pinned versions are listed in `env/`.

**Training environment** (Python 3.10; also used for the LoRA merge):

```bash
conda create -n statealign-train python=3.10 -y && conda activate statealign-train
pip install torch==2.8.0                 # first: flash_attn builds against the installed PyTorch
pip install -r env/requirements-train.txt
```

`flash_attn` is optional. Without it, remove its line from the requirements file and run training with `ATTN=sdpa`.

**Evaluation environment** (Python 3.12):

```bash
conda create -n statealign-eval python=3.12 -y && conda activate statealign-eval
pip install -r env/requirements-eval.txt
bash third_party/fetch.sh                # clones BFCL at the pinned commit and applies third_party/bfcl/bfcl.patch
pip install -e third_party/gorilla/berkeley-function-call-leaderboard
```

**Base model.** Download [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) to a local directory:

```bash
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

## Data

`data/statealign_example_100.jsonl` holds 100 task states in the format the training code reads: 40 `EXEC`,
20 `NEED_USER`, 15 `NEED_FETCH`, 10 `INFEASIBLE` and 15 `STOP`, of which 71 come with a paired state.
[data/README.md](data/README.md) describes every field and shows how to step through a state with the training
environment, which needs no GPU.

The states are constructed from trajectories of [ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) and
[LoopTool-23k](https://huggingface.co/datasets/zhangkangning/LoopTool-23k).

## Training

A short run on the example data checks the installation and exercises the whole training path: multi-step rollouts
on a vLLM engine colocated with the trainer, outcome rewards, state-contrast scoring, checkpointing.

```bash
MODEL=$PWD/models/Qwen3-8B NPROC=2 bash scripts/train/train.sh configs/train/example_qwen3_8b.sh
```

`NPROC` is the number of GPUs. The run takes 20 optimizer steps and writes `runs/example_qwen3_8b/checkpoint-{10,20}`
and `runs/example_qwen3_8b/final`. The episodes used in each step are logged to `runs/example_qwen3_8b/rollouts/`.
The run does not reproduce the paper's model.

The setting of the paper is in `configs/train/statealign_qwen3_8b.sh`. It needs the full training file, which is not
part of this release.

```bash
MODEL=$PWD/models/Qwen3-8B DATA=<training file> bash scripts/train/train.sh configs/train/statealign_qwen3_8b.sh
```

| Setting | Value |
|---|---|
| Base model | Qwen3-8B, thinking on |
| Adapter | LoRA, rank 64, alpha 128, dropout 0.05, attention and MLP projections |
| Optimisation | 400 steps, learning rate 2e-5 with 15 warm-up steps then constant, gradient clipping 0.2 |
| Batch | 320 episodes per step: 8 GPUs, micro-batch 2, gradient accumulation 20 |
| Rollouts | 8 per task state, temperature 1.0 |
| Episode | up to 4 model steps, 1,536 new tokens per step, 6,144 per episode |
| Outcome reward | discount 0.8 per step before the execution, factor 0.5 for extra calls |
| KL weight to the initial model | 0.04 |
| State-contrast term | weight 0.1, per-token clip 1.0 |
| Group selection | dynamic sampling for up to 3 rounds, 20% of groups reserved for successful paired groups |
| Hardware | eight 80 GB GPUs |

The number of episodes per optimizer step is `NPROC x PDBS x GA`, and `PDBS x GA` must be a multiple of the 8
rollouts per state. Checkpoints are written every `SAVE_STEPS` steps, and a restarted run continues from its newest
complete checkpoint. Logging to Weights & Biases is off unless `--wandb_project <name>` is passed.
`python -m statealign.training.train --help` lists all options.

## Evaluation on BFCL

The evaluation runs [StateAlign-8B](https://huggingface.co/ZhexiLu/StateAlign-8B) on the four BFCL v4 multi-turn
categories (base, missing function, missing parameter, long context) with BFCL's own Qwen3-8B function-calling
handler:

```bash
# evaluation environment
hf download ZhexiLu/StateAlign-8B --local-dir models/StateAlign-8B
RESULTS=results bash scripts/eval/bfcl.sh models/StateAlign-8B statealign_8b
```

The script starts one vLLM server on 4 GPUs (`NUM_GPUS`), runs `bfcl generate` and `bfcl evaluate`, and writes
`results/bfcl/statealign_8b/{result,score}`. The settings are those of the paper: temperature 0.001, and the model
served with YaRN factor 2, which gives a 65,536-token context. BFCL's scorer, data and inference loop are not
modified. The patch applied by `third_party/fetch.sh` adds one flag to the command that
starts the vLLM server (`--generation-config vllm`), so that the sampling parameters BFCL does not set are vLLM's
defaults.

The model directory is a full Qwen3-8B checkpoint. A LoRA checkpoint of your own training run is first merged into
the base model:

```bash
# training environment
bash scripts/eval/merge_lora.sh runs/<run name>/checkpoint-400 models/Qwen3-8B merged/<run name>
```

## Results

BFCL v4 multi-turn accuracy (%).

| Model | Overall | Base | Miss Func | Miss Param | Long Context |
|---|---|---|---|---|---|
| Qwen3-8B | 45.5 | 50.3 | 49.1 | 37.9 | 44.6 |
| StateAlign-8B | **55.2** | **61.4** | **56.1** | **48.8** | **54.6** |

Each number is the mean of six evaluations on H100 GPUs, so a single evaluation will not match it exactly.

## Citation

```bibtex
@misc{lu2026statealign,
  title  = {StateAlign: Internalizing State-Aware Decision-Making for Reliable Multi-Turn Tool Agents via Reinforcement Learning},
  author = {Lu, Zhexi and Chaudhury, Subhajit and Pedapati, Tejaswini and Murugesan, Keerthiram and Yu, Lei},
  year   = {2026}
}
```

## License and acknowledgements

The code is released under the [Apache License 2.0](LICENSE). Third-party components and their licenses are listed
in [THIRD_PARTY.md](THIRD_PARTY.md).

Training builds on [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft) and
[vLLM](https://github.com/vllm-project/vllm). The evaluation uses the
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla). The example data are built from
[ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) and
[LoopTool-23k](https://huggingface.co/datasets/zhangkangning/LoopTool-23k).
