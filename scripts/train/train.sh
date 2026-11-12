#!/bin/bash
# Train one StateAlign run on a single multi-GPU node.
#   MODEL=<Qwen3 model dir> bash scripts/train/train.sh configs/train/example_qwen3_8b.sh [more statealign.training.train flags]
# Environment:
#   MODEL   base model directory (a local copy of, e.g., Qwen/Qwen3-8B)                  required
#   DATA    training file; overrides the one named in the config (relative to the repository root or absolute)
#   RUNS    output root (default: <repository>/runs); the run is written to $RUNS/<NAME>
#   NPROC   number of GPUs = training processes (default 8);  PDBS  per-device micro-batch (default 2)
# Episodes per optimizer step = NPROC * PDBS * GA (8 * 2 * 20 = 320 in the paper setting); PDBS * GA must be a multiple
# of the 8 rollouts per task state. Checkpoints are written every SAVE_STEPS steps; a restarted run continues from its
# newest complete checkpoint.
set -euo pipefail
CFG=${1:?config file}; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA_OVERRIDE=${DATA:-}
source "$CFG"
: "${MODEL:?set MODEL (base model directory)}"
DATA=${DATA_OVERRIDE:-$DATA}
case "$DATA" in /*) ;; *) DATA=$ROOT/$DATA ;; esac
[ -f "$DATA" ] || { echo "training file not found: $DATA"; exit 1; }
RUNS=${RUNS:-$ROOT/runs}
NPROC=${NPROC:-8}; PDBS=${PDBS:-2}; PORT=${PORT:-29500}
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=   # vLLM's sleep-mode allocator rejects expandable segments
accelerate launch --config_file "$ROOT/configs/accelerate_ddp.yaml" --num_processes "$NPROC" --num_machines 1 \
  --main_process_port "$PORT" -m statealign.training.train \
  --data "$DATA" --model "$MODEL" --out "$RUNS/$NAME" --run_name "$NAME" \
  --kinds "$KINDS" --kind_weights "$KIND_WEIGHTS" --horizon "$HORIZON" --gamma "$GAMMA" \
  --step_tokens "$STEP_TOKENS" --total_tokens "$TOTAL_TOKENS" --pair_kappa "$PAIR_KAPPA" --succ_share "$SUCC_SHARE" \
  --pdbs "$PDBS" --ga "$GA" --max_steps "$MAX_STEPS" --save_steps "${SAVE_STEPS:-25}" \
  --vllm_gmu "${VLLM_GMU:-0.45}" --attn "${ATTN:-flash_attention_2}" "$@"
