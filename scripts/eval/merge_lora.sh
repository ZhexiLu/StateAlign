#!/bin/bash
# Merge a training checkpoint into its base model for serving (CPU).
#   bash scripts/eval/merge_lora.sh <run>/checkpoint-400 <base model dir> <out dir>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=
python -m statealign.evaluation.merge_lora --adapter "${1:?adapter}" --base "${2:?base}" --out "${3:?out}"
