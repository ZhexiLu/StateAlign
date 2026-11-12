#!/bin/bash
# BFCL v4 multi-turn (base, miss_func, miss_param, long_context) for StateAlign-8B, one evaluation.
#   RESULTS=<output dir> bash scripts/eval/bfcl.sh <model dir> <tag>
# The model is a merged Qwen3-8B checkpoint (scripts/eval/merge_lora.sh). It is evaluated with BFCL's own Qwen3-8B
# function-calling handler at temperature 0.001 and served with YaRN factor 2 (65,536-token context).
# Runs in the evaluation environment (env/requirements-eval.txt) with the BFCL checkout of third_party/fetch.sh.
# BFCL starts one vLLM server on NUM_GPUS GPUs (default 4). Outputs: $RESULTS/bfcl/<tag>/{result,score}.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=$(readlink -f "${1:?model dir}"); TAG=${2:?tag}
HANDLER=Qwen/Qwen3-8B-FC
: "${RESULTS:?set RESULTS (evaluation outputs)}"
mkdir -p "$RESULTS/bfcl/$TAG"; RUN=$(cd "$RESULTS/bfcl/$TAG" && pwd)
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
MODEL=$(python -m statealign.evaluation.serving_view --model "$MODEL" --out "$RUN/model_view" --yarn 2)
# BFCL reads the server port from LOCAL_SERVER_PORT; a free one is picked unless it is set
export BFCL_PROJECT_ROOT=$RUN
export LOCAL_SERVER_PORT=${LOCAL_SERVER_PORT:-$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])')}
cd "$RUN"
bfcl generate --model "$HANDLER" --local-model-path "$MODEL" --backend vllm --temperature 0.001 \
  --test-category multi_turn --num-gpus "${NUM_GPUS:-4}" --gpu-memory-utilization "${GMU:-0.75}" --result-dir "$RUN/result"
bfcl evaluate --model "$HANDLER" --test-category multi_turn --result-dir "$RUN/result" --score-dir "$RUN/score"
cat "$RUN/score/data_multi_turn.csv"; echo
