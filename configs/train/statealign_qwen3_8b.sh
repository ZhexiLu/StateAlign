# StateAlign on Qwen3-8B, the setting of the paper: 400 steps of 320 episodes (8 GPUs x 2 x 20) on eight 80 GB GPUs.
# The run needs the full training file, which is not part of this release (set DATA to its path once you have it);
# configs/train/example_qwen3_8b.sh runs the same objective on the 100-state example.
NAME=statealign_qwen3_8b
DATA=data/statealign_train.jsonl
KINDS="EXEC,NEED_USER,NEED_FETCH,INFEASIBLE,STOP"
# row repetition per task-state type (resampling of the training distribution)
KIND_WEIGHTS="EXEC=0.32,NEED_USER=0.3200,NEED_FETCH=1.0108,INFEASIBLE=0.7095,STOP=0.15"
HORIZON=4            # model steps per episode, including the step after the execution
GAMMA=0.8            # discount per step before the execution
STEP_TOKENS=1536     # new tokens per model step
TOTAL_TOKENS=6144    # completion tokens per episode
PAIR_KAPPA=0.1       # weight of the state-contrast term
SUCC_SHARE=0.2       # share of groups reserved for successful groups
GA=20
MAX_STEPS=400
SAVE_STEPS=25
