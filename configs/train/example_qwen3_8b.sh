# A short run on the 100-state example (data/statealign_example_100.jsonl) that exercises the whole training path.
# Objective and optimisation settings are those of the paper run; the batch is smaller (NPROC x 2 x 4 episodes per
# step) and every task-state type is used once. It checks the installation and does not reproduce the paper's model.
NAME=example_qwen3_8b
DATA=data/statealign_example_100.jsonl
KINDS="EXEC,NEED_USER,NEED_FETCH,INFEASIBLE,STOP"
KIND_WEIGHTS="EXEC=1,NEED_USER=1,NEED_FETCH=1,INFEASIBLE=1,STOP=1"
HORIZON=4
GAMMA=0.8
STEP_TOKENS=1536
TOTAL_TOKENS=6144
PAIR_KAPPA=0.1
SUCC_SHARE=0.2
GA=4
MAX_STEPS=20
SAVE_STEPS=10
