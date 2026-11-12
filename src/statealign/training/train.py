"""Train a LoRA adapter with StateAlign.

    accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes 8 \
        -m statealign.training.train --data <rows.jsonl> --model <base model dir> --out <run dir> [options]

scripts/train/train.sh wraps this command with a settings file from configs/train/.

Each data row is one task state (data/README.md). The policy prompt is rendered with the Qwen3 chat template; rows with
a paired state ("twin_*" fields) also get the two scoring inputs of the state-contrast term. Rows are repeated per
kind by --kind_weights (fractional weights are rounded stochastically with --seed) and shuffled.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys
import time

from statealign.chat import Qwen3Format
from statealign.env import Episode


def latest_full_state(out):
    """Newest checkpoint of a run directory that carries a complete restart state, or None."""
    cks = [c for c in glob.glob(os.path.join(out, "checkpoint-*")) if os.path.exists(os.path.join(c, ".full_state_ok"))]
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[1])) if cks else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training rows (jsonl), see data/README.md")
    ap.add_argument("--model", required=True, help="base model directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run_name", default="")
    # task states and reward
    ap.add_argument("--kinds", default="EXEC,NEED_USER,NEED_FETCH,INFEASIBLE,STOP")
    ap.add_argument("--kind_weights", default="EXEC=0.32,NEED_USER=0.3200,NEED_FETCH=1.0108,INFEASIBLE=0.7095,STOP=0.15",
                    help="row repetition per kind (resampling by condition)")
    ap.add_argument("--gamma", type=float, default=0.8, help="discount per step before the execution")
    ap.add_argument("--horizon", type=int, default=4, help="max model steps per episode (incl. the continuation step)")
    ap.add_argument("--step_tokens", type=int, default=1536, help="max new tokens per model step")
    ap.add_argument("--total_tokens", type=int, default=6144, help="max completion tokens per episode")
    # state-contrast term and group selection
    ap.add_argument("--pair_kappa", type=float, default=0.1, help="weight of the state-contrast term (0 = off)")
    ap.add_argument("--pair_clip", type=float, default=1.0, help="per-token score clip")
    ap.add_argument("--succ_share", type=float, default=0.2, help="share of groups reserved for successful groups")
    ap.add_argument("--dyn_rounds", type=int, default=3, help="max generation rounds of dynamic sampling per step")
    ap.add_argument("--dyn_over", type=float, default=2.0, help="fresh groups drawn per missing group")
    # optimisation
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=15, help="linear warm-up, then constant")
    ap.add_argument("--beta", type=float, default=0.04, help="KL weight to the initial policy")
    ap.add_argument("--num_generations", type=int, default=8, help="rollouts per task state (G)")
    ap.add_argument("--pdbs", type=int, default=2, help="per-device micro-batch (episodes)")
    ap.add_argument("--ga", type=int, default=20, help="gradient accumulation; episodes per step = procs * pdbs * ga")
    ap.add_argument("--max_grad_norm", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    # model and engine
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--vllm_gmu", type=float, default=0.45, help="GPU memory share of the colocated vLLM engine")
    ap.add_argument("--vllm_max_model_len", type=int, default=0, help="0 = the model's own length")
    ap.add_argument("--vllm_max_seqs", type=int, default=256)
    ap.add_argument("--vllm_batched_tokens", type=int, default=16384)
    ap.add_argument("--vllm_sleep", action=argparse.BooleanOptionalAction, default=True,
                    help="release the engine's weights and KV cache between rollouts")
    ap.add_argument("--vllm_no_cascade", action=argparse.BooleanOptionalAction, default=True,
                    help="standard attention kernel instead of vLLM's cascade kernel")
    # run management
    ap.add_argument("--full_state", default="latest", choices=["off", "all", "latest"])
    ap.add_argument("--resume", default="auto", help="auto = newest complete checkpoint in --out; never; or a path")
    ap.add_argument("--nan_guard", action=argparse.BooleanOptionalAction, default=True,
                    help="stop before an optimizer step with a non-finite gradient")
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many rows")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig

    from statealign.training.trainer import StateAlignTrainer

    rank = int(os.environ.get("RANK", "0"))
    tok = AutoTokenizer.from_pretrained(a.model)
    fmt = Qwen3Format(tok)
    use_pair = a.pair_kappa > 0
    keep = set(a.kinds.split(","))
    w = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.kind_weights.split(",") if kv}
    rng = random.Random(a.seed)
    t0 = time.time()
    by_prompt, scoring, pair, rows, per_kind = {}, {}, {}, [], collections.Counter()
    for line in open(a.data):
        if not line.strip():
            continue
        b = json.loads(line)
        if b["kind"] not in keep:
            continue
        prompt = fmt.policy_prompt(b["tools"], b["messages"])
        by_prompt[prompt] = b
        if use_pair and b.get("twin_messages"):
            scoring[prompt] = fmt.scoring_prompt(b["tools"], b["messages"], b["gold_response"])
            pair[prompt] = fmt.scoring_prompt(b["twin_tools"], b["twin_messages"], b["twin_ref"])
        k = w.get(b["kind"], 1.0)
        n_rep = int(k) + (1 if rng.random() < (k - int(k)) else 0)
        rows.extend([{"prompt": prompt, "bid": b["bid"], "kind": b["kind"]}] * n_rep)
        per_kind[b["kind"]] += n_rep
        if a.limit and len(by_prompt) >= a.limit:
            break
    ds = Dataset.from_list(rows).shuffle(seed=a.seed)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        print(f"[statealign] {len(by_prompt)} task states -> {len(rows)} rows {dict(per_kind)}, {len(pair)} with a "
              f"paired state, rendered in {time.time() - t0:.0f}s", flush=True)
        print(f"[statealign] gamma={a.gamma} horizon={a.horizon} pair_kappa={a.pair_kappa} "
              f"pair_clip={a.pair_clip} succ_share={a.succ_share} G={a.num_generations} "
              f"episodes/step={a.pdbs * a.ga * world} lr={a.lr} beta={a.beta} model={a.model}", flush=True)

    def outcome_reward(prompts, completions, ep_reward=None, **kw):
        return [float(r) for r in ep_reward]

    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = GRPOConfig(
        output_dir=a.out, run_name=a.run_name or os.path.basename(a.out.rstrip("/")),
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=a.warmup_steps,
        max_steps=a.max_steps, per_device_train_batch_size=a.pdbs, gradient_accumulation_steps=a.ga,
        max_grad_norm=a.max_grad_norm, num_generations=a.num_generations, max_completion_length=a.total_tokens,
        temperature=a.temperature, top_p=1.0, beta=a.beta, loss_type="dapo", scale_rewards="group",
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_gmu, vllm_tensor_parallel_size=1,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=a.save_steps, save_total_limit=None,
        report_to=["wandb"] if a.wandb_project else [], seed=a.seed, remove_unused_columns=False,
        log_completions=False, mask_truncated_completions=False, vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_correction=False,  # computed inside the training forward (trainer._compute_loss)
        disable_dropout=True,
        model_init_kwargs={"device_map": None, "dtype": torch.bfloat16, "attn_implementation": a.attn},
        **({"vllm_max_model_length": a.vllm_max_model_len} if a.vllm_max_model_len else {}),
    )
    if a.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", a.wandb_project)
    trainer = StateAlignTrainer(
        model=a.model, reward_funcs=[outcome_reward], args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora,
        rows=by_prompt, scoring_prompts=scoring, pair_prompts=pair, pool=[r["prompt"] for r in rows], env_cls=Episode,
        gamma=a.gamma, horizon=a.horizon, step_tokens=a.step_tokens, total_tokens=a.total_tokens,
        pair_kappa=a.pair_kappa, pair_clip=a.pair_clip, succ_share=a.succ_share, dyn_rounds=a.dyn_rounds,
        dyn_over=a.dyn_over, seed=a.seed, stop_ids=fmt.stop_ids(),
        vllm_max_seqs=a.vllm_max_seqs, vllm_batched_tokens=a.vllm_batched_tokens, vllm_sleep=a.vllm_sleep,
        vllm_no_cascade=a.vllm_no_cascade, full_state=a.full_state, sort_trim=True)
    if a.nan_guard:
        class NanGuard(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
                g = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if g and not bool(torch.stack([torch.isfinite(x).all() for x in g]).all()):
                    raise RuntimeError(f"non-finite gradient before optimizer step {state.global_step + 1}")

        trainer.add_callback(NanGuard())
    resume = None if a.resume == "never" else (latest_full_state(a.out) if a.resume == "auto" else a.resume)
    if resume:
        if rank == 0:
            print(f"[statealign] resuming from {resume}", flush=True)
        trainer.load_sampler_state(resume)
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()
    trainer.save_model(os.path.join(a.out, "final"))
    if rank == 0:
        print("[statealign] done", flush=True)
    if a.vllm_sleep:
        # an engine left asleep must be mapped back before teardown, and vLLM's CuMem allocator aborts in the
        # interpreter's teardown: close the logger, flush and exit without it
        if trainer._asleep:
            torch.cuda.empty_cache()
            trainer._engine().wake_up()
            trainer._asleep = False
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        os._exit(0)


if __name__ == "__main__":
    main()
