"""StateAlign trainer: GRPO on multi-step episodes over constructed task states, plus the state-contrast term.

Objective per optimizer step (Section 3.3):
    J = J_outcome + kappa * J_pair
J_outcome   GRPO (trl, loss "dapo") with group-normalised advantages of the outcome reward R (statealign.env); a k3 KL
            to the frozen initial policy with weight beta.
J_pair      for every sampled token y_t of a response to a task state h that has a paired state h~,
                s_t = clip(log pi_0(y_t | h, z) - log pi_0(y_t | h~, z~), -c, c)
            with pi_0 the frozen initial model (adapter disabled), (h, z) / (h~, z~) the scoring inputs of the two
            states (shared scoring instruction + the reference path of each state). The term is a per-token policy
            gradient with s_t as the token's score: -sum_t sg(s_t) log pi_theta(y_t) / (number of scored tokens).
            Scores cover reasoning and action tokens up to the execution step; the appended continuation step
            receives outcome supervision only.

Episodes are generated in trl's rollout hook on the colocated vLLM engine: step 1 from the policy prompt; after a
non-terminal step the environment turn (recorded tool result or simulated user reply) is appended and the model
generates again, up to `horizon` steps, all in one completion sequence. Environment tokens are attended to but
excluded from every loss term.

Group selection (per rank and step, G rollouts per state):
  dynamic sampling   groups whose rewards are all equal carry no outcome signal; they are replaced by groups of fresh
                     states for up to `dyn_rounds` generation rounds (fresh draws = `dyn_over` x the missing groups);
                     slots still missing are filled with no-signal groups, which enter no loss term but the pair term
  reserved groups    with succ_share > 0, round(n_groups * succ_share) slots are reserved for successful groups (all
                     rollouts succeed, equal reward) whose state has a paired counterpart. Their outcome advantage is
                     zero, but the pair term trains on them.
"""
import collections
import json
import os
import pickle
import time

import torch
from trl import GRPOTrainer
from trl.trainer.utils import entropy_from_logits, selective_log_softmax, split_tensor_dict

PROMPT_KEYS = ("prompt_ids", "prompt_mask")
KINDS = ("EXEC", "NEED_USER", "NEED_FETCH", "INFEASIBLE", "STOP")
FIRST_ACTIONS = ("exec", "ask", "fetch", "tell", "wrong_call", "none", "truncated", "repeat", "call")


def _pad_right(rows, width, value, dtype, device):
    out = torch.full((len(rows), width), value, dtype=dtype, device=device)
    for i, r in enumerate(rows):
        r = r[:width]
        if len(r):
            out[i, :len(r)] = torch.tensor(r, dtype=dtype, device=device)
    return out


def sort_by_length(batch):
    """Reorder every per-sequence entry of the generation batch by real (prompt + completion) length."""
    n = batch["prompt_mask"].shape[0]
    idx = torch.argsort(batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1))
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            out[k] = v[idx]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[i] for i in idx.tolist()]
        else:
            out[k] = v
    return out


def trim(batch):
    """Cut a micro-batch's padding: prompts to their longest real prompt (left-padded), completions and every
    per-token tensor aligned with them to the longest real completion (right-padded)."""
    pm, cm = batch["prompt_mask"], batch["completion_mask"]
    n, tp, tc = pm.shape[0], pm.shape[1], cm.shape[1]
    p_len, c_len = int(pm.sum(1).max().item()), int(cm.sum(1).max().item())
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == n:
            if k in PROMPT_KEYS and v.shape[1] == tp:
                v = v[:, tp - p_len:]
            elif k not in PROMPT_KEYS and v.shape[1] == tc:
                v = v[:, :c_len]
        out[k] = v
    return out


def _vllm_module():
    try:  # trl >= 1.0: the colocated engine lives in trl.generation.vllm_generation
        import trl.generation.vllm_generation as m
        return m
    except ImportError:
        import trl.trainer.grpo_trainer as m
        return m


class StateAlignTrainer(GRPOTrainer):
    def __init__(self, *args, rows=None, scoring_prompts=None, pair_prompts=None, pool=None, env_cls=None,
                 gamma=0.8, horizon=4, step_tokens=1536, total_tokens=6144, pair_kappa=0.1, pair_clip=1.0,
                 succ_share=0.2, dyn_rounds=3, dyn_over=2.0, seed=42, is_cap=3.0, kl_cap=10.0, stop_ids=None,
                 vllm_stop=None, vllm_max_seqs=256, vllm_batched_tokens=16384, vllm_sleep=True, vllm_no_cascade=True,
                 full_state="latest", sort_trim=True, **kwargs):
        # trl builds the colocated engine with small max_num_seqs / max_num_batched_tokens; multi-step episodes and
        # dynamic sampling need more. The engine kwargs are raised by wrapping trl's LLM constructor during __init__.
        gm = _vllm_module()
        orig_llm = getattr(gm, "LLM", None)
        if orig_llm is not None:
            def _llm(*a, **kw):
                if vllm_no_cascade:  # standard attention kernel instead of vLLM's cascade kernel
                    kw["disable_cascade_attn"] = True
                if vllm_sleep:  # engine memory from vLLM's CuMem pool, released between rollouts
                    kw["enable_sleep_mode"] = True
                kw["max_num_seqs"] = max(int(kw.get("max_num_seqs") or 0), vllm_max_seqs)
                kw["max_num_batched_tokens"] = max(int(kw.get("max_num_batched_tokens") or 0), vllm_batched_tokens)
                return orig_llm(*a, **kw)
            gm.LLM = _llm
        try:
            super().__init__(*args, **kwargs)
        finally:
            if orig_llm is not None:
                gm.LLM = orig_llm
        self.rows, self.scoring_prompts, self.pair_prompts = rows, scoring_prompts or {}, pair_prompts or {}
        self.pool = list(pool or [])
        self.env_cls = env_cls
        self.gamma, self.horizon, self.step_tokens, self.total_tokens = gamma, horizon, step_tokens, total_tokens
        self.pair_kappa, self.pair_clip, self.succ_share = float(pair_kappa), float(pair_clip), float(succ_share)
        self.dyn_rounds, self.dyn_over = dyn_rounds, dyn_over
        self.is_cap, self.kl_cap = is_cap, kl_cap
        self.stop_ids = set(stop_ids or {self.processing_class.eos_token_id})
        self.vllm_stop = list(vllm_stop) if vllm_stop else None
        self.full_state, self.sort_trim = full_state, sort_trim
        import random as _random
        self._rng = _random.Random(seed * 1000 + self.accelerator.process_index)
        self._live_rate = 0.3  # running share of generated groups with outcome signal (sizes the first round)
        self._grp_cats = collections.Counter()
        self._gen_stats, self._dyn_info, self._gen_time = [], (1, 0, 0, 0), 0.0
        # vLLM sleep (level 2) between rollouts: the training forward / backward gets the whole GPU
        self.vllm_sleep, self._asleep = vllm_sleep, False
        if hasattr(self, "vllm_generation"):  # trl >= 1.0 calls sync_weights() directly: wake the engine first
            orig_sync = self.vllm_generation.sync_weights

            def _sync_weights():
                self._wake()
                return orig_sync()

            self.vllm_generation.sync_weights = _sync_weights
        if vllm_sleep:
            self._sleep()
        self.rollout_func = lambda prompts, trainer: trainer._rollout(prompts)

    # ------------------------------------------------------------------ engine
    def _engine(self):
        return self.vllm_generation.llm if hasattr(self, "vllm_generation") else self.llm

    def _sleep(self):
        if self.vllm_sleep and not self._asleep:
            self._engine().sleep(level=2)  # discards weights and KV cache
            self._asleep = True

    def _wake(self):
        if self.vllm_sleep and self._asleep:
            torch.cuda.empty_cache()
            self._engine().wake_up()
            self._asleep = False

    def _move_model_to_vllm(self):  # trl < 1.0 weight push
        self._wake()
        return super()._move_model_to_vllm()

    def _sync_weights(self):
        if hasattr(self, "vllm_generation"):
            return self.vllm_generation.sync_weights()
        return self._move_model_to_vllm()

    # ------------------------------------------------------------------ rollouts and group selection
    def _rollout(self, prompts):
        if not self.vllm_sleep:
            return self._rollout_impl(prompts)
        if self._asleep:  # no weight push since the last sleep: level 2 dropped the weights
            self._sync_weights()
        try:
            return self._rollout_impl(prompts)
        finally:
            self._sleep()

    def _rollout_impl(self, prompts):
        """trl rollout hook: episodes for `prompts` (G consecutive copies per state) with dynamic sampling."""
        G = self.num_generations
        n_groups = len(prompts) // G if G else 0
        blocks_ok = G > 1 and n_groups * G == len(prompts) and all(
            len(set(prompts[g * G:(g + 1) * G])) == 1 for g in range(n_groups))
        t0 = time.time()
        if not (blocks_ok and self.pool):
            first = self._run_episodes(list(prompts))
            self._gen_stats = list(first["stats"])
            self._gen_time = time.time() - t0
            self._dyn_info = (1, n_groups, n_groups, sum(self._group_live(first, g, G) for g in range(n_groups)))
            return self._pack([(first, g, True) for g in range(n_groups)], G)
        # first round: trl's groups plus enough fresh groups to expect n_groups signal groups at the running rate
        rate = min(1.0, max(0.15, self._live_rate))
        extra = min(3 * n_groups, max(0, int(n_groups / rate + 0.999) - n_groups))
        fresh0 = [self._rng.choice(self.pool) for _ in range(extra)]
        first = self._run_episodes(list(prompts) + [p for p in fresh0 for _ in range(G)])
        self._gen_stats = list(first["stats"])

        def note(res, g):  # logging: category of every generated group
            self._grp_cats[((res["rows"][g * G] or {}).get("kind"), self._group_cat(res, g, G))] += 1

        if self.succ_share > 0:
            return self._select_with_reserved(first, n_groups, extra, G, t0, note)
        kept, spare, live0 = [], [], 0
        for g in range(n_groups + extra):  # trl's own groups first, then the fresh ones
            note(first, g)
            ok = self._group_live(first, g, G)
            live0 += ok
            if ok and len(kept) < n_groups:
                kept.append((first, g))
            elif g < n_groups or not ok:
                spare.append((first, g))
        self._live_rate = 0.8 * self._live_rate + 0.2 * (live0 / max(1, n_groups + extra))
        rounds, generated = 1, n_groups + extra
        while len(kept) < n_groups and rounds < self.dyn_rounds:
            need = n_groups - len(kept)
            m = max(need, int(need * self.dyn_over + 0.999))
            fresh = [self._rng.choice(self.pool) for _ in range(m)]
            res = self._run_episodes([p for p in fresh for _ in range(G)])
            self._gen_stats += res["stats"]
            for g in range(m):
                note(res, g)
                if self._group_live(res, g, G) and len(kept) < n_groups:
                    kept.append((res, g))
                else:
                    spare.append((res, g))
            rounds += 1
            generated += m
        n_live = len(kept)
        final = [(r, g, True) for r, g in kept] + [(r, g, False) for r, g in spare[:n_groups - n_live]]
        self._gen_time = time.time() - t0
        self._dyn_info = (rounds, generated, n_groups, n_live)
        return self._pack(final, G)

    def _select_with_reserved(self, first, n_groups, extra, G, t0, note):
        """Dynamic sampling with n_succ = round(n_groups * succ_share) slots for successful groups whose state has a
        paired counterpart; the other slots take outcome-signal groups."""
        n_succ = min(n_groups - 1, int(round(n_groups * self.succ_share)))
        n_live_target = n_groups - n_succ
        kept, succ, spare = [], [], []
        has_pair = lambda res, g: bool(self.pair_prompts.get(res["prompts"][g * G]))

        def sort_group(res, g):
            note(res, g)
            cat = self._group_cat(res, g, G)
            if cat == "live":
                (kept if len(kept) < n_live_target else spare).append((res, g))
                return True
            (succ if cat == "allsucc" and has_pair(res, g) else spare).append((res, g))
            return False

        live0 = sum(sort_group(first, g) for g in range(n_groups + extra))
        self._live_rate = 0.8 * self._live_rate + 0.2 * (live0 / max(1, n_groups + extra))
        rounds, generated = 1, n_groups + extra
        while len(kept) < n_live_target and rounds < self.dyn_rounds:
            need = n_live_target - len(kept)
            m = max(need, int(need * self.dyn_over + 0.999))
            fresh = [self._rng.choice(self.pool) for _ in range(m)]
            res = self._run_episodes([p for p in fresh for _ in range(G)])
            self._gen_stats += res["stats"]
            for g in range(m):
                sort_group(res, g)
            rounds += 1
            generated += m
        final = [(r, g, True) for r, g in kept] + [(r, g, False) for r, g in succ[:n_succ]]
        rest = [x for x in spare if self._group_cat(x[0], x[1], G) == "live"] + succ[n_succ:] + \
            [x for x in spare if self._group_cat(x[0], x[1], G) != "live"]
        n_live = len(kept)
        for r, g in rest:
            if len(final) >= n_groups:
                break
            is_live = self._group_cat(r, g, G) == "live"
            final.append((r, g, is_live))
            n_live += is_live
        self._metrics["train"]["sa/reserved_groups"].append(float(min(n_succ, len(succ))))
        self._gen_time = time.time() - t0
        self._dyn_info = (rounds, generated, n_groups, n_live)
        return self._pack(final, G)

    @staticmethod
    def _group_live(res, g, G):
        """Does group g carry outcome signal (>= 2 non-void episodes with different rewards)?"""
        live = [res["rewards"][i] for i in range(g * G, (g + 1) * G) if not res["voids"][i]]
        return len(live) >= 2 and max(live) - min(live) > 1e-6

    @staticmethod
    def _group_cat(res, g, G):
        """live (outcome signal) | allfail | allsucc (all > 0, equal) | void (< 2 non-void episodes)."""
        live = [res["rewards"][i] for i in range(g * G, (g + 1) * G) if not res["voids"][i]]
        if len(live) < 2:
            return "void"
        if max(live) - min(live) > 1e-6:
            return "live"
        return "allfail" if max(live) <= 1e-6 else "allsucc"

    def _pack(self, final, G):
        """Concatenate the chosen groups into the rollout output; ep_live marks outcome-signal groups."""
        cols = ("prompts", "prompt_ids", "comp", "lmask", "gmask", "lps", "rewards", "voids", "eps", "rows")
        out = {c: [] for c in cols}
        live = []
        for res, g, is_live in final:
            for c in cols:
                out[c] += res[c][g * G:(g + 1) * G]
            live += [1 if is_live else 0] * G
        eps = out["eps"]
        return {"prompt_ids": out["prompt_ids"], "completion_ids": out["comp"], "logprobs": out["lps"],
                "ep_reward": out["rewards"], "ep_loss_mask": out["lmask"], "ep_gate_mask": out["gmask"],
                "ep_logps": out["lps"], "ep_prompt": out["prompts"], "ep_live": live,
                "ep_kind": [b["kind"] for b in out["rows"]], "ep_first": [e.first_action() for e in eps],
                "ep_label": [e.label for e in eps], "ep_steps": [e.steps for e in eps],
                "ep_gates": [list(e.gates) for e in eps], "ep_actions": [list(e.actions) for e in eps],
                "ep_void": out["voids"]}

    def _sampling_params(self, budget):
        from vllm import SamplingParams
        sp = dict(n=1, temperature=self.temperature, top_p=self.top_p,
                  top_k=-1 if (self.top_k is None or self.top_k <= 0) else self.top_k,
                  min_p=0.0 if self.min_p is None else self.min_p,
                  repetition_penalty=self.repetition_penalty, max_tokens=max(1, budget), logprobs=0)
        if self.vllm_stop:
            sp["stop_token_ids"] = list(self.vllm_stop)
        return SamplingParams(**sp)

    def _run_episodes(self, prompts):
        """Multi-step episodes for a list of policy prompts on this rank's colocated engine. An episode's next step
        is submitted as soon as its previous step has been processed (no per-step barrier across episodes)."""
        from vllm.sampling_params import RequestOutputKind

        tok = self.processing_class
        n = len(prompts)
        rows = [self.rows[p] for p in prompts]
        prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        eps = [self.env_cls(b, max_steps=self.horizon) for b in rows]
        comp, lmask, gmask, lps = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
        llm = self._engine()
        eng = llm.llm_engine
        owner, ngen = {}, [0] * n

        def submit(i):
            sp = self._sampling_params(min(self.step_tokens, self.total_tokens - len(comp[i])))
            sp.output_kind = RequestOutputKind.FINAL_ONLY  # as LLM.generate sets it
            rid = str(next(llm.request_counter))
            owner[rid] = i
            ngen[i] += 1
            eng.add_request(rid, {"prompt_token_ids": prompt_ids[i] + comp[i]}, sp)

        if self.horizon > 0:
            for i in range(n):
                submit(i)
        while eng.has_unfinished_requests():
            for o in eng.step():
                if not getattr(o, "finished", False) or o.request_id not in owner:
                    continue
                i = owner.pop(o.request_id)
                oo = o.outputs[0]
                gen = list(oo.token_ids)
                lp = [next(iter(d.values())).logprob for d in oo.logprobs] if oo.logprobs else [0.0] * len(gen)
                gate = int(eps[i].contrast_gate())
                comp[i] += gen
                lmask[i] += [1] * len(gen)
                gmask[i] += [gate] * len(gen)
                lps[i] += lp
                if not gen or gen[-1] not in self.stop_ids:  # step cut by the token budget: failure
                    eps[i].steps += 1
                    eps[i].gates.append(int(eps[i].executable()))
                    eps[i].actions.append("truncated")
                    eps[i]._finish(False, "truncated")
                    continue
                terminal, env = eps[i].step(tok.decode(gen, skip_special_tokens=False))
                if terminal:
                    continue
                env_ids = tok(env, add_special_tokens=False)["input_ids"]
                if len(comp[i]) + len(env_ids) + 64 > self.total_tokens:
                    eps[i]._finish(False, "budget")
                    continue
                comp[i] += env_ids
                lmask[i] += [0] * len(env_ids)
                gmask[i] += [0] * len(env_ids)
                lps[i] += [0.0] * len(env_ids)
                if ngen[i] < self.horizon:
                    submit(i)
        if owner:
            raise RuntimeError(f"{len(owner)} rollout requests never finished")
        for e in eps:
            if not e.done:
                e._finish(False, "horizon")
        rewards = [e.reward(self.gamma) for e in eps]
        # void episodes (an unrecorded lookup the offline environment cannot answer): reward = mean of the non-void
        # episodes of the same state (zero advantage), all tokens out of every loss term
        voids = [bool(getattr(e, "void", False)) for e in eps]
        if any(voids):
            by_prompt = {}
            for i, p in enumerate(prompts):
                by_prompt.setdefault(p, []).append(i)
            for idx in by_prompt.values():
                live = [rewards[i] for i in idx if not voids[i]]
                fill = sum(live) / len(live) if live else 0.0
                for i in idx:
                    if voids[i]:
                        rewards[i] = fill
                        lmask[i] = [0] * len(lmask[i])
                        gmask[i] = [0] * len(gmask[i])
        # statistics of every generated episode (the policy on the data mixture, not the selected batch)
        stats = [(b["kind"], rewards[i] > 0 and not voids[i], rewards[i], eps[i].steps,
                  eps[i].label == "exec_premature", voids[i], eps[i].first_action()) for i, b in enumerate(rows)]
        return {"prompts": list(prompts), "prompt_ids": prompt_ids, "comp": comp, "lmask": lmask, "gmask": gmask,
                "lps": lps, "rewards": rewards, "voids": voids, "eps": eps, "rows": rows, "stats": stats}

    # ------------------------------------------------------------------ micro-batches
    def _prepare_inputs(self, generation_batch):
        """Sort the generation batch by length and trim each micro-batch to its own longest prompt / completion
        (the step gradient is a sum over completions, so the split does not change the objective)."""
        if not self.model.training:
            return super()._prepare_inputs(generation_batch)
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            t0 = time.time()
            batch = self._generate_and_score_completions(generation_batch)
            self._metrics["train"]["gen_time"].append(time.time() - t0)
            if self.sort_trim:
                batch = sort_by_length(batch)
                batches = [trim(b) for b in split_tensor_dict(batch, self.args.steps_per_generation)]
            else:
                from trl.trainer.utils import shuffle_sequence_dict
                batches = split_tensor_dict(shuffle_sequence_dict(batch), self.args.steps_per_generation)
            self._buffered_inputs = batches
        return self._buffered_inputs[self._step % self.args.steps_per_generation]

    # ------------------------------------------------------------------ masks, normalisers, metrics
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            return out
        dev = out["completion_ids"].device
        B, C = out["completion_ids"].shape
        cm = out["completion_mask"]
        lm = _pad_right([x["ep_loss_mask"] for x in inputs], C, 0, cm.dtype, dev)
        gm = _pad_right([x["ep_gate_mask"] for x in inputs], C, 0, cm.dtype, dev) * lm
        live = torch.tensor([float(x.get("ep_live", 1)) for x in inputs], dtype=cm.dtype, device=dev).unsqueeze(1)
        out["ep_loss_mask"], out["ep_gate_mask"], out["ep_rl_mask"] = lm, gm, lm * live
        out["ep_samp_logps"] = _pad_right([x["ep_logps"] for x in inputs], C, 0.0, torch.float32, dev)
        # outcome-term normaliser: model tokens of outcome-signal groups (whole step, all ranks);
        # pair-term normaliser: scored tokens of every selected group
        out["num_items_in_batch"] = self.accelerator.gather((cm * lm * live).sum()).sum()
        out["ep_scored_total"] = self.accelerator.gather((gm * cm).sum()).sum()
        if self.pair_kappa > 0:
            key = lambda x: x.get("ep_prompt") or x["prompt"]
            out["ep_scoring_prompt"] = [self.scoring_prompts.get(key(x)) or "" for x in inputs]
            out["ep_pair_prompt"] = [self.pair_prompts.get(key(x)) or "" for x in inputs]
        self._log_rollouts(inputs, out, lm)
        self._log_rollout_metrics(inputs, out, cm, lm, gm, dev)
        return out

    def _log_rollout_metrics(self, inputs, out, cm, lm, gm, dev):
        M = self._metrics["train"]
        mask = cm * lm
        M["sa/gen_time"].append(self._gen_time)
        M["sa/scored_token_frac"].append(((gm * cm).sum() / mask.sum().clamp(min=1)).item())
        M["sa/env_token_frac"].append((cm * (1 - lm)).sum().item() / max(1, cm.sum().item()))
        cols = ["n", "success", "reward", "steps", "premature", "void"] + [f"first_{a}" for a in FIRST_ACTIONS]
        stat = torch.zeros(len(KINDS), len(cols), dtype=torch.float64, device=dev)
        for kind, succ, rew, steps, prem, void, first in self._gen_stats:
            if kind in KINDS:
                vals = [1, succ, rew, steps, prem, void] + [first == a for a in FIRST_ACTIONS]
                stat[KINDS.index(kind)] += torch.tensor([float(v) for v in vals], dtype=torch.float64, device=dev)
        stat = self.accelerator.reduce(stat, reduction="sum")
        tot = stat[:, 0].sum().item()
        if tot > 0:
            M["sa/reward_all"].append(stat[:, 2].sum().item() / tot)
        for r, k in enumerate(KINDS):
            n = stat[r, 0].item()
            if n > 0:
                for c, name in enumerate(cols[1:], start=1):
                    M[f"sa/{k}/{name}"].append(stat[r, c].item() / n)
                M[f"sa/{k}/count"].append(n)
        rounds, generated, n_groups, n_live = self._dyn_info
        dy = self.accelerator.reduce(torch.tensor([rounds, generated, n_groups, n_live], dtype=torch.float64,
                                                  device=dev), reduction="sum")
        M["sa/dyn_rounds"].append(dy[0].item() / self.accelerator.num_processes)
        M["sa/dyn_generated_groups"].append(dy[1].item())
        M["sa/dyn_live_frac"].append(dy[3].item() / max(1.0, dy[2].item()))
        cats = ("live", "allfail", "allsucc", "void")
        gc = torch.zeros(len(KINDS), len(cats), dtype=torch.float64, device=dev)
        for (k, c), v in self._grp_cats.items():
            if k in KINDS and c in cats:
                gc[KINDS.index(k), cats.index(c)] += v
        self._grp_cats = collections.Counter()
        gc = self.accelerator.reduce(gc, reduction="sum")
        if gc.sum().item() > 0:
            for c, name in enumerate(cats):
                M[f"sa/grp_{name}"].append(gc[:, c].sum().item() / gc.sum().item())

    def _log_rollouts(self, inputs, out, lm):
        """Every selected episode: <output_dir>/rollouts/rank<r>.jsonl, one JSON line per episode."""
        d = os.path.join(self.args.output_dir, "rollouts")
        os.makedirs(d, exist_ok=True)
        tok = self.processing_class
        step, rank = self.state.global_step, self.accelerator.process_index
        adv = out["advantages"].detach().float().cpu().tolist()
        cm, lmc = out["completion_mask"].detach().cpu(), lm.detach().cpu()
        with open(os.path.join(d, f"rank{rank}.jsonl"), "a") as f:
            for i, x in enumerate(inputs):
                n = int(cm[i].sum().item())
                ids = out["completion_ids"][i, :n].detach().cpu().tolist()
                f.write(json.dumps({
                    "step": step, "rank": rank, "row": i, "bid": self.rows[x.get("ep_prompt") or x["prompt"]]["bid"],
                    "rl_live": x.get("ep_live", 1), "kind": x["ep_kind"], "label": x["ep_label"],
                    "first": x["ep_first"], "steps": x["ep_steps"], "reward": x["ep_reward"], "advantage": adv[i],
                    "actions": x.get("ep_actions"), "gates": x.get("ep_gates"),
                    "n_model_tokens": int((cm[i] * lmc[i]).sum().item()),
                    "n_env_tokens": int((cm[i] * (1 - lmc[i])).sum().item()),
                    "text": tok.decode(ids, skip_special_tokens=False)}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ loss
    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        lm = inputs["ep_loss_mask"].to(completion_mask.dtype)
        mask = completion_mask * lm
        rmask = completion_mask * inputs["ep_rl_mask"].to(completion_mask.dtype)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        C = completion_ids.size(1)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=C + 1, use_cache=False).logits
        logits = logits[:, :-1, :][:, -C:, :]
        lt = logits / self.temperature
        per_token_logps = selective_log_softmax(lt, completion_ids)
        with torch.no_grad():
            entropies = entropy_from_logits(lt)

        # ---- outcome term: clipped GRPO surrogate (dapo), importance-corrected for the vLLM sampler
        adv = inputs["advantages"]
        if adv.dim() == 1:
            adv = adv.unsqueeze(1)
        old = inputs.get("old_per_token_logps")
        old = per_token_logps.detach() if old is None else old
        coef_1 = torch.exp(per_token_logps - old)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)
        # one optimizer step per generation batch: this forward's log-probs are pi_old; per-token pi_old / pi_vLLM,
        # truncated at is_cap
        d = (per_token_logps.detach().float() - inputs["ep_samp_logps"]) * mask
        ratio = torch.exp(d).clamp(max=self.is_cap)
        per_token_loss = per_token_loss * ratio.to(per_token_loss.dtype)
        M = self._metrics["train"]
        M["sa/is_logratio_token"].append((d.sum() / mask.sum().clamp(min=1)).item())
        if self.beta != 0.0:
            ref = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref - per_token_logps) - (ref - per_token_logps) - 1
            per_token_kl = per_token_kl.clamp(max=self.kl_cap)  # k3 is unbounded for very unlikely sampled tokens
            per_token_loss = per_token_loss + self.beta * per_token_kl
        normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
        loss_outcome = (per_token_loss * rmask).sum() / normalizer.clamp(min=1)

        # ---- state-contrast term
        loss_pair = logits.sum() * 0.0
        gm = inputs["ep_gate_mask"].bool() & mask.bool()
        if self.pair_kappa > 0 and int(gm.sum().item()) > 0:
            pair = inputs["ep_pair_prompt"]
            has = gm.any(dim=1).tolist()
            sel_rows = [i for i in range(len(pair)) if pair[i] and has[i]]
            M["sa/pair_rows_frac"].append(len(sel_rows) / max(1, len(pair)))
            if sel_rows:
                tok = self.processing_class
                dev = completion_ids.device
                sel = torch.tensor(sel_rows, device=dev)
                unwrapped = self.accelerator.unwrap_model(model)

                def base_logps(prompts):  # log pi_0(y_t | scoring input), adapter disabled
                    enc = tok(prompts, add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt")
                    t_ids, t_mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                    with torch.no_grad(), unwrapped.disable_adapter():
                        tl = unwrapped(input_ids=torch.cat([t_ids, completion_ids[sel]], dim=1),
                                       attention_mask=torch.cat([t_mask, completion_mask[sel]], dim=1),
                                       logits_to_keep=C + 1, use_cache=False).logits
                        tl = tl[:, :-1, :][:, -C:, :]
                        return selective_log_softmax(tl.float(), completion_ids[sel])

                lp_a = base_logps([inputs["ep_scoring_prompt"][i] for i in sel_rows])
                lp_b = base_logps([pair[i] for i in sel_rows])
                g = gm[sel].to(per_token_logps.dtype)
                s = (lp_a - lp_b).clamp(-self.pair_clip, self.pair_clip).to(per_token_logps.dtype) * g
                norm_pair = (inputs["ep_scored_total"] / self.accelerator.num_processes).clamp(min=1)
                loss_pair = -(s.detach() * per_token_logps[sel] * g).sum() / norm_pair
                ng = g.sum().clamp(min=1)
                M["sa/pair_score_mean"].append(((s * g).sum() / ng).item())
                M["sa/pair_score_absmean"].append(((s.abs() * g).sum() / ng).item())
                M["sa/pair_score_pos_frac"].append((((s > 0).float() * g).sum() / ng).item())
                del lp_a, lp_b

        loss = loss_outcome + self.pair_kappa * loss_pair
        cnt = mask.sum().clamp(min=1)
        M["sa/loss_outcome"].append(loss_outcome.item())
        if self.pair_kappa > 0:
            M["sa/loss_pair"].append(loss_pair.item())
        M["entropy"].append(self.accelerator.gather((entropies * mask).sum() / cnt).nanmean().item())
        if self.beta != 0.0:
            M["kl"].append(self.accelerator.gather((per_token_kl * mask).sum() / cnt).nanmean().item())
        low = ((coef_1 < 1 - self.epsilon_low) & (adv < 0)).float()
        high = ((coef_1 > 1 + self.epsilon_high) & (adv > 0)).float()
        M["clip_ratio/region_mean"].append(self.accelerator.gather(((low + high) * mask).sum() / cnt).nanmean().item())
        return loss

    # ------------------------------------------------------------------ checkpoints and restart
    def sampler_state(self):
        return {"rng": self._rng.getstate(), "live_rate": self._live_rate, "global_step": self.state.global_step}

    def load_sampler_state(self, ckpt):
        """Restore this rank's fresh-draw RNG and running signal rate saved with the checkpoint."""
        with open(os.path.join(ckpt, f"sampler_state_rank{self.accelerator.process_index}.pkl"), "rb") as f:
            st = pickle.load(f)
        self._rng.setstate(st["rng"])
        self._live_rate = st["live_rate"]

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Reload the adapter onto this rank's device (peft's default loads it onto cuda:0 from every rank)."""
        m = self.model if model is None else model
        if getattr(m, "load_adapter", None) is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        orig, dev = m.load_adapter, str(self.accelerator.device)

        def load_adapter(*a, **k):
            k.setdefault("torch_device", dev)
            return orig(*a, **k)

        m.load_adapter = load_adapter
        try:
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        finally:
            del m.load_adapter

    FULL_STATE_FILES = ("optimizer.pt", "scheduler.pt", "scaler.pt", ".full_state_ok")

    def _save_checkpoint(self, model, trial):
        """full_state 'all' | 'latest' | 'off': besides HF's checkpoint every rank saves its sampler state, and rank 0
        marks a complete set with .full_state_ok; with 'latest' older checkpoints keep only the adapter."""
        super()._save_checkpoint(model, trial)
        if self.full_state == "off":
            return
        import glob

        run_dir = self._get_output_dir(trial=trial)
        ck = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")
        rank, world = self.accelerator.process_index, self.accelerator.num_processes
        tmp = os.path.join(ck, f".sampler_state_rank{rank}.pkl.partial")
        with open(tmp, "wb") as f:
            pickle.dump(self.sampler_state(), f)
        os.replace(tmp, os.path.join(ck, f"sampler_state_rank{rank}.pkl"))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            need = ["adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
            need += [f"rng_state_{r}.pth" if world > 1 else "rng_state.pth" for r in range(world)]
            need += [f"sampler_state_rank{r}.pkl" for r in range(world)]
            missing = [x for x in need if not os.path.exists(os.path.join(ck, x))]
            if missing:
                print(f"[statealign] checkpoint {ck} incomplete (missing {missing})", flush=True)
            else:
                with open(os.path.join(ck, ".full_state_ok"), "w") as f:
                    f.write(f"step {self.state.global_step} world {world}\n")
                if self.full_state == "latest":
                    for old in glob.glob(os.path.join(run_dir, "checkpoint-*")):
                        if os.path.abspath(old) == os.path.abspath(ck):
                            continue
                        victims = [os.path.join(old, x) for x in self.FULL_STATE_FILES]
                        victims += glob.glob(os.path.join(old, "rng_state*.pth"))
                        victims += glob.glob(os.path.join(old, "sampler_state_rank*.pkl"))
                        for v in victims:
                            if os.path.exists(v):
                                os.remove(v)
        self.accelerator.wait_for_everyone()
