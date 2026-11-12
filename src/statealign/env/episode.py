"""Episodes on constructed task states: the interaction and the outcome reward used in training (Section 3).

An instance (one row of the training data) is a task state: tools, an interaction history, and what the next action
must achieve. Its `kind` is one of

    EXEC        every required value is available: execute the target call(s)
    NEED_USER   a required value is missing and only the user can supply it: ask, then execute
    NEED_FETCH  a required value is missing and a listed read-only tool can return it: retrieve it, then execute
    INFEASIBLE  no listed tool can fulfil the request: decline without calling a substitute
    STOP        the tool results already answer the request: reply in text, without another call

Episode(row, max_steps).step(text) is called with every raw model step and returns (terminal, env_text): env_text is
the chat-template text that follows the step (the recorded tool result or a simulated user reply, then the next
generation prompt) or "" when the episode has ended. After the episode, reward(gamma) is the outcome reward.

Interaction feedback comes from the source trajectory only:
  * a call that matches a target call on its checked arguments is an execution attempt (terminal);
  * a call that re-issues a recorded call returns the recorded observation;
  * a well-formed read-only call that was never recorded cannot be answered offline: the episode is void (no reward
    signal, excluded from the loss);
  * a question gets a simulated reply: NEED_USER states supply the missing values the question names (any question
    when a single value is pending), other states answer with a fixed sentence;
  * any other call, or a text answer where an action is required, ends the episode as a failure.
An execution attempt succeeds when every target call is matched, no wrong call is in the same step, and the state was
executable at that step (the missing values were obtained earlier in the episode).

Continuation (rows with a "cont" field): after a successful execution the recorded results of the target calls are
returned and the model takes one more step, judged by the next recorded step of the source trajectory (a text reply,
or the next execution). The episode succeeds only if this step succeeds. Calls beyond the targets in the executing
step scale the reward by EXTRA_CALL_SCALE.

Outcome reward: R = gamma^(k-1) * success * alpha, k = the step of the execution (the continuation step is not
discounted), alpha = EXTRA_CALL_SCALE if the executing step carried extra calls, else 1.
"""
import json
import re

from .text import (_ACTIVE, READ_VERBS, _fmt, _tool_names, canon, canon_scalar, is_placeholder, leaves_of, norm_key,
                   norm_name, parse_response, same_value, sentences, sep_norm, top_arg, words_of_name, ASK_REQUEST, OFFER)

UNINFORMATIVE = '{"error": "No data found for this request."}'
REPLY_ALREADY = "Everything you need is in my earlier messages. Use your judgment for anything I didn't specify."
REPLY_UNKNOWN_FETCH = "I don't know that."
REPLY_UNCLEAR = "I'm not sure what else you need from me."
REPLY_INFEASIBLE = "I don't have anything else to add."
TOOL_TURN = "\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
USER_TURN = "\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"
EXTRA_CALL_SCALE = 0.5


def close_think(text):
    """A well-formed <tool_call> after an unclosed <think> is parsed as a call (BFCL's Qwen handler extracts it)."""
    if "</think>" not in (text or "") and "<tool_call>" in (text or ""):
        text = text.replace("<think>", "<think>\n</think>", 1) if "<think>" in text else "</think>" + text
    return text


def _load(v):
    return json.loads(v) if isinstance(v, str) else v


class InfoEpisode:
    """EXEC / NEED_USER / NEED_FETCH / INFEASIBLE states (one or more steps up to max_steps)."""

    def __init__(self, b, max_steps=3):
        self.b, self.kind, self.max_steps = b, b["kind"], max_steps
        tools = _load(b["tools"])
        self.tools = _tool_names(tools)
        self.lookup = set(b.get("lookup_fns") or [])
        self.targets = [(t["name"], canon(t["arguments"], set(t["check"]))) for t in b["target"]]
        self.target_names = {n for n, _ in self.targets}
        self.replays = [(r["name"], canon(r["arguments"]), r["observation"]) for r in b.get("replay") or []]
        self.supplied, self.replayed = set(), set()
        self.steps, self.done, self.success = 0, False, 0
        self.gates, self.actions, self.label = [], [], None
        self.void = False
        self.equiv = [set(t.get("equiv") or []) - {t["name"]} for t in b["target"]]
        self.specs = {}
        for t in tools:
            fn = (t.get("function") or t) if isinstance(t, dict) else {}
            if fn.get("name"):
                self.specs[fn["name"]] = fn
        # pagination: the target function is itself the required lookup, so a reply after fetching counts as success
        self.fetch_pagination = self.kind == "NEED_FETCH" and any(
            b["replay"][j]["name"] in self.target_names for j in b.get("required") or [])
        self.fetched_required = False
        msgs = _load(b["messages"])
        self.context_text = " ".join(str(m.get("content") or "") for m in msgs) + " " + " ".join(
            str(s["value"]) for s in b.get("supply") or [])
        self.span_ctx = sep_norm(" ".join(str(m.get("content") or "") for m in msgs)).lower()
        self.enum_args = set()
        for fn in self.specs.values():
            props = (fn.get("parameters") or {}).get("properties") if isinstance(fn.get("parameters"), dict) else None
            for k, pv in (props or {}).items():
                if isinstance(pv, dict) and pv.get("enum"):
                    self.enum_args.add(norm_key(k))
        self.defaults = {}
        for name, fn in self.specs.items():
            props = (fn.get("parameters") or {}).get("properties") if isinstance(fn.get("parameters"), dict) else None
            props = props if isinstance(props, dict) else (fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {})
            self.defaults[name] = {k: canon_scalar(p["default"]) for k, p in props.items()
                                   if isinstance(p, dict) and "default" in p and not isinstance(p["default"], (dict, list))}
            self.defaults[name]["__params__"] = set(props.keys())

    # ---- information state
    def executable(self):
        if self.kind == "INFEASIBLE":
            return False
        if self.kind == "NEED_USER":
            return len(self.supplied) == len(self.b["supply"])
        if self.kind == "NEED_FETCH":
            return set(self.b["required"]) <= self.replayed
        return True

    def contrast_gate(self):
        """1 on the steps whose tokens receive the state-contrast term."""
        return 1

    # ---- call matching
    def readonly(self, name):
        """A function is read-only if it is a recorded lookup or its name / description starts with a lookup verb."""
        if name in self.lookup:
            return True
        fn = self.specs.get(name) or {}
        w = words_of_name(fn.get("description") or "")
        n = words_of_name(name)
        return bool((w and w[0] in READ_VERBS) or (n and n[0] in READ_VERBS))

    def _canon_keys(self, args, target_keys):
        """Canonical arguments, top-level keys renamed to the target's spelling when the model used the schema's
        spelling and the recorded call one the schema lacks (case, separators, a leading 'is')."""
        ca = canon(args if isinstance(args, dict) else {}, None)
        params = self.defaults.get(self._cur_fn, {}).get("__params__", set())
        tk = {norm_key(top_arg(k)): top_arg(k) for k in target_keys if top_arg(k) not in params}
        out = {}
        for path, v in ca.items():
            top = top_arg(path)
            nt = tk.get(norm_key(top)) if top in params else None
            out[(nt + path[len(top):]) if nt and nt != top else path] = v
        return out

    def _match_target(self, name, args, used=()):
        self._cur_fn = name
        for i, (n, t) in enumerate(self.targets):
            if i in used:  # two identical parallel target calls: each generated call matches only one of them
                continue
            if n == name:
                ca = self._canon_keys(args, t.keys())
                if all(k in ca and same_value(ca[k], v, top_arg(k)) for k, v in t.items()):
                    return i
        for i, (n, t) in enumerate(self.targets):  # functions labelled equivalent to the target during construction
            if i in used:
                continue
            if name in self.equiv[i] and t:
                vals = list(canon(args if isinstance(args, dict) else {}, None).values())
                if all(any(same_value(mv, tv, top_arg(k)) for mv in vals) for k, tv in t.items()):
                    return i
        return self._match_alt(name, args)

    def _match_alt(self, name, args):
        """Another read-only function with the same normalised name whose argument values cover the checked values."""
        if name in self.target_names or not self.readonly(name):
            return None
        vals = list(canon(args if isinstance(args, dict) else {}, None).values())
        for i, (n, t) in enumerate(self.targets):
            if self.specs.get(n) is None and self.kind != "INFEASIBLE":
                continue
            if not self.readonly(n) and n not in self.lookup:
                continue
            if norm_name(name) != norm_name(n) or not t:
                continue
            if all(any(same_value(mv, tv, top_arg(k)) for mv in vals) for k, tv in t.items()):
                return i
        return None

    def _match_replay(self, name, args):
        """Index of the recorded call this call re-issues, or None. Arguments equal to the schema default are ignored;
        when the row lists key_args, only those (the values visible in the history) must match."""
        dflt = self.defaults.get(name, {})
        strip = lambda d: {k: v for k, v in d.items() if dflt.get(k) != v}
        ca = strip(canon(args if isinstance(args, dict) else {}, None))
        for j, (n, r, _) in enumerate(self.replays):
            if n != name:
                continue
            key = (self.b.get("replay") or [{}] * (j + 1))[j].get("key_args")
            if key:
                full = canon(args if isinstance(args, dict) else {}, None)
                if all(any(top_arg(k) == a for k in full) for a in key) and all(
                        k in full and same_value(full[k], v, top_arg(k)) for k, v in r.items() if top_arg(k) in key):
                    return j
                continue
            r = strip(r)
            if not r and ca:
                continue
            keys = set(r) | set(ca)
            if all(k in ca and k in r and same_value(ca[k], r[k], top_arg(k)) for k in keys):
                return j
        return None

    def _finish(self, success, label):
        self.done, self.success, self.label = True, int(bool(success)), label
        return True, ""

    @staticmethod
    def _offer_only(text):
        """True if every question in the text is an offer or a menu item after an offer lead-in."""
        if not OFFER.search(text or ""):
            return False
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        for s in sentences(text):
            if ASK_REQUEST.search(s):
                return False
        for l in lines:
            if l.rstrip("*_ )\"'`").endswith("?") and not OFFER.search(l) and not re.match(r"^(\d+[.)]|[-*•])\s", l):
                return False
        return True

    # ---- one model step
    def step(self, text):
        _ACTIVE["ctx"], _ACTIVE["enum_args"] = self.span_ctx, self.enum_args
        text = close_think(text)
        self.steps += 1
        self.gates.append(int(self.executable()))
        p = parse_response(text)
        kind = p["kind"]
        if kind == "ask" and self._offer_only(p["text"]):
            kind = "tell"
        if kind == "none":
            self.actions.append("none")
            return self._finish(False, "none:" + str(p.get("reason")))
        if kind == "call":
            return self._step_calls(p["calls"])
        if kind == "ask":
            return self._step_ask(p["text"])
        self.actions.append("tell")
        if self.fetch_pagination and self.fetched_required:
            return self._finish(True, "tell_after_fetch")
        return self._finish(self.kind == "INFEASIBLE", "tell")

    def _step_calls(self, calls):
        # pass 1: classify every call
        cls, used = [], set()
        for c in calls:
            name, args = c["function"]["name"], c["function"]["arguments"]
            if any(is_placeholder(v) and str(v) not in self.context_text
                   for _, v in leaves_of(args if isinstance(args, dict) else {})):
                cls.append(("placeholder", name, None))
                continue
            if name not in self.tools:
                cls.append(("unknown_fn", name, None))
                continue
            ti = self._match_target(name, args, used)
            if ti is None:
                ti = self._match_target(name, args)  # matches only an already-used target -> duplicate call
            if ti is not None and ti not in used:
                used.add(ti)
                cls.append(("target", name, ti))
                continue
            if ti is not None:
                cls.append(("dup_target", name, ti))
                continue
            rj = self._match_replay(name, args)
            if rj is not None:
                cls.append(("replay", name, rj))
                continue
            if name in self.target_names:
                cls.append(("target_wrong_value", name, None))
                continue
            cls.append(("lookup" if self.readonly(name) else "other_fn", name, None))
        # pass 2: decide
        exec_hits = [x for c, _, x in cls if c == "target"]
        wrong, why, obs = False, set(), []
        for c, name, x in cls:
            if c in ("placeholder", "unknown_fn", "other_fn"):
                wrong = True
                why.add(c)
            elif c == "dup_target":
                if not self.readonly(name):
                    wrong = True
                    why.add("target_wrong_value")
            elif c == "target_wrong_value":
                if not (exec_hits and self.readonly(name)):  # extra read-only calls next to the execution are allowed
                    wrong = True
                    why.add(c)
            elif c == "replay":
                obs.append(("replay", x))
            elif c == "lookup":
                obs.append(("lookup", None))
        if exec_hits or wrong:
            self.actions.append("exec" if exec_hits else "wrong_call")
            if not wrong and len(set(exec_hits)) == len(self.targets) and self.gates[-1] == 1:
                return self._finish(True, "exec_ok")
            if "placeholder" in why:
                return self._finish(False, "call_placeholder")
            if not self.gates[-1] and (exec_hits or "target_wrong_value" in why):
                return self._finish(False, "exec_premature")  # target called before the missing value was obtained
            if "unknown_fn" in why:
                return self._finish(False, "call_unknown_fn")
            if "other_fn" in why:
                return self._finish(False, "call_other_fn")
            return self._finish(False, "exec_wrong_value" if "target_wrong_value" in why else "exec_incomplete")
        self.actions.append("fetch")
        if any(kind_o == "lookup" for kind_o, _ in obs):
            self.void = True  # a grounded read-only call that was never recorded: no truthful offline answer
            return self._finish(False, "void_unsimulated")
        bodies = []
        for _, j in obs:
            self.replayed.add(j)
            bodies.append(self.replays[j][2])
        if self.kind == "NEED_FETCH" and set(self.b["required"]) <= self.replayed:
            self.fetched_required = True
        if self.steps >= self.max_steps:
            return self._finish(False, "horizon")
        body = "\n".join(f"<tool_response>\n{o}\n</tool_response>" for o in bodies)
        return False, TOOL_TURN.format(body=body)

    def _step_ask(self, question):
        self.actions.append("ask")
        if self.steps >= self.max_steps:
            return self._finish(False, "horizon")
        if self.kind == "NEED_USER":
            q = set(words_of_name(question or ""))
            pending = [k for k in range(len(self.b["supply"])) if k not in self.supplied]
            parts = []
            for k in pending:
                s = self.b["supply"][k]
                hit = set(s["words"]) & q or s["label"].lower() in (question or "").lower()
                if hit or len(pending) == 1:
                    self.supplied.add(k)
                    parts.append(f"the {s['label']} is {_fmt(s['value'])}")
            reply = ("Sure, " + " and ".join(parts) + ".") if parts else REPLY_UNCLEAR
        elif self.kind == "NEED_FETCH":
            reply = REPLY_UNKNOWN_FETCH
        elif self.kind == "INFEASIBLE":
            reply = REPLY_INFEASIBLE
        else:
            reply = REPLY_ALREADY
        return False, USER_TURN.format(body=reply)

    def reward(self, gamma):
        return (gamma ** max(0, self.steps - 1)) * self.success if self.done else 0.0

    def first_action(self):
        return self.actions[0] if self.actions else "none"


class StopEpisode(InfoEpisode):
    """Adds STOP states: one terminal step; a text reply succeeds, re-issuing a recorded call ("repeat") or any other
    call fails."""

    def executable(self):
        if self.kind == "STOP":
            return True
        return super().executable()

    def step(self, text):
        if self.kind != "STOP":
            return super().step(text)
        _ACTIVE["ctx"], _ACTIVE["enum_args"] = self.span_ctx, self.enum_args
        text = close_think(text)
        self.steps += 1
        self.gates.append(1)
        p = parse_response(text)
        if p["kind"] == "none":
            self.actions.append("none")
            return self._finish(False, "none:" + str(p.get("reason")))
        if p["kind"] == "call":
            repeat = any(self._match_replay(c["function"]["name"], c["function"]["arguments"]) is not None
                         for c in p["calls"])
            self.actions.append("repeat" if repeat else "call")
            return self._finish(False, "stop_repeat" if repeat else "stop_call")
        self.actions.append(p["kind"])  # "tell" or "ask"
        return self._finish(True, "stop_ok")


class Episode(StopEpisode):
    """The training environment of StateAlign: all five task states, the extra-call scale and the continuation step.
    max_steps counts the continuation step: the states before execution get max_steps - 1 steps."""

    def __init__(self, b, max_steps=4):
        super().__init__(b, max(1, max_steps - 1))
        self.cont = b.get("cont")
        self.phase, self.scale, self.exec_step, self.inner, self.exec_calls = 0, 1.0, None, None, []

    def executable(self):
        if self.phase == 1:
            return self.inner.executable() if self.inner is not None else True
        return super().executable()

    def contrast_gate(self):
        return 0 if self.phase == 1 else 1  # continuation tokens receive outcome supervision only

    @staticmethod
    def _key(name, args):
        return name, json.dumps(canon(args if isinstance(args, dict) else {}, None), sort_keys=True, default=str)

    def step(self, text):
        if self.phase == 1:
            return self._cont_step(text)
        terminal, env = super().step(text)
        if not (terminal and self.label == "exec_ok"):
            return terminal, env
        calls = parse_response(close_think(text)).get("calls") or []
        extra = max(0, len(calls) - len(self.targets))
        self.scale = EXTRA_CALL_SCALE if extra else 1.0
        self.exec_step = self.steps
        if not self.cont:
            return terminal, env
        self.exec_calls = [self._key(c["function"]["name"], c["function"]["arguments"]) for c in calls]
        bodies = [m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                  for m in self.cont["tail"][1:]] + [UNINFORMATIVE] * extra
        if self.cont["kind"] == "exec":
            nb = dict(self.b, kind="EXEC", messages=_load(self.b["messages"]) + self.cont["tail"], cont=None)
            nb.update({k: v for k, v in self.cont["next"].items() if v is not None})
            nb["supply"], nb["required"] = [], []
            self.inner = StopEpisode(nb, 1)
        self.done, self.success, self.label, self.phase = False, 0, None, 1
        body = "\n".join(f"<tool_response>\n{o}\n</tool_response>" for o in bodies)
        return False, TOOL_TURN.format(body=body)

    def _cont_step(self, text):
        p = parse_response(close_think(text))
        calls = (p.get("calls") or []) if p["kind"] == "call" else []
        repeat = any(self._key(c["function"]["name"], c["function"]["arguments"]) in self.exec_calls for c in calls)
        if self.cont["kind"] == "stop":  # the source replies in text after the results
            self.steps += 1
            self.gates.append(1)
            if p["kind"] == "none":
                self.actions.append("none")
                return self._finish(False, "cont_none:" + str(p.get("reason")))
            if calls:
                stop = StopEpisode(dict(self.b, kind="STOP", replay=self.cont["next"]["replay"], cont=None), 1)
                repeat = repeat or any(stop._match_replay(c["function"]["name"], c["function"]["arguments"]) is not None
                                       for c in calls)
                self.actions.append("repeat" if repeat else "call")
                return self._finish(False, "cont_repeat" if repeat else "cont_call")
            self.actions.append(p["kind"])
            return self._finish(True, "cont_stop_ok")
        # the source executes again: the next recorded execution state, one step
        if repeat:
            self.steps += 1
            self.gates.append(1)
            self.actions.append("repeat")
            return self._finish(False, "cont_repeat")
        self.inner.step(text)
        self.steps += 1
        self.gates.append(self.inner.gates[-1] if self.inner.gates else 1)
        self.actions.append((self.inner.actions or ["none"])[-1])
        return self._finish(bool(self.inner.done and self.inner.success), "cont_" + (self.inner.label or "horizon"))

    def reward(self, gamma):
        if not self.done:
            return 0.0
        if self.exec_step is not None:
            return (gamma ** max(0, self.exec_step - 1)) * self.success * self.scale
        return super().reward(gamma)
