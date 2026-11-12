"""Qwen3 chat format: the native chat template with thinking on (the format BFCL's Qwen3 FC handler uses).

Policy prompt   tools + interaction history, no system message, generation prompt with thinking enabled.
Scoring input   the same tools + history with one system message: the shared scoring instruction followed by the
                reference path of the task state (Appendix B, "State-contrast scoring"). The Qwen3 template places that
                system text in front of the tool list.
Environment turns (tool results, simulated user replies) are the Qwen3 turns of statealign.env.episode; a step ends
with the tokenizer's eos (<|im_end|>).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "scoring_instruction.txt")) as _fh:
    SCORING_INSTRUCTION = _fh.read()


def load(v):
    """A field may be a JSON string or a native object."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def reference_block(path):
    """The reference path of a task state: the ideal output of each step, in order."""
    return (
        "=== Ideal outputs for this conversation, step by step ===\n"
        + (path or "").strip()
        + "\n=== End ideal outputs ===\n"
        "At every step, produce the ideal output for the step you are at."
    )


def scoring_system(reference_path, instruction=SCORING_INSTRUCTION):
    return instruction.rstrip() + "\n\n" + reference_block(reference_path)


class Qwen3Format:
    name = "qwen3"

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def policy_prompt(self, tools, messages):
        return self._render(tools, messages, None)

    def scoring_prompt(self, tools, messages, reference_path):
        return self._render(tools, messages, scoring_system(reference_path))

    def _render(self, tools, messages, system):
        history = load(messages) or []
        msgs = ([{"role": "system", "content": system}] if system is not None else []) + history
        return self.tok.apply_chat_template(msgs, tools=load(tools) or None, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=True)

    def stop_ids(self):
        return {self.tok.eos_token_id}
