# Example data

`statealign_example_100.jsonl` holds 100 task states in the format the training code reads. It is an example of the
training data, large enough to run the whole training path (`configs/train/example_qwen3_8b.sh`). The full training
set used in the paper is not part of this release.

The states are constructed from trajectories of two public tool-use datasets,
[ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) and
[LoopTool-23k](https://huggingface.co/datasets/zhangkangning/LoopTool-23k), both released under Apache-2.0.
Names, e-mail addresses, phone numbers and identifiers in the conversations are synthetic values from these sources.

## Contents

| Task state (`kind`) | What the agent has to do | States |
|---|---|---|
| `EXEC` | every required value is available: execute the target call(s) | 40 |
| `NEED_USER` | a required value is missing and only the user can supply it: ask, then execute | 20 |
| `NEED_FETCH` | a required value is missing and a listed read-only tool can return it: retrieve it, then execute | 15 |
| `INFEASIBLE` | no listed tool can fulfil the request: decline without calling a substitute | 10 |
| `STOP` | the tool results already answer the request: reply in text, without another call | 15 |

71 states come with a paired state: the same task with one condition changed, so that a different next action is
required (for example, the same request with the needed value stated by the user). 24 states have a continuation: the
episode goes on for one more step after the execution.

## Row format

One JSON object per line, one task state per object.

| Field | Meaning |
|---|---|
| `bid` | state id |
| `kind` | task state: `EXEC`, `NEED_USER`, `NEED_FETCH`, `INFEASIBLE` or `STOP` |
| `tools` | tool list (JSON string, OpenAI function format) |
| `messages` | interaction history (JSON string); earlier calls are written as `<tool_call>` blocks |
| `target` | target calls: `name`, `arguments`, `check` (the arguments compared with the generated call), `equiv` (functions accepted as equivalent) |
| `supply` | `NEED_USER`: the values the simulated user gives when asked (`function`, `arg`, `value`), with the `label` and `words` used to recognise which value a question asks for |
| `replay` | recorded calls of the source trajectory with their `observation`; a re-issued call returns the recorded observation (`key_args`: the arguments that must match) |
| `required` | `NEED_FETCH`: indices into `replay` of the calls that must be made before the execution |
| `lookup_fns` | read-only lookup functions of the tool list |
| `gold_response` | reference path of the state, as numbered steps. It is input to the state-contrast scorer only. The environment never reads it. |
| `twin_kind`, `twin_tools`, `twin_messages`, `twin_ref` | the paired state: its kind, tool list, history and reference path (`null` when the state has no pair). For a `STOP` state the pair is the same task before the tool results were returned. |
| `cont` | continuation (`null` when absent): `kind` is `stop` (the next step is a text reply) or `exec` (the next step is another execution), `tail` holds the call turn and its results, and `next` holds what the next step is judged on |

`src/statealign/env/episode.py` documents how an episode on a state proceeds and how its outcome reward is computed.

## Stepping through a state

The environment needs no GPU. Run this from the repository root with `PYTHONPATH=src`:

```python
import json
from statealign.env import Episode

rows = [json.loads(line) for line in open("data/statealign_example_100.jsonl")]
row = next(r for r in rows if r["kind"] == "NEED_USER")
episode = Episode(row, max_steps=4)

# a model step is the raw output: the reasoning block, then the action
ask = "<think>\nA required value is missing.\n</think>\n\nCould you tell me the " + row["supply"][0]["label"] + "?"
done, env_turn = episode.step(ask)
print(done)          # False: the simulated user replies and the episode continues
print(env_turn)      # the reply, formatted as the next turn of the chat template

target = row["target"][0]
call = "<think>\nNow the call can be made.\n</think>\n\n<tool_call>\n" + json.dumps(
    {"name": target["name"], "arguments": target["arguments"]}) + "\n</tool_call>"
done, env_turn = episode.step(call)
print(done, episode.reward(gamma=0.8))   # True 0.8: a correct execution at the second step
```
