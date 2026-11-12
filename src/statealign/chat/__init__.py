"""Chat format: prompt rendering for the policy and for the state-contrast scorer.

This release covers Qwen3 models (native chat template, JSON <tool_call> blocks, thinking on). Qwen3Format exposes
policy_prompt(tools, messages), scoring_prompt(tools, messages, reference_path) and stop_ids().
"""
from .qwen3 import Qwen3Format
