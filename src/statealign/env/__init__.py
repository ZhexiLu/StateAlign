"""Training environment: multi-step episodes over the five task states (simulated tool results and user replies, a
continuation step after the execution, outcome reward). See episode.py."""
from .episode import Episode, InfoEpisode, StopEpisode, TOOL_TURN, USER_TURN

KINDS = ("EXEC", "NEED_USER", "NEED_FETCH", "INFEASIBLE", "STOP")
