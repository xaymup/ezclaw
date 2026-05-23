"""Pressing [A] = 'allow for session' in any sub-agent's auth prompt
must authorize the entire MultiAgentSystem, not just that one role.

Regression test for the bug where each SpecializedAgent kept its own
session_authorized bool; the [A] key only authorized whichever agent
happened to ask, and the next architect step would re-prompt."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import _SharedAuthState, SpecializedAgent, MultiAgentSystem


def test_specialized_agents_share_auth_state_via_explicit_object():
    """Two agents constructed with the SAME shared_auth see each other's writes."""
    shared = _SharedAuthState()
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch.object(SpecializedAgent, "_pre_embed_tools", lambda self: None):
            a = SpecializedAgent(
                "executor",
                {"model": "m", "system_prompt": "p"},
                db=MagicMock(),
                auth_state=shared,
            )
            b = SpecializedAgent(
                "researcher",
                {"model": "m", "system_prompt": "p"},
                db=MagicMock(),
                auth_state=shared,
            )

    assert a.session_authorized is False
    assert b.session_authorized is False

    # User pressed [A] inside an executor prompt
    a.session_authorized = True

    # Researcher MUST also see it now
    assert b.session_authorized is True


def test_multi_agent_system_propagates_session_authorized_to_all_subagents():
    """The integrated path — when MultiAgentSystem.session_authorized is set
    via /authorize or via [A] in any prompt, every SpecializedAgent picks it up."""
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch.object(SpecializedAgent, "_pre_embed_tools", lambda self: None):
                with patch("multi_agent.load_skills", return_value=[]):
                    with patch("multi_agent.create_memory_tools", lambda db: None):
                        mas = MultiAgentSystem()

    assert mas.session_authorized is False
    for agent in mas.agents.values():
        assert agent.session_authorized is False

    # Simulate the cli.py setter path (used by /authorize command)
    mas.session_authorized = True

    # Every sub-agent reads True
    for name, agent in mas.agents.items():
        assert agent.session_authorized is True, \
            f"{name} did not pick up session_authorized through shared state"

    # And toggling back propagates too
    mas.session_authorized = False
    for agent in mas.agents.values():
        assert agent.session_authorized is False


def test_subagent_setting_session_authorized_propagates_to_siblings():
    """The realistic [A]-key path — a sub-agent's tool loop sets
    self.session_authorized = True when auth == 'allow_session'. ALL
    sibling sub-agents must see that immediately."""
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch.object(SpecializedAgent, "_pre_embed_tools", lambda self: None):
                with patch("multi_agent.load_skills", return_value=[]):
                    with patch("multi_agent.create_memory_tools", lambda db: None):
                        mas = MultiAgentSystem()

    executor = mas.agents["executor"]
    researcher = mas.agents["researcher"]
    assert researcher.session_authorized is False

    # Simulate: user pressed [A] during an executor auth prompt
    executor.session_authorized = True

    # Researcher should NOT re-prompt next time it has an auth-required tool
    assert researcher.session_authorized is True
    # And MultiAgentSystem-level property reads True
    assert mas.session_authorized is True
