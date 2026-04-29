"""Tests for RoundRobinNegotiationManager parsing helpers."""

from __future__ import annotations

import pytest

from simulation.scenarios.fishing.contracting.negotiation import (
    RoundRobinNegotiationManager,
    _extract_constitution_text,
    _parse_vote_tag,
)


def _mgr(n_agents: int = 3) -> RoundRobinNegotiationManager:
    return RoundRobinNegotiationManager(
        agents=[],  # type: ignore[arg-type]
        max_turns=10,
        min_agree_agents=n_agents,
    )


# ---------------------------------------------------------------------------
# _parse_vote_tag
# ---------------------------------------------------------------------------

class TestParseVoteTag:
    def test_vote_yes_tag(self):
        assert _parse_vote_tag("<VOTE_YES> looks good") is True

    def test_vote_yes_tag_case_insensitive(self):
        assert _parse_vote_tag("<vote_yes> ok") is True

    def test_bare_yes_at_start(self):
        assert _parse_vote_tag("Yes, I accept the contract.") is False

    def test_bare_yes_uppercase_at_start(self):
        assert _parse_vote_tag("YES this works") is False

    def test_yes_buried_in_text_not_counted(self):
        # "yes" appears but NOT at the start of the message
        assert _parse_vote_tag("No, but yes in principle") is False

    def test_vote_no(self):
        assert _parse_vote_tag("<VOTE_NO> reject") is False

    def test_empty(self):
        assert _parse_vote_tag("") is False


# ---------------------------------------------------------------------------
# _parse_agreement_signal — both tags detected independently
# ---------------------------------------------------------------------------

class TestParseAgreementSignal:
    def setup_method(self):
        self.mgr = _mgr()

    def test_agree_only(self):
        has_agree, has_disagree = self.mgr._parse_agreement_signal("<AGREE> sounds good")
        assert has_agree is True
        assert has_disagree is False

    def test_disagree_only(self):
        has_agree, has_disagree = self.mgr._parse_agreement_signal("<DISAGREE> I reject this")
        assert has_agree is False
        assert has_disagree is True

    def test_both_tags_detected(self):
        # Bug fix: previously returned (True, False) — DISAGREE was ignored
        has_agree, has_disagree = self.mgr._parse_agreement_signal("<AGREE> <DISAGREE>")
        assert has_agree is True
        assert has_disagree is True

    def test_agree_with_target(self):
        has_agree, has_disagree = self.mgr._parse_agreement_signal('<AGREE to="Alice">')
        assert has_agree is True
        assert has_disagree is False

    def test_agree_with_bare_name(self):
        has_agree, has_disagree = self.mgr._parse_agreement_signal("<AGREE Alice>")
        assert has_agree is True
        assert has_disagree is False

    def test_no_tags(self):
        has_agree, has_disagree = self.mgr._parse_agreement_signal("Let us all fish less.")
        assert has_agree is False
        assert has_disagree is False


# ---------------------------------------------------------------------------
# _is_own_proposal_message
# ---------------------------------------------------------------------------

class TestIsOwnProposalMessage:
    def setup_method(self):
        self.mgr = _mgr()

    # Pure agreement — should NOT be own proposal
    def test_agree_no_constitution(self):
        response = "<AGREE> I agree with Alice's proposal."
        nl = "I agree with Alice's proposal."
        assert self.mgr._is_own_proposal_message("Bob", response, nl, True, False) is False

    # New proposal without any tags — IS own proposal
    def test_new_proposal_no_tags(self):
        response = "<constitution>Fish 8 tons each</constitution>"
        nl = "Fish 8 tons each"
        assert self.mgr._is_own_proposal_message("Alice", response, nl, False, False) is True

    # DISAGREE + constitution block — IS a counterproposal (was broken: always returned False)
    def test_disagree_with_constitution_is_counterproposal(self):
        response = "<DISAGREE> I reject that. <constitution>Fish 6 tons each</constitution>"
        nl = "Fish 6 tons each"
        result = self.mgr._is_own_proposal_message("Bob", response, nl, False, True)
        assert result is True  # counterproposal

    # DISAGREE without constitution — NOT own proposal
    def test_disagree_without_constitution_not_proposal(self):
        response = "<DISAGREE> I reject this entirely."
        nl = "I reject this entirely."
        result = self.mgr._is_own_proposal_message("Bob", response, nl, False, True)
        assert result is False

    # AGREE + constitution — treated as new proposal (constitution takes precedence)
    def test_agree_with_constitution_is_proposal(self):
        response = "<AGREE> <constitution>Fish 10 tons each</constitution>"
        nl = "Fish 10 tons each"
        result = self.mgr._is_own_proposal_message("Carol", response, nl, True, False)
        assert result is True

    # Empty nl_contract — never own proposal
    def test_empty_nl_contract(self):
        result = self.mgr._is_own_proposal_message("X", "<AGREE>", None, True, False)
        assert result is False


# ---------------------------------------------------------------------------
# _check_consensus
# ---------------------------------------------------------------------------

class TestCheckConsensus:
    def test_consensus_reached(self):
        mgr = _mgr(n_agents=3)
        agreements = {
            "Alice": "Fish 8 tons each",
            "Bob": "Fish 8 tons each",
            "Carol": "fish 8 tons each",  # case difference — normalizes
        }
        result = mgr._check_consensus(agreements)
        assert result is not None
        assert result[0] in ("Fish 8 tons each", "fish 8 tons each")
        assert len(result[1]) == 3

    def test_consensus_not_reached(self):
        mgr = _mgr(n_agents=3)
        agreements = {
            "Alice": "Fish 8 tons each",
            "Bob": "Fish 10 tons each",
            "Carol": "Fish 8 tons each",
        }
        result = mgr._check_consensus(agreements)
        assert result is None  # only 2 agree, need 3

    def test_consensus_partial_agreements(self):
        mgr = _mgr(n_agents=2)
        agreements = {
            "Alice": "Fish 8 tons each",
            "Bob": "Fish 8 tons each",
        }
        result = mgr._check_consensus(agreements)
        assert result is not None


# ---------------------------------------------------------------------------
# _extract_agree_target
# ---------------------------------------------------------------------------

class TestExtractAgreeTarget:
    def setup_method(self):
        self.mgr = _mgr()

    def test_no_target(self):
        assert self.mgr._extract_agree_target("<AGREE>") is None

    def test_bare_name(self):
        assert self.mgr._extract_agree_target("<AGREE Alice>") == "Alice"

    def test_keyed_unquoted(self):
        assert self.mgr._extract_agree_target("<AGREE to=Alice>") == "Alice"

    def test_keyed_quoted(self):
        assert self.mgr._extract_agree_target('<AGREE to="Alice">') == "Alice"

    def test_no_agree_tag(self):
        assert self.mgr._extract_agree_target("no tag here") is None


# ---------------------------------------------------------------------------
# _ensure_tagged_round_robin_response — constitution block skips clarification
# ---------------------------------------------------------------------------

class TestEnsureTaggedResponse:
    """Non-async smoke test: verify the signal-detection branch is correct."""

    def setup_method(self):
        self.mgr = _mgr()

    def test_constitution_block_no_tag_is_a_proposal(self):
        response = "I propose this: <constitution>Fish 8 tons each month.</constitution>"
        # Must NOT trigger re-prompting: constitution block is itself a clear signal
        has_agree, has_disagree = self.mgr._parse_agreement_signal(response)
        has_constitution = _extract_constitution_text(response) is not None
        assert has_agree is False
        assert has_disagree is False
        assert has_constitution is True  # skip-condition is satisfied


# ---------------------------------------------------------------------------
# _resolve_agreement_target_contract — transitive resolution
# ---------------------------------------------------------------------------

class TestResolveAgreementTargetContract:
    def setup_method(self):
        self.mgr = _mgr()

    def test_direct_proposer_lookup(self):
        result = self.mgr._resolve_agreement_target_contract(
            target_proposer="Jack",
            proposal_by_proposer={"jack": "cap at 8 tons"},
            proposals=[(0, "Jack", "cap at 8 tons")],
        )
        assert result == "cap at 8 tons"

    def test_bare_agree_falls_back_to_last_proposal(self):
        # <AGREE> with no target → latest proposal
        result = self.mgr._resolve_agreement_target_contract(
            target_proposer=None,
            proposal_by_proposer={"jack": "cap at 8 tons"},
            proposals=[(0, "Jack", "cap at 8 tons"), (1, "Emma", "cap at 6 tons")],
        )
        assert result == "cap at 6 tons"

    def test_transitive_agree_to_emma_who_agreed_to_jack(self):
        # Emma agreed to Jack's proposal (Emma not in proposal_by_proposer).
        # Luke says <AGREE to=Emma> — should resolve to Jack's contract transitively.
        agreements = {
            "Jack": "cap at 8 tons",
            "Emma": "cap at 8 tons",  # Emma already agreed to Jack
        }
        result = self.mgr._resolve_agreement_target_contract(
            target_proposer="Emma",
            proposal_by_proposer={"jack": "cap at 8 tons"},
            proposals=[(0, "Jack", "cap at 8 tons")],
            agreements=agreements,
        )
        assert result == "cap at 8 tons"

    def test_unknown_target_returns_none(self):
        # Target hasn't spoken at all — should not fall back to last proposal.
        result = self.mgr._resolve_agreement_target_contract(
            target_proposer="Nobody",
            proposal_by_proposer={"jack": "cap at 8 tons"},
            proposals=[(0, "Jack", "cap at 8 tons")],
            agreements={"Jack": "cap at 8 tons"},
        )
        assert result is None

    def test_transitive_chain_produces_original_contract(self):
        # Full scenario: Jack proposes, Emma agrees to Jack, Luke agrees to Emma.
        # All three should end up with Jack's contract.
        agreements = {
            "Jack": "cap at 8 tons",
            "Emma": "cap at 8 tons",  # agreed to Jack
        }
        luke_result = self.mgr._resolve_agreement_target_contract(
            target_proposer="Emma",
            proposal_by_proposer={"jack": "cap at 8 tons"},
            proposals=[(0, "Jack", "cap at 8 tons")],
            agreements=agreements,
        )
        assert luke_result == "cap at 8 tons"  # Jack's original proposal
