"""Tests for rule evaluation."""

from __future__ import annotations

from torchquant._types import Algorithm, LayerKind
from torchquant.config import QuantRecipe, QuantScheme
from torchquant.graph import QuantNode
from torchquant.rules import QuantDecision, RuleEngine, default_rule


def _node(fqn: str = "linear") -> QuantNode:
    return QuantNode(fqn=fqn, kind=LayerKind.LINEAR, param_count=16)


def _scheme(weight_bits: int, algorithm: Algorithm = Algorithm.RTN) -> QuantScheme:
    return QuantScheme(weight_bits=weight_bits, algorithm=algorithm)


def test_default_rule_respects_ignore_before_overrides() -> None:
    """Ignore entries win even when an override exists for the same FQN."""
    recipe = QuantRecipe(
        overrides={"linear": _scheme(weight_bits=4, algorithm=Algorithm.GPTQ)},
        ignore=frozenset({"linear"}),
    )

    assert default_rule(_node(), recipe) is None


def test_default_rule_prefers_override_over_default_scheme() -> None:
    """Overrides replace the default scheme for exact FQN matches."""
    default_scheme = _scheme(weight_bits=8)
    override_scheme = _scheme(weight_bits=4, algorithm=Algorithm.GPTQ)
    recipe = QuantRecipe(
        default_scheme=default_scheme,
        overrides={"linear": override_scheme},
    )

    assert default_rule(_node(), recipe) == QuantDecision(
        node=_node(),
        scheme=override_scheme,
    )


def test_rule_engine_uses_first_matching_rule() -> None:
    """Rule evaluation stops after the first non-None decision."""
    calls: list[str] = []
    recipe = QuantRecipe(default_scheme=_scheme(weight_bits=8))
    node = _node()

    def first_rule(_node: QuantNode, _recipe: QuantRecipe) -> QuantDecision | None:
        calls.append("first")
        return None

    def second_rule(
        current_node: QuantNode, current_recipe: QuantRecipe
    ) -> QuantDecision:
        calls.append("second")
        return QuantDecision(node=current_node, scheme=current_recipe.default_scheme)

    def third_rule(_node: QuantNode, _recipe: QuantRecipe) -> QuantDecision:
        calls.append("third")
        return QuantDecision(node=node, scheme=_scheme(weight_bits=2))

    decisions = RuleEngine([first_rule, second_rule, third_rule]).decide([node], recipe)

    assert calls == ["first", "second"]
    assert decisions == [QuantDecision(node=node, scheme=recipe.default_scheme)]


def test_rule_engine_skips_nodes_when_all_rules_return_none() -> None:
    """Nodes are omitted when no rule matches them."""
    recipe = QuantRecipe(default_scheme=_scheme(weight_bits=8))
    node = _node()

    def no_match_rule(_node: QuantNode, _recipe: QuantRecipe) -> QuantDecision | None:
        return None

    decisions = RuleEngine([no_match_rule]).decide([node], recipe)

    assert decisions == []
