"""Rule engine: assign quantization schemes to discovered nodes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .config import QuantRecipe, QuantScheme
from .graph import QuantNode

type Rule = Callable[[QuantNode, QuantRecipe], QuantDecision | None]


@dataclass(frozen=True)
class QuantDecision:
    """The rule engine's verdict for one node."""

    node: QuantNode
    scheme: QuantScheme


class RuleEngine:
    """Ordered rule chain -- first match wins."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        self._rules = list(rules)

    def decide(
        self,
        nodes: list[QuantNode],
        recipe: QuantRecipe,
    ) -> list[QuantDecision]:
        """Apply rules to all nodes and return decisions.

        Args:
            nodes: Quantizable nodes discovered from the model.
            recipe: Quantization recipe with default scheme and overrides.

        Returns:
            List of decisions, one per quantized node.
        """
        raise NotImplementedError


def default_rule(node: QuantNode, recipe: QuantRecipe) -> QuantDecision | None:
    """Default rule: apply recipe's default scheme, respecting overrides/ignore.

    Args:
        node: A quantizable node.
        recipe: The quantization recipe.

    Returns:
        A QuantDecision if the node should be quantized, None otherwise.
    """
    raise NotImplementedError
