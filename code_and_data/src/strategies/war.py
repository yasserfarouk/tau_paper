from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field
from negams.gb.components.acceptance import AcceptAnyRational
from negmas.common import PreferencesChange, PreferencesChangeType
from negmas.gb.common import GBState, ResponseType
from negmas.gb.components.base import AcceptancePolicy, OfferingPolicy
from negmas.gb.negotiators.modular.mapneg import MAPNegotiator

from negmas import PresortingInverseUtilityFunction

if TYPE_CHECKING:
    from negmas.common import PreferencesChange
    from negmas.gb import GBState
    from negmas.gb.negotiators.base import GBNegotiator
    from negmas.outcomes import Outcome

from negmas import warnings

__all__ = [
    "WARNegotiator",
]


@define
class AcceptAnyRational(AcceptancePolicy):
    """
    Accepts any rational outcome.
    """

    def __call__(
        self, state: GBState, offer: Outcome, source: str | None
    ) -> ResponseType:
        if not self.negotiator or not self.negotiator.preferences:
            return ResponseType.REJECT_OFFER
        pref = self.negotiator.preferences
        if not pref.is_worse(offer, None):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


@define
class WAROfferingPolicy(OfferingPolicy):
    next_indx: int = 0
    sorter: PresortingInverseUtilityFunction | None = field(repr=False, default=None)
    _last_offer: Outcome | None = field(init=False, default=None)
    _repeating: bool = field(init=False, default=False)
    _irrational: bool = field(init=False, default=True)
    _irrational_index: int = field(init=False, default=-1)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            return
        self._irrational = True
        self._irrational_index = int(self.negotiator.nmi.n_outcomes) - 1
        if any(
            _.type
            not in (
                PreferencesChangeType.Scale,
                PreferencesChangeType.ReservedOutcome,
                PreferencesChangeType.ReservedValue,
            )
            for _ in changes
        ):
            if self.sorter is not None:
                warnings.warn(
                    f"Sorter is already initialized. May be on_preferences_changed is called twice!!"
                )
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
            self.next_indx = 0
            self._repeating = False

    def on_negotiation_start(self, state) -> None:
        self._repeating = False
        self._irrational = True
        self._irrational_index = self.negotiator.nmi.n_outcomes - 1  # type: ignore
        return super().on_negotiation_start(state)

    def __call__(self, state: GBState) -> Outcome | None:
        if not self.negotiator or not self.negotiator.ufun or not self.negotiator.nmi:
            return self._last_offer
        if self._repeating:
            return self._last_offer
        if not self._irrational and self.next_indx >= self.negotiator.nmi.n_outcomes:
            return self._last_offer
        if not self.sorter:
            warnings.warn(
                f"Sorter is not initialized. May be on_preferences_changed is never called before propose!!"
            )
            self.sorter = PresortingInverseUtilityFunction(
                self.negotiator.ufun, rational_only=True, eps=-1, rel_eps=-1
            )
            self.sorter.init()
        nxt = self._irrational_index if self._irrational else self.next_indx
        outcome = self.sorter.outcome_at(nxt)
        if self._irrational:
            if (
                outcome is None
                or self.sorter.utility_at(self._irrational_index)
                >= self.negotiator.ufun.reserved_value
            ):
                self._irrational = False
                assert self._last_offer is None
                outcome = self.sorter.outcome_at(self.next_indx)
            else:
                self._irrational_index -= 1
                return outcome
        if (
            outcome is None
            or self.sorter.utility_at(self.next_indx)
            < self.negotiator.ufun.reserved_value
        ):
            self._repeating = True
            return self._last_offer
        self.next_indx += 1
        self._last_offer = outcome
        return outcome


class WARNegotiator(MAPNegotiator):
    """
    Wasting Accepting Any (an equilibrium but not complete)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptAnyRational()
        kwargs["offering"] = WAROfferingPolicy()
        super().__init__(*args, **kwargs)
