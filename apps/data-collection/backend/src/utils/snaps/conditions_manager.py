import time
from typing import Dict, Iterable, Optional
from .conditions.base_condition import ConditionBase


class ConditionsManager:
    """
    Manages registration and evaluation of snap trigger conditions.

    Tracks cooldowns, enabled states, and timing for each condition,
    ensuring that individual triggers fire only when their criteria
    and cooldown constraints are met.
    """

    def __init__(self, default_cooldown_s: float = 300, enabled: bool = True):
        self.enabled = bool(enabled)
        self.default_cooldown_s = max(0.0, float(default_cooldown_s))
        self.cooldowns: Dict[str, float] = {}
        self.key_enabled: Dict[str, bool] = {}
        self.last_sent: Dict[str, float] = {}
        self.conditions: Dict[str, "ConditionBase"] = {}

    def register_condition(
        self, condition: ConditionBase, enabled=False, cooldown_s=-1.0
    ):
        key = condition.key
        self.conditions[key] = condition
        self.key_enabled[key] = enabled
        self.cooldowns[key] = cooldown_s if cooldown_s >= 0 else self.default_cooldown_s

    def register_conditions(self, conditions: Iterable[ConditionBase]):
        for cond in conditions:
            self.register_condition(cond)

    def set_enabled(self, on: bool) -> None:
        self.enabled = on

    def set_key_enabled(self, key: str, on: bool) -> None:
        self.key_enabled[str(key)] = bool(on)

    def set_cooldown(self, key: str, seconds: float) -> None:
        self.cooldowns[str(key)] = max(0.0, float(seconds))

    def reset_cooldowns(self, keys: Optional[Iterable[str]] = None) -> None:
        if keys is None:
            self.last_sent.clear()
        else:
            for k in keys:
                self.last_sent.pop(str(k), None)

    def is_key_enabled(self, key: str) -> bool:
        return self.key_enabled.get(str(key), False)

    def get_cooldown(self, key: str) -> float:
        return self.cooldowns.get(str(key), self.default_cooldown_s)

    def hit(self, key: str) -> bool:
        """
        Returns True if this condition/key should fire now (and records the hit),
        False if it's still cooling down or globally/key disabled.
        """
        if not self.enabled or not self.is_key_enabled(key):
            return False
        now = time.time()
        last = self.last_sent.get(key, -1.0)
        cd = self.get_cooldown(key)
        if last < 0.0 or (now - last) >= cd:
            self.last_sent[key] = now
            return True
        return False

    def evaluate(self, **kwargs):
        """
        Iterate through all registered conditions,
        yield (condition, extras) for those that fire.
        """
        for key, cond in self.conditions.items():
            if self.is_key_enabled(key) and cond.should_trigger(self, **kwargs):
                yield cond, cond.handle(**kwargs)
