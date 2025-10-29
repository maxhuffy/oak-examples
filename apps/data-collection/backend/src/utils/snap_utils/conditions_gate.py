import time
from typing import Dict, Iterable, Optional


class ConditionsGate:
    """
    Generic per-condition cooldown gate.
    - First hit for a key fires immediately.
    - Subsequent hits for the same key are allowed only after its cooldown.
    - Supports global enable + per-key enable + per-key cooldown overrides.
    """

    def __init__(self, default_cooldown_s: float = 0.0, enabled: bool = True):
        self.enabled = bool(enabled)
        self.default_cooldown_s = max(0.0, float(default_cooldown_s))
        self.cooldowns: Dict[str, float] = {}
        self.key_enabled: Dict[str, bool] = {}
        self.last_sent: Dict[str, float] = {}

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    def set_default(self, seconds: float) -> None:
        self.default_cooldown_s = max(0.0, float(seconds))

    def set_key_enabled(self, key: str, on: bool) -> None:
        self.key_enabled[str(key)] = bool(on)

    def set_cooldown(self, key: str, seconds: float) -> None:
        self.cooldowns[str(key)] = max(0.0, float(seconds))

    def reset(self, keys: Optional[Iterable[str]] = None) -> None:
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
