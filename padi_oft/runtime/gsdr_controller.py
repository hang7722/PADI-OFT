import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _smoothstep(x: float) -> float:
    x = _clamp01(x)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class PadiGSDRConfig:
    g_low: float = 0.25
    g_no_prune: float = 0.95
    max_keep_ratio: float = 0.75
    geometry_ema_alpha: float = 0.65
    token_quantum: int = 2
    full_keep_only_on_no_prune: bool = True

    def __post_init__(self) -> None:
        assert 0.0 <= self.g_low < self.g_no_prune <= 1.0
        assert 0.0 < self.max_keep_ratio <= 1.0
        assert self.g_low < self.g_no_prune
        assert 0.0 < self.geometry_ema_alpha <= 1.0
        assert self.token_quantum >= 1


@dataclass
class PadiGSDRState:
    geometry_risk_smooth: float = 0.0
    step: int = 0


@dataclass
class PadiGSDROutput:
    g_raw: float
    geometry_risk_smooth: float
    base_keep_ratio: float
    keep_ratio_cont: float
    raw_keep_tokens: float
    keep_ratio_quantized: float
    keep_ratio: float
    base_keep_tokens: int
    keep_tokens: int
    num_vision_tokens: int
    no_prune: bool
    debug: Dict[str, Any] = field(default_factory=dict)


class PadiGSDRController:
    def __init__(self, config: Optional[PadiGSDRConfig] = None):
        self.config = config or PadiGSDRConfig()
        self.state = PadiGSDRState()

    def reset(self) -> None:
        self.state = PadiGSDRState()

    def update(self, geometry_risk: float, base_keep_ratio: float, num_vision_tokens: int) -> PadiGSDROutput:
        assert 0.0 < base_keep_ratio <= 1.0
        assert num_vision_tokens > 0

        c = self.config
        g_raw = _clamp01(geometry_risk)
        prev_smooth = self.state.geometry_risk_smooth
        geometry_risk_smooth = c.geometry_ema_alpha * g_raw + (1.0 - c.geometry_ema_alpha) * prev_smooth
        self.state.geometry_risk_smooth = geometry_risk_smooth
        self.state.step += 1

        no_prune = g_raw >= c.g_no_prune or geometry_risk_smooth >= c.g_no_prune

        effective_max_keep_ratio = max(base_keep_ratio, min(1.0, c.max_keep_ratio))

        if no_prune:
            keep_ratio_cont = 1.0
        elif geometry_risk_smooth <= c.g_low:
            keep_ratio_cont = base_keep_ratio
        else:
            x = (geometry_risk_smooth - c.g_low) / (c.g_no_prune - c.g_low)
            boost = _smoothstep(x)
            keep_ratio_cont = base_keep_ratio + (effective_max_keep_ratio - base_keep_ratio) * boost

        base_keep_tokens = int(math.ceil(num_vision_tokens * base_keep_ratio))
        raw_keep_tokens = num_vision_tokens * keep_ratio_cont
        keep_tokens = int(math.ceil(raw_keep_tokens / c.token_quantum) * c.token_quantum)
        keep_tokens = max(base_keep_tokens, min(num_vision_tokens, keep_tokens))

        if no_prune:
            keep_tokens = num_vision_tokens

        if not no_prune and c.full_keep_only_on_no_prune:
            max_non_full_tokens = max(base_keep_tokens, num_vision_tokens - c.token_quantum)
            keep_tokens = min(keep_tokens, max_non_full_tokens)

        keep_ratio_quantized = keep_tokens / num_vision_tokens
        keep_ratio = keep_ratio_quantized

        debug = {
            "g_raw": g_raw,
            "geometry_risk_smooth": geometry_risk_smooth,
            "base_keep_ratio": base_keep_ratio,
            "keep_ratio_cont": keep_ratio_cont,
            "raw_keep_tokens": raw_keep_tokens,
            "keep_ratio_quantized": keep_ratio_quantized,
            "keep_ratio": keep_ratio,
            "base_keep_tokens": base_keep_tokens,
            "keep_tokens": keep_tokens,
            "num_vision_tokens": num_vision_tokens,
            "no_prune": no_prune,
            "g_low": c.g_low,
            "g_no_prune": c.g_no_prune,
            "max_keep_ratio": c.max_keep_ratio,
            "effective_max_keep_ratio": effective_max_keep_ratio,
            "geometry_ema_alpha": c.geometry_ema_alpha,
            "token_quantum": c.token_quantum,
            "full_keep_only_on_no_prune": c.full_keep_only_on_no_prune,
            "step": self.state.step,
        }

        return PadiGSDROutput(
            g_raw=g_raw,
            geometry_risk_smooth=geometry_risk_smooth,
            base_keep_ratio=base_keep_ratio,
            keep_ratio_cont=keep_ratio_cont,
            raw_keep_tokens=raw_keep_tokens,
            keep_ratio_quantized=keep_ratio_quantized,
            keep_ratio=keep_ratio,
            base_keep_tokens=base_keep_tokens,
            keep_tokens=keep_tokens,
            num_vision_tokens=num_vision_tokens,
            no_prune=no_prune,
            debug=debug,
        )
