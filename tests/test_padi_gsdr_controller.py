from padi_oft.runtime.gsdr_controller import PadiGSDRConfig, PadiGSDRController, PadiGSDROutput


def test_gsdr_imports():
    assert PadiGSDRConfig is not None
    assert PadiGSDRController is not None
    assert PadiGSDROutput is not None


def test_gsdr_config_defaults():
    cfg = PadiGSDRConfig()
    assert cfg.g_low == 0.20
    assert cfg.g_no_prune == 0.95
    assert cfg.geometry_ema_alpha == 0.65
    assert cfg.token_quantum == 2
    assert cfg.full_keep_only_on_no_prune is True


def test_gsdr_low_risk_uses_base_budget():
    ctrl = PadiGSDRController()
    out = ctrl.update(geometry_risk=0.0, base_keep_ratio=0.5, num_vision_tokens=256)
    assert out.keep_tokens == 128
    assert out.keep_ratio == 0.5
    assert out.no_prune is False


def test_gsdr_medium_risk_increases_budget():
    ctrl = PadiGSDRController()
    out = None
    for _ in range(5):
        out = ctrl.update(geometry_risk=0.7, base_keep_ratio=0.5, num_vision_tokens=256)
    assert out is not None
    assert out.keep_tokens > out.base_keep_tokens
    assert out.keep_tokens < out.num_vision_tokens
    assert out.keep_ratio > out.base_keep_ratio
    assert out.no_prune is False


def test_gsdr_raw_no_prune_immediate():
    ctrl = PadiGSDRController()
    out = ctrl.update(geometry_risk=0.96, base_keep_ratio=0.5, num_vision_tokens=256)
    assert out.no_prune is True
    assert out.keep_tokens == 256
    assert out.keep_ratio == 1.0


def test_gsdr_non_no_prune_cannot_full_keep_by_default():
    ctrl = PadiGSDRController()
    out = None
    for _ in range(12):
        out = ctrl.update(geometry_risk=0.94, base_keep_ratio=0.5, num_vision_tokens=512)
    assert out is not None
    assert out.no_prune is False
    assert out.keep_tokens < out.num_vision_tokens
    assert out.keep_tokens == out.num_vision_tokens - out.debug["token_quantum"]


def test_gsdr_no_prune_allows_full_keep():
    ctrl = PadiGSDRController()
    out = ctrl.update(geometry_risk=0.96, base_keep_ratio=0.5, num_vision_tokens=512)
    assert out.no_prune is True
    assert out.keep_tokens == 512
    assert out.keep_ratio == 1.0
    assert out.keep_ratio_quantized == 1.0


def test_gsdr_debug_contains_continuous_and_quantized_budget():
    ctrl = PadiGSDRController()
    out = ctrl.update(geometry_risk=0.6, base_keep_ratio=0.5, num_vision_tokens=256)
    assert "keep_ratio_cont" in out.debug
    assert "raw_keep_tokens" in out.debug
    assert "keep_ratio_quantized" in out.debug
    assert "full_keep_only_on_no_prune" in out.debug


def test_gsdr_token_quantum_rounds_up():
    ctrl = PadiGSDRController(PadiGSDRConfig(token_quantum=2))
    out = ctrl.update(geometry_risk=0.4, base_keep_ratio=0.51, num_vision_tokens=251)
    assert out.keep_tokens % ctrl.config.token_quantum == 0
    assert out.keep_tokens >= out.base_keep_tokens


def test_gsdr_never_below_base_budget():
    ctrl = PadiGSDRController()
    for risk in [0.0, 0.5, 0.96]:
        out = ctrl.update(geometry_risk=risk, base_keep_ratio=0.5, num_vision_tokens=256)
        assert out.keep_tokens >= out.base_keep_tokens


def test_gsdr_repeated_high_risk_budget_increases():
    ctrl = PadiGSDRController()
    out = None
    for _ in range(8):
        out = ctrl.update(geometry_risk=0.949, base_keep_ratio=0.5, num_vision_tokens=256)
    assert out is not None
    assert out.no_prune is False
    assert out.keep_tokens > out.base_keep_tokens


def test_gsdr_reset():
    ctrl = PadiGSDRController()
    ctrl.update(geometry_risk=0.9, base_keep_ratio=0.5, num_vision_tokens=256)
    ctrl.reset()
    assert ctrl.state.geometry_risk_smooth == 0.0
    assert ctrl.state.step == 0
