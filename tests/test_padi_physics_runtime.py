from padi_oft.runtime.physics_runtime import (
    PadiPhysicsAwareRuntime,
    PadiPhysicsConfig,
    PadiPhysicsState,
    PadiSignalOutput,
)
from padi_oft.runtime.video_overlay import overlay_padi_scores_on_frame
import numpy as np


def test_imports():
    assert PadiPhysicsAwareRuntime is not None
    assert PadiPhysicsConfig is not None
    assert PadiPhysicsState is not None
    assert PadiSignalOutput is not None


def test_first_frame_defaults():
    rt = PadiPhysicsAwareRuntime()
    out = rt.update(eef_pos=[0, 0, 0], gripper_value=0.5)
    assert out.transit_score == 0.0
    assert out.geometry_risk == 0.0
    assert out.precise_active is False


def test_output_range_multi_step():
    rt = PadiPhysicsAwareRuntime(PadiPhysicsConfig(startup_guard_steps=0))
    seq = [
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.015, 0.005, 0.002],
        [0.02, 0.01, 0.003],
        [0.03, 0.02, 0.004],
    ]
    for p in seq:
        out = rt.update(eef_pos=p, gripper_value=0.9)
        assert 0.0 <= out.geometry_risk <= 1.0
        assert 0.0 <= out.transit_score <= 1.0
        assert isinstance(out.precise_active, bool)


def test_precise_candidate_motion_or_logic():
    cfg = PadiPhysicsConfig(precise_entry_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.8)
    out = rt.update([0.001, 0.001, 1.1], gripper_value=0.8)
    assert out.debug["precise_candidate"] is True


def test_local_interaction_can_trigger_precise_candidate():
    cfg = PadiPhysicsConfig(precise_entry_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.8)
    out = rt.update([0.001, 0.001, 0.0001], gripper_value=0.8)
    assert out.debug["local_interaction_candidate"] is True
    assert out.debug["precise_candidate"] is True


def test_curvature_path_triggers_precise():
    cfg = PadiPhysicsConfig(precise_entry_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0, 0, 0], gripper_value=0.8)
    out = rt.update([0.0001, 0.0001, 0.01], gripper_value=0.8)
    assert out.precise_active is True


def test_precise_hysteresis_exit():
    cfg = PadiPhysicsConfig(precise_entry_steps=1, precise_exit_steps=2, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.8)

    # Trigger precise candidate -> precise_active True
    out = rt.update([0.0001, 0.0001, 1.1], gripper_value=0.8)
    assert out.precise_active is True

    # Two non-candidate steps should deactivate precise_active
    out = rt.update([0.05, 0.05, 1.1], gripper_value=0.8)
    assert out.precise_active is True
    out = rt.update([0.10, 0.10, 1.1], gripper_value=0.8)
    assert out.precise_active is False


def test_transit_multiplicative_and_precise_suppression():
    cfg = PadiPhysicsConfig(precise_entry_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0, 0, 0], gripper_value=0.9)
    out1 = rt.update([0.01, 0.01, 0.0], gripper_value=0.9)
    out2 = rt.update([0.0101, 0.0101, 0.01], gripper_value=0.9)
    assert out1.transit_score >= 0.0
    assert out2.transit_score <= out1.transit_score


def test_geometry_precise_term_dominant_and_suppression_cap():
    cfg = PadiPhysicsConfig(precise_entry_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0, 0, 0], gripper_value=0.8)
    out_precise = rt.update([0.0001, 0.0001, 1.2], gripper_value=0.8)

    rt2 = PadiPhysicsAwareRuntime(cfg)
    rt2.update([0, 0, 0], gripper_value=0.8)
    out_non = rt2.update([0.03, 0.03, 0.0], gripper_value=0.8)
    assert out_precise.geometry_risk >= out_non.geometry_risk

    rt3 = PadiPhysicsAwareRuntime(PadiPhysicsConfig(precise_entry_steps=1, startup_guard_steps=10))
    rt3.update([0, 0, 0], gripper_value=0.2)
    out_sup = rt3.update([0, 0, 0], gripper_value=0.2)
    assert out_sup.geometry_risk <= 0.15


def test_reset_state():
    rt = PadiPhysicsAwareRuntime(PadiPhysicsConfig(startup_guard_steps=0))
    rt.update([0.0, 0.0, 0.0], gripper_value=0.7)
    rt.update([0.01, 0.0, 0.0], gripper_value=0.7)
    rt.update([0.02, 0.01, 0.0], gripper_value=0.7)
    rt.reset()

    s = rt.state
    assert s.last_eef_pos is None
    assert s.step_in_episode == 0
    assert s.precise_active is False
    assert s.transit_score == 0.0
    assert s.geometry_risk == 0.0


def test_gripper_score_nonzero_for_small_qpos_close_semantics():
    cfg = PadiPhysicsConfig(gripper_engage_steps=1, precise_entry_steps=10, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.001)
    out = rt.update([0.02, 0.01, 0.0], gripper_value=0.001)
    assert out.debug["gripper_engaged"] is True
    assert out.debug["gripper_score"] > 0.0
    assert out.transit_score > 0.0


def test_open_gripper_behavior():
    cfg = PadiPhysicsConfig(gripper_engage_steps=1, precise_entry_steps=10, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=2.0)
    out = rt.update([0.02, 0.01, 0.0], gripper_value=2.0)
    assert out.debug["gripper_engaged"] is False
    assert out.debug["gripper_score"] == 0.0


def test_local_thresholds_match_mapping_defaults():
    rt = PadiPhysicsAwareRuntime(PadiPhysicsConfig(precise_entry_steps=10, startup_guard_steps=0))
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    out_true = rt.update([0.003, 0.003, 0.001], gripper_value=0.0)
    assert out_true.debug["local_interaction_candidate"] is True

    rt2 = PadiPhysicsAwareRuntime(PadiPhysicsConfig(precise_entry_steps=10, startup_guard_steps=0))
    rt2.update([0.0, 0.0, 0.0], gripper_value=0.0)
    out_false = rt2.update([0.01, 0.0, 0.0], gripper_value=0.0)
    assert out_false.debug["local_interaction_candidate"] is False


def test_local_interaction_requires_gripper_engaged():
    cfg = PadiPhysicsConfig(gripper_engage_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=2.0)
    out = rt.update([0.001, 0.001, 0.0005], gripper_value=2.0)
    assert out.debug["gripper_engaged"] is False
    assert out.debug["local_interaction_candidate"] is False


def test_local_interaction_true_when_gripper_engaged_and_slow():
    cfg = PadiPhysicsConfig(gripper_engage_steps=1, startup_guard_steps=0)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    out = rt.update([0.001, 0.001, 0.0005], gripper_value=0.0)
    assert out.debug["gripper_engaged"] is True
    assert out.debug["local_interaction_candidate"] is True


def test_first_frame_appends_short_window_position():
    rt = PadiPhysicsAwareRuntime()
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    assert len(rt.state.short_window_positions) == 1
    assert rt.state.last_eef_pos is not None


def test_d_z_ratio_test():
    rt = PadiPhysicsAwareRuntime(PadiPhysicsConfig(startup_guard_steps=0, precise_entry_steps=1))
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    out = rt.update([0.001, 0.0, 0.01], gripper_value=0.0)
    assert out.debug["d_z"] > 1.0


def test_curvature_window_test():
    rt = PadiPhysicsAwareRuntime(PadiPhysicsConfig(startup_guard_steps=0, short_window_size=10))
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    rt.update([0.01, 0.0, 0.0], gripper_value=0.0)
    rt.update([0.01, 0.01, 0.0], gripper_value=0.0)
    out = rt.update([0.02, 0.01, 0.0], gripper_value=0.0)
    assert out.debug["curvature"] > 0.0


def test_precise_candidate_uses_u_s_u_xy_test():
    cfg = PadiPhysicsConfig(startup_guard_steps=0, precise_entry_steps=1)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    out = rt.update([0.001, 0.0, 0.01], gripper_value=0.0)
    assert "u_s" in out.debug and "u_xy" in out.debug
    assert out.debug["precise_candidate"] in [True, False]


def test_geometry_can_reach_one_like_original_test():
    cfg = PadiPhysicsConfig(startup_guard_steps=0, precise_entry_steps=1)
    rt = PadiPhysicsAwareRuntime(cfg)
    rt.update([0.0, 0.0, 0.0], gripper_value=0.0)
    out = rt.update([0.0001, 0.0, 0.2], gripper_value=0.0)
    assert out.geometry_risk >= 0.9


def test_defaults_align_source_of_truth():
    cfg = PadiPhysicsConfig()
    assert cfg.precise_ratio_thresh == 1.3
    assert cfg.precise_speed_thresh == 0.85
    assert cfg.precise_total_speed_thresh == 0.75
    assert cfg.precise_curvature_thresh == 0.12
    assert cfg.precise_entry_steps == 1
    assert cfg.precise_exit_steps == 4
    assert cfg.local_total_speed_thresh == 0.014
    assert cfg.local_xy_speed_thresh == 0.0075
    assert cfg.local_z_speed_thresh == 0.013
    assert cfg.gripper_close_score_thresh == 0.45
    assert cfg.gripper_open_score_thresh == 0.20


def test_first_frame_and_reset_behavior_source_truth():
    rt = PadiPhysicsAwareRuntime()
    out = rt.update([0, 0, 0], gripper_value=0.0)
    assert out.transit_score == 0.0
    assert out.geometry_risk == 0.0
    rt.reset()
    assert rt.state.last_eef_pos is None


def test_final_oft_calibrated_defaults():
    cfg = PadiPhysicsConfig()
    assert cfg.geometry_curve_offset == 0.015
    assert cfg.geometry_slow_total_base == 0.018
    assert cfg.local_total_speed_thresh == 0.014


def test_overlay_helper_returns_same_shape_dtype():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    out = PadiSignalOutput(geometry_risk=0.5, precise_active=True, transit_score=0.3, debug={})
    over = overlay_padi_scores_on_frame(frame, out, step_idx=3)
    assert over.shape == frame.shape
    assert over.dtype == frame.dtype == np.uint8
    assert not np.array_equal(over, frame)


def test_overlay_helper_handles_none_signal():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    over = overlay_padi_scores_on_frame(frame, None, step_idx=0)
    assert over.shape == frame.shape
    assert over.dtype == np.uint8
    assert not np.array_equal(over, frame)


def test_overlay_helper_handles_startup_debug():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    out = PadiSignalOutput(
        geometry_risk=0.1125,
        precise_active=False,
        transit_score=0.0,
        debug={"startup_guard_active": True, "local_interaction_candidate": True},
    )
    over = overlay_padi_scores_on_frame(frame, out, step_idx=1)
    assert over.shape == frame.shape
    assert over.dtype == np.uint8
    assert not np.array_equal(over, frame)


def test_overlay_helper_handles_precise_debug():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    out = PadiSignalOutput(
        geometry_risk=0.9,
        precise_active=True,
        transit_score=0.1,
        debug={"startup_guard_active": False, "local_interaction_candidate": True},
    )
    over = overlay_padi_scores_on_frame(frame, out, step_idx=10)
    assert over.shape == frame.shape
    assert over.dtype == np.uint8
    assert not np.array_equal(over, frame)


def test_overlay_helper_supports_positions():
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    out = PadiSignalOutput(geometry_risk=0.3, precise_active=False, transit_score=0.5, debug={})
    for pos in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        over = overlay_padi_scores_on_frame(frame, out, step_idx=0, position=pos)
        assert over.shape == frame.shape
        assert over.dtype == np.uint8


def test_runtime_runs_with_final_default_profile():
    rt = PadiPhysicsAwareRuntime(PadiPhysicsConfig())
    seq = [[0.0, 0.0, 0.0], [0.004, 0.003, 0.001], [0.007, 0.005, 0.0015]]
    out = None
    for p in seq:
        out = rt.update(p, gripper_value=0.0)
    assert out is not None
    assert 0.0 <= out.geometry_risk <= 1.0
