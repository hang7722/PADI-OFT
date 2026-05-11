from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class PadiPhysicsConfig:
    # Defaults are conservative and mapping-aligned; exact legacy defaults may vary by PADI-VLA branch.
    precise_ratio_thresh: float = 1.3
    precise_speed_thresh: float = 0.85
    precise_total_speed_thresh: float = 0.75
    precise_curvature_thresh: float = 0.12
    precise_entry_steps: int = 1
    precise_exit_steps: int = 4

    local_total_speed_thresh: float = 0.006
    local_xy_speed_thresh: float = 0.0045
    local_z_speed_thresh: float = 0.0025

    gripper_close_score_thresh: float = 0.45
    gripper_engage_steps: int = 2
    gripper_open_score_thresh: float = 0.20
    gripper_open_steps: int = 2
    carry_entry_close_score_min: float = 0.55

    startup_guard_steps: int = 6
    startup_guard_min_release_steps: int = 6
    startup_guard_release_mean_speed: float = 0.0035
    startup_guard_release_total_speed: float = 0.0040
    startup_guard_motion_confirm_steps: int = 2

    startup_precise_min_speed: float = 0.0045
    startup_precise_min_total_speed: float = 0.0050
    startup_precise_confirm_steps: int = 2

    early_stationary_veto_steps: int = 30
    stationary_veto_mean_speed: float = 0.0008
    stationary_veto_total_speed: float = 0.0010
    stationary_veto_confirm_steps: int = 3

    ema_decay: float = 0.9
    short_window_size: int = 8
    geometry_dz_offset: float = 0.9
    geometry_dz_scale: float = 0.8
    geometry_curve_offset: float = 0.08
    geometry_curve_scale: float = 0.25
    geometry_slow_total_base: float = 0.012
    geometry_slow_total_scale: float = 0.010
    geometry_slow_xy_base: float = 0.008
    geometry_slow_xy_scale: float = 0.006
    geometry_slow_recent_base: float = 0.010
    geometry_slow_recent_scale: float = 0.008
    geometry_precise_inactive_multiplier: float = 0.45
    geometry_suppression_cap: float = 0.15

    @classmethod
    def oft_calibrated(cls) -> "PadiPhysicsConfig":
        cfg = cls()
        cfg.local_total_speed_thresh = 0.009
        cfg.local_xy_speed_thresh = 0.006
        cfg.local_z_speed_thresh = 0.004
        cfg.geometry_curve_offset = 0.05
        cfg.geometry_curve_scale = 0.25
        cfg.geometry_slow_total_base = 0.014
        cfg.geometry_slow_total_scale = 0.010
        cfg.geometry_slow_xy_base = 0.010
        cfg.geometry_slow_xy_scale = 0.006
        cfg.geometry_slow_recent_base = 0.012
        cfg.geometry_slow_recent_scale = 0.008
        return cfg

    @classmethod
    def oft_calibrated_v2(cls) -> "PadiPhysicsConfig":
        cfg = cls()
        cfg.local_total_speed_thresh = 0.012
        cfg.local_xy_speed_thresh = 0.0065
        cfg.local_z_speed_thresh = 0.011
        cfg.geometry_curve_offset = 0.025
        cfg.geometry_curve_scale = 0.20
        cfg.geometry_slow_total_base = 0.016
        cfg.geometry_slow_total_scale = 0.010
        cfg.geometry_slow_xy_base = 0.012
        cfg.geometry_slow_xy_scale = 0.006
        cfg.geometry_slow_recent_base = 0.014
        cfg.geometry_slow_recent_scale = 0.008
        return cfg

    @classmethod
    def oft_calibrated_v3(cls) -> "PadiPhysicsConfig":
        cfg = cls()
        cfg.local_total_speed_thresh = 0.014
        cfg.local_xy_speed_thresh = 0.0075
        cfg.local_z_speed_thresh = 0.013
        cfg.geometry_curve_offset = 0.015
        cfg.geometry_curve_scale = 0.18
        cfg.geometry_slow_total_base = 0.018
        cfg.geometry_slow_total_scale = 0.011
        cfg.geometry_slow_xy_base = 0.013
        cfg.geometry_slow_xy_scale = 0.007
        cfg.geometry_slow_recent_base = 0.016
        cfg.geometry_slow_recent_scale = 0.009
        return cfg


@dataclass
class PadiPhysicsState:
    precise_active: bool = False
    transit_score: float = 0.0
    geometry_risk: float = 0.0
    last_eef_pos: Optional[np.ndarray] = None
    step_in_episode: int = 0

    precise_entry_counter: int = 0
    precise_exit_counter: int = 0
    short_window_positions: list = field(default_factory=list)
    recent_total_speed_window: list = field(default_factory=list)
    recent_step_disp_window: list = field(default_factory=list)

    ema_total_speed: float = 0.0
    ema_xy_speed: float = 0.0
    ema_z_speed: float = 0.0

    gripper_engaged: bool = False
    gripper_stably_closed: bool = False
    holding_confidence: float = 0.0

    startup_guard_active: bool = True
    motion_release_counter: int = 0
    startup_precise_counter: int = 0
    stationary_veto_counter: int = 0
    precise_suppressed_by_stationary_veto: bool = False

    gripper_close_counter: int = 0
    gripper_open_counter: int = 0
    gripper_close_score: float = 0.0
    gripper_open_score: float = 0.0
    last_gripper_value: float = 0.0


@dataclass
class PadiSignalOutput:
    geometry_risk: float
    precise_active: bool
    transit_score: float
    debug: Dict[str, Any]


class PadiPhysicsAwareRuntime:
    def __init__(self, config: Optional[PadiPhysicsConfig] = None) -> None:
        self.config = config or PadiPhysicsConfig()
        self.state = PadiPhysicsState()

    def reset(self) -> None:
        self.state = PadiPhysicsState()

    def _validate_eef_pos(self, eef_pos: Any) -> np.ndarray:
        arr = np.asarray(eef_pos, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 3:
            raise ValueError(f"eef_pos must have at least 3 elements, got shape={arr.shape}")
        return arr[:3]


    def _compute_proprio_features(self, current_pos: np.ndarray, last_pos: np.ndarray):
        s, c = self.state, self.config
        eps = 1e-6
        dp = current_pos - last_pos
        total_speed = float(np.linalg.norm(dp))
        xy_speed = float(np.linalg.norm(dp[:2]))
        z_speed = float(abs(dp[2]))

        ema_total = c.ema_decay * s.ema_total_speed + (1.0 - c.ema_decay) * total_speed
        ema_xy = c.ema_decay * s.ema_xy_speed + (1.0 - c.ema_decay) * xy_speed
        ema_z = c.ema_decay * s.ema_z_speed + (1.0 - c.ema_decay) * z_speed

        u_s = float(total_speed / (ema_total + eps))
        u_xy = float(xy_speed / (ema_xy + eps))
        d_z = float(z_speed / (xy_speed + eps))

        positions = list(s.short_window_positions) + [current_pos.copy()]
        if len(positions) >= 2:
            path_len = 0.0
            for i in range(1, len(positions)):
                path_len += float(np.linalg.norm(positions[i] - positions[i - 1]))
            net_disp = float(np.linalg.norm(positions[-1] - positions[0]))
            curvature = float(1.0 - net_disp / (path_len + eps))
        else:
            curvature = 0.0

        return total_speed, xy_speed, z_speed, u_s, u_xy, d_z, curvature, ema_total, ema_xy, ema_z
    def update(self, eef_pos, gripper_value=None, gripper_cmd=None, step_idx=None, action=None, obs=None) -> PadiSignalOutput:
        s, c = self.state, self.config
        s.step_in_episode += 1
        p = self._validate_eef_pos(eef_pos)

        gv = 0.0 if gripper_value is None else float(gripper_value)
        # Align with PADI-VLA mapping: close_score is derived from raw qpos via (1 - gripper_value).
        s.last_gripper_value = gv
        s.gripper_close_score = float(np.clip(1.0 - gv, 0.0, 1.5))
        s.gripper_open_score = float(np.clip(1.0 - s.gripper_close_score, 0.0, 1.0))

        if s.gripper_close_score > c.gripper_close_score_thresh:
            s.gripper_close_counter += 1
        else:
            s.gripper_close_counter = 0
        if s.gripper_open_score > c.gripper_open_score_thresh:
            s.gripper_open_counter += 1
        else:
            s.gripper_open_counter = 0

        s.gripper_engaged = s.gripper_close_counter >= c.gripper_engage_steps
        s.gripper_stably_closed = s.gripper_close_counter >= c.gripper_engage_steps
        s.holding_confidence = _clamp01((s.gripper_close_score - c.carry_entry_close_score_min) / 0.5)

        if s.last_eef_pos is None:
            s.short_window_positions.append(p.copy())
            if len(s.short_window_positions) > c.short_window_size:
                s.short_window_positions.pop(0)
            s.last_eef_pos = p.copy()
            debug = self._build_debug(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, 1.0)
            return PadiSignalOutput(0.0, s.precise_active, 0.0, debug)

        total_speed, xy_speed, z_speed, u_s, u_xy, d_z, curvature, ema_total, ema_xy, ema_z = self._compute_proprio_features(p, s.last_eef_pos)
        s.ema_total_speed = ema_total
        s.ema_xy_speed = ema_xy
        s.ema_z_speed = ema_z

        s.recent_total_speed_window.append(total_speed)
        s.recent_step_disp_window.append(total_speed)
        s.short_window_positions.append(p.copy())
        for w in (s.recent_total_speed_window, s.recent_step_disp_window, s.short_window_positions):
            if len(w) > c.short_window_size:
                w.pop(0)

        recent_mean_speed = float(np.mean(s.recent_total_speed_window)) if s.recent_total_speed_window else 0.0
        recent_mean_disp = float(np.mean(s.recent_step_disp_window)) if s.recent_step_disp_window else 0.0

        local_interaction_candidate = bool(
            s.gripper_engaged
            and total_speed < c.local_total_speed_thresh
            and xy_speed < c.local_xy_speed_thresh
            and z_speed < c.local_z_speed_thresh
        )

        motion_ok = recent_mean_speed >= c.startup_guard_release_mean_speed and total_speed >= c.startup_guard_release_total_speed
        s.motion_release_counter = s.motion_release_counter + 1 if motion_ok else 0
        if s.step_in_episode > c.startup_guard_steps or (s.step_in_episode >= c.startup_guard_min_release_steps and s.motion_release_counter >= c.startup_guard_motion_confirm_steps):
            s.startup_guard_active = False

        precise_candidate_motion = ((d_z > c.precise_ratio_thresh and u_xy < c.precise_speed_thresh)
            or (u_s < c.precise_total_speed_thresh and curvature > c.precise_curvature_thresh))
        precise_candidate = precise_candidate_motion or local_interaction_candidate

        startup_precise_suppressed = False
        if s.startup_guard_active:
            weak_motion = recent_mean_speed < c.startup_precise_min_speed or total_speed < c.startup_precise_min_total_speed
            if weak_motion:
                startup_precise_suppressed = True
                precise_candidate = False

        if precise_candidate:
            s.precise_entry_counter += 1
            s.precise_exit_counter = 0
        else:
            s.precise_exit_counter += 1
            s.precise_entry_counter = 0

        if not s.precise_active and s.precise_entry_counter >= c.precise_entry_steps:
            s.precise_active = True
        if s.precise_active and s.precise_exit_counter >= c.precise_exit_steps:
            s.precise_active = False

        s.precise_suppressed_by_stationary_veto = False
        if s.step_in_episode <= c.early_stationary_veto_steps:
            stationary = recent_mean_speed <= c.stationary_veto_mean_speed and total_speed <= c.stationary_veto_total_speed
            s.stationary_veto_counter = s.stationary_veto_counter + 1 if stationary else 0
            if s.stationary_veto_counter >= c.stationary_veto_confirm_steps and s.precise_active:
                s.precise_active = False
                s.precise_suppressed_by_stationary_veto = True

        gripper_base = 0.0
        if s.gripper_stably_closed:
            gripper_base = 0.85
        elif s.gripper_engaged:
            gripper_base = 0.50
        gripper_score = _clamp01(gripper_base + 0.15 * _clamp01((s.holding_confidence - 0.50) / 0.35))

        speed_n = _clamp01((recent_mean_speed - 0.0075) / 0.0055)
        disp_n = _clamp01((recent_mean_disp - 0.0075) / 0.0055)
        total_n = _clamp01((total_speed - 0.0070) / 0.0055)
        xy_n = _clamp01((xy_speed - 0.0050) / 0.0045)
        motion_linear = _clamp01(0.40 * speed_n + 0.35 * disp_n + 0.15 * total_n + 0.10 * xy_n)
        motion_score = _clamp01(motion_linear**1.35)
        not_precise_score = 0.05 if s.precise_active else 1.0
        s.transit_score = _clamp01(gripper_score * motion_score * not_precise_score)

        precise_term = 1.0 if s.precise_active else 0.0
        local_term = 1.0 if local_interaction_candidate else 0.0
        dz_term = _clamp01((d_z - c.geometry_dz_offset) / c.geometry_dz_scale)
        curve_term = _clamp01((curvature - c.geometry_curve_offset) / c.geometry_curve_scale)
        slow_total = _clamp01((c.geometry_slow_total_base - total_speed) / c.geometry_slow_total_scale)
        slow_xy = _clamp01((c.geometry_slow_xy_base - xy_speed) / c.geometry_slow_xy_scale)
        slow_recent = _clamp01((c.geometry_slow_recent_base - recent_mean_speed) / c.geometry_slow_recent_scale)
        slow_term = _clamp01((slow_total + slow_xy + slow_recent) / 3.0)
        base_risk = _clamp01(0.55 * precise_term + 0.20 * local_term + 0.10 * dz_term + 0.10 * curve_term + 0.05 * slow_term)
        geometry_risk_before_precise_inactive_multiplier = base_risk
        if not s.precise_active:
            base_risk *= c.geometry_precise_inactive_multiplier
        geometry_risk_before_suppression_cap = base_risk
        geometry_suppression_applied = bool(startup_precise_suppressed or s.precise_suppressed_by_stationary_veto)
        if geometry_suppression_applied:
            base_risk = min(base_risk, c.geometry_suppression_cap)
        s.geometry_risk = _clamp01(base_risk)

        s.last_eef_pos = p.copy()
        debug = self._build_debug(total_speed, xy_speed, z_speed, d_z, curvature, recent_mean_speed, recent_mean_disp, precise_candidate, local_interaction_candidate, startup_precise_suppressed, gripper_score, motion_score, not_precise_score, precise_term, local_term, dz_term, curve_term, slow_total, slow_xy, slow_recent, slow_term, geometry_risk_before_precise_inactive_multiplier, geometry_risk_before_suppression_cap, geometry_suppression_applied, u_s, u_xy)
        return PadiSignalOutput(s.geometry_risk, bool(s.precise_active), s.transit_score, debug)

    def _build_debug(self, total_speed, xy_speed, z_speed, d_z, curvature, recent_mean_speed, recent_mean_disp, precise_candidate, local_interaction_candidate, startup_precise_suppressed=False, gripper_score=0.0, motion_score=0.0, not_precise_score=1.0, precise_term=0.0, local_term=0.0, dz_term=0.0, curve_term=0.0, slow_total=0.0, slow_xy=0.0, slow_recent=0.0, slow_term=0.0, geometry_risk_before_precise_inactive_multiplier=0.0, geometry_risk_before_suppression_cap=0.0, geometry_suppression_applied=False, u_s=0.0, u_xy=0.0):
        s = self.state
        return {
            "precise_active": bool(s.precise_active),
            "transit_score": float(s.transit_score),
            "geometry_risk": float(s.geometry_risk),
            "precise_candidate": bool(precise_candidate),
            "local_interaction_candidate": bool(local_interaction_candidate),
            "recent_mean_speed": float(recent_mean_speed),
            "recent_mean_disp": float(recent_mean_disp),
            "total_speed": float(total_speed),
            "xy_speed": float(xy_speed),
            "z_speed": float(z_speed),
            "d_z": float(d_z),
            "curvature": float(curvature),
            "u_s": float(u_s),
            "u_xy": float(u_xy),
            "ema_total_speed": float(s.ema_total_speed),
            "ema_xy_speed": float(s.ema_xy_speed),
            "ema_z_speed": float(s.ema_z_speed),
            "gripper_value": float(s.last_gripper_value),
            "gripper_close_score": float(s.gripper_close_score),
            "gripper_open_score": float(s.gripper_open_score),
            "gripper_closed_counter": int(s.gripper_close_counter),
            "gripper_open_counter": int(s.gripper_open_counter),
            "gripper_engaged": bool(s.gripper_engaged),
            "gripper_stably_closed": bool(s.gripper_stably_closed),
            "holding_confidence": float(s.holding_confidence),
            "gripper_score": float(gripper_score),
            "motion_score": float(motion_score),
            "not_precise_score": float(not_precise_score),
            "precise_term": float(precise_term),
            "local_term": float(local_term),
            "dz_term": float(dz_term),
            "curve_term": float(curve_term),
            "slow_total": float(slow_total),
            "slow_xy": float(slow_xy),
            "slow_recent": float(slow_recent),
            "slow_term": float(slow_term),
            "geometry_risk_before_precise_inactive_multiplier": float(geometry_risk_before_precise_inactive_multiplier),
            "geometry_risk_before_suppression_cap": float(geometry_risk_before_suppression_cap),
            "geometry_suppression_applied": bool(geometry_suppression_applied),
            "startup_guard_active": bool(s.startup_guard_active),
            "startup_precise_suppressed": bool(startup_precise_suppressed),
            "stationary_veto_counter": int(s.stationary_veto_counter),
            "precise_suppressed_by_stationary_veto": bool(s.precise_suppressed_by_stationary_veto),
            "step_in_episode": int(s.step_in_episode),
        }
