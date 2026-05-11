# PADI Physics-Aware Signal Mapping（PADI-VLA）

## 1. 范围说明

本 mapping **仅覆盖** PADI 运行时 physics-aware 信号：

- `geometry_risk`
- `precise_active`
- `transit_score`

本文档明确**排除**：

- FastV
- GSDR
- action memory
- goal prior
- transformer / modeling forward 改动
- token pruning
- OFT 集成

说明：虽然这些模块在同一代码文件中共存，本文仅抽取与三个 physics-aware 信号的计算、状态更新、配置读取、telemetry 输出直接相关的路径。

---

## 2. 源文件清单

| 文件 | 作用 | 相关符号 | 是否核心 | 备注 |
|---|---|---|---|---|
| `src/openvla/prismatic/extern/hf/padi_runtime.py` | 定义三信号核心计算函数与基础启发式 | `compute_transit_score`, `compute_geometry_risk`, `compute_precise_active`, `compute_precise_active_with_startup_guard`, `apply_stationary_veto`, `compute_proprio_features` | 是 | 三个信号的主要公式在此实现 |
| `src/openvla/prismatic/extern/hf/modeling_prismatic.py` | 在运行时调用 `padi_runtime`，更新 FSM/state，并写入 payload/telemetry | `PadiFSMState`, `PadiPhysicsState`, `update_precise_state`, `update_padi_fsm`, `update_padi_step` | 是 | 三个信号的实际更新顺序、reset、存储、消费入口 |
| `src/openvla/prismatic/extern/hf/configuration_prismatic.py` | 声明 geometry dynamic keep 相关配置项（间接影响 geometry_risk 的消费与平滑可视化） | `use_geometry_dynamic_keep`, `geometry_keep_*` | 间接 | 不改变三信号原始公式，但影响 geometry_risk 下游动态 keep |
| `tools/visualize_attention_episode.py` | 从 `last_inference_stats` 读取 telemetry 并可视化 | `telemetry = getattr(model, "last_inference_stats", {})` | 间接 | 信号相关字段可被该工具读取展示 |

---

## 3. 信号定义

### geometry_risk

- **计算位置**：`padi_runtime.compute_geometry_risk(...)`。
- **输入变量**：`precise_active`, `local_interaction_candidate`, `d_z`, `curvature`, `total_speed`, `xy_speed`, `recent_mean_speed`。
- **依赖 state**：`state.precise_active`、`state.gripper_engaged`（通过 `local_interaction_candidate` 间接影响）、最近速度窗口统计。
- **公式（启发式）**：
  - `precise_term = 1.0 if precise_active else 0.0`
  - `local_term = 1.0 if local_interaction_candidate else 0.0`
  - `dz_term = clamp01((d_z - 0.9)/0.8)`
  - `curve_term = clamp01((curvature - 0.08)/0.25)`
  - `slow_term = clamp01((slow_total + slow_xy + slow_recent)/3)`，其中 `slow_total/xy/recent` 分别由低速区间映射得到
  - `base_risk = clamp01(0.55*precise_term + 0.20*local_term + 0.10*dz_term + 0.10*curve_term + 0.05*slow_term)`
  - 若 `not precise_active`，再乘 `0.45`。
- **阈值**：见上式中的 `0.9/0.8/0.08/0.25/0.012/0.010/0.008/0.006/0.010/0.008`。
- **clamp/smoothing**：原始风险多处 `clamp01`，最终输出 `clamp01`；在 geometry dynamic keep 分支另有 `update_geometry_risk_smooth` 做非对称 EMA（用于下游 keep，不改变 `physics_state.geometry_risk` 原始值）。
- **默认值**：episode 首帧（`last_eef_pos is None`）为 `0.0`。
- **reset 行为**：`PadiPhysicsState.geometry_risk` 在 `reset_padi_state()` 后默认为 `0.0`；首帧也被显式写回 `0.0`。
- **存储位置**：`PadiPhysicsState.geometry_risk`。
- **消费位置**：
  - runtime payload (`update_padi_fsm` 返回字典)
  - `self.padi_physics_state`（供后续步骤/模块读取）
  - geometry dynamic keep 读取 `self.padi_physics_state.geometry_risk` 作为 previous-step source。

### precise_active

- **计算位置**：
  1) `update_precise_state()` 中构造 `precise_candidate_motion`
  2) `compute_precise_active_with_startup_guard(...)`
  3) `apply_stationary_veto(...)` 二次抑制。
- **输入变量**：`d_z`, `u_xy`, `u_s`, `curvature`, `local_interaction_candidate`, `recent_mean_speed`, `total_speed`, `step_in_episode`，以及 entry/exit 计数器。
- **依赖 state**：`state.precise_active`、`state.precise_entry_counter`、`state.precise_exit_counter`、startup/stationary veto 相关计数器与标志。
- **触发条件**：
  - `precise_candidate_motion = (d_z > ratio_thresh and u_xy < speed_thresh) or (u_s < total_speed_thresh and curvature > curvature_thresh)`。
  - `precise_candidate = precise_candidate_motion or local_interaction_candidate`。
  - 命中 candidate 连续达到 `precise_entry_steps` 进入；连续非 candidate 达到 `precise_exit_steps` 退出（hysteresis）。
  - startup guard 与 stationary veto 可阻止/清零早期误触发。
- **阈值**：默认阈值常量在 `padi_runtime.py` 顶部（如 ratio/speed/entry/exit/startup/veto 等）并在模型初始化读取。
- **类型**：布尔型主信号（`bool`），并伴随派生标志（`precise_candidate`, `startup_precise_suppressed`, `precise_suppressed_by_stationary_veto`）。
- **reset 行为**：`PadiFSMState.precise_active=False`，相关计数器清零；首帧保持默认。
- **存储位置**：
  - 主存储：`PadiFSMState.precise_active`
  - 镜像输出：`PadiPhysicsState.precise_active`
- **消费位置**：
  - `compute_transit_score`（`precise_active` 时乘 `0.05` 惩罚）
  - `compute_geometry_risk`
  - payload/telemetry
  - 其他下游（例如 gating 逻辑）读取 `physics_state.precise_active`。

### transit_score

- **计算位置**：`padi_runtime.compute_transit_score(...)`，在 `update_padi_fsm` 中调用。
- **输入变量**：`precise_active`, `gripper_engaged`, `gripper_stably_closed`, `holding_confidence`, `recent_mean_speed`, `recent_mean_disp`, `total_speed`, `xy_speed`。
- **依赖 state**：gripper 相关计数器/分数、`precise_active`、recent speed/disp 窗口统计。
- **公式（启发式）**：
  - gripper 分数：`0/0.50/0.85 + 0.15*clamp01((holding_confidence-0.50)/0.35)`
  - motion 分数：`motion_linear = clamp01(0.40*speed + 0.35*disp + 0.15*total + 0.10*xy)` 后 `motion_score = clamp01(motion_linear**1.35)`
  - precise 抑制：`not_precise_score = 0.05 if precise_active else 1.0`
  - 输出：`clamp01(gripper_score * motion_score * not_precise_score)`。
- **阈值**：速度映射阈值 `0.0075/0.0055/0.0070/0.0050/0.0045`，holding 归一化阈值 `0.50/0.35`。
- **smoothing 行为**：函数内无跨步平滑；仅即时计算并 clamp。
- **reset 行为**：首帧与 reset 后默认 `0.0`。
- **存储位置**：`PadiPhysicsState.transit_score`。
- **消费位置**：payload/telemetry，下游 gating 与统计逻辑读取。

---

## 4. 输入依赖 mapping

| 信号 | 必需输入 | 可选输入 | 来自环境观测？ | 来自动作？ | 来自模型内部？ |
|---|---|---|---|---|---|
| `geometry_risk` | `eef_pos`（经 `compute_proprio_features` 得 `d_z/curvature/total/xy`）、`gripper_value`（影响 `local_interaction_candidate`） 、`state.precise_active` | `step_idx`（仅 debug） | 是（robot observation） | 否（`gripper_cmd` 被 `del`） | 否 |
| `precise_active` | `eef_pos` 派生特征、`gripper_value`、`step_in_episode`、内部计数器 | `gripper_cmd`（未使用） | 是 | 否 | 否 |
| `transit_score` | `gripper_value` 派生分数、`eef_pos` 派生速度/位移、`precise_active` | 无 | 是 | 否 | 否 |

输入来源标注：

- `robot observation`：`eef_pos`, `gripper_value`
- `previous robot observation`：`state.last_eef_pos`, short/recent windows
- `executed action`：本三信号主路径未使用
- `gripper command`：`update_padi_fsm` 中 `del gripper_cmd`，未进入三信号计算
- `step index`：用于 payload/debug，不进入公式
- `model config`：阈值与窗口参数
- `internal PADI state`：计数器、窗口、hysteresis 状态
- `attention/model internals`：三信号本身不依赖

**结论（可复用性）**：三信号核心计算是 model-independent（仅依赖机器人时序观测 + PADI 内部状态机），可在 OFT runtime 复用而不改模型 forward。

---

## 5. 状态依赖 mapping

| State class / object | 字段 | 被哪个信号使用 | 初始值 | 更新规则 | reset 时机 |
|---|---|---|---|---|---|
| `PadiFSMState` | `precise_active` | 三者（直接/间接） | `False` | 由 `compute_precise_active_with_startup_guard` + `apply_stationary_veto` 更新 | episode reset |
| `PadiFSMState` | `precise_entry_counter`, `precise_exit_counter` | `precise_active` | `0` | candidate/非candidate 递增或清零 | episode reset |
| `PadiFSMState` | `last_eef_pos` | 三者 | `None` | 每步末尾写入当前 `eef_pos` | episode reset |
| `PadiFSMState` | `short_window_positions`, `recent_total_speed_window`, `recent_step_disp_window` | 三者 | `deque(...)` | 每步 append，用于速度/曲率统计 | episode reset |
| `PadiFSMState` | `ema_total_speed`, `ema_xy_speed`, `ema_z_speed` | `precise_active`, `geometry_risk`（通过特征） | `0.0` | `compute_proprio_features` 输出回写 | episode reset |
| `PadiFSMState` | `gripper_*` counters/scores & `gripper_engaged` | `transit_score`, `precise_active`, `geometry_risk` | `False/0.0/0` | `update_gripper_engaged` 更新 | episode reset |
| `PadiFSMState` | `startup_guard_active`, `motion_release_counter`, `startup_precise_counter` | `precise_active` | `False/0/0` | startup guard 逻辑更新 | episode reset |
| `PadiFSMState` | `stationary_veto_counter`, `precise_suppressed_by_stationary_veto` | `precise_active`（并影响 `geometry_risk` 上限） | `0/False` | stationary veto 更新 | episode reset |
| `PadiPhysicsState` | `precise_active` | 输出镜像 | `False` | `update_padi_fsm` 尾部同步 | episode reset |
| `PadiPhysicsState` | `transit_score` | 输出 | `0.0` | 每步由 `compute_transit_score` 写入 | episode reset |
| `PadiPhysicsState` | `geometry_risk` | 输出 | `0.0` | 每步由 `compute_geometry_risk` 写入 | episode reset |

---

## 6. 更新顺序

基于 `update_padi_fsm` 实际源码，顺序如下：

1. 递增 `step_in_episode`；第 1 步开启 startup guard。  
2. 用当前 `gripper_value` 更新 gripper 分数/计数器/`gripper_engaged`。  
3. 若首帧（`last_eef_pos is None`）：写默认 `transit_score=0.0`, `geometry_risk=0.0`, `precise_active` 当前值并返回。  
4. 计算 proprio 特征（速度、EMA、`d_z`、`curvature` 等）。  
5. 更新 recent 窗口并算 `recent_mean_speed` / `recent_mean_disp`。  
6. 计算 `local_interaction_candidate`。  
7. 更新 startup guard 状态。  
8. 更新 `precise_active`（hysteresis + startup suppression）。  
9. 应用 stationary veto（可能强制 `precise_active=False`）。  
10. 更新 recent precise exit latch。  
11. 计算 `transit_score`。  
12. 计算 `geometry_risk`。  
13. 若 startup/stationary suppression 触发，则 `geometry_risk = min(geometry_risk, 0.15)`。  
14. 生成 payload。  
15. 同步写入 `PadiPhysicsState`。  
16. `update_padi_step` 封装返回并附加 debug 字段。  

---

## 7. 配置字段

| Config 字段 | 默认值 | 源文件 | 影响信号 | OFT runtime-only 是否必需 | 备注 |
|---|---|---|---|---|---|
| `padi_precise_ratio_thresh` | 见模型初始化默认 | `modeling_prismatic.py` | `precise_active` | 是 | `d_z/u_xy` 触发门限 |
| `padi_precise_speed_thresh` | 同上 | 同上 | `precise_active` | 是 | 同上 |
| `padi_precise_total_speed_thresh` | 同上 | 同上 | `precise_active` | 是 | `u_s` 条件 |
| `padi_precise_curvature_thresh` | 同上 | 同上 | `precise_active` | 是 | `curvature` 条件 |
| `padi_precise_entry_steps` / `padi_precise_exit_steps` | 同上 | 同上 | `precise_active` | 是 | hysteresis |
| `padi_local_total_speed_thresh` / `xy` / `z` | 同上 | 同上 | `precise_active`, `geometry_risk` | 是 | 决定 `local_interaction_candidate` |
| `padi_gripper_close_score_thresh` / `engage_steps` / `open_*` | 同上 | 同上 | `transit_score`（并间接影响其他） | 是 | gripper 状态提取 |
| `padi_startup_*` | 同上 | 同上 | `precise_active`（并间接影响 `geometry_risk` cap） | 是 | startup guard |
| `padi_early_stationary_veto_*` | 同上 | 同上 | `precise_active`（并间接影响 `geometry_risk` cap） | 是 | 早期静止抑制 |
| `use_geometry_dynamic_keep` + `geometry_keep_*` | `False` + 表中默认 | `configuration_prismatic.py` | 不改原始三信号；消费 `geometry_risk` 做 keep | 否（信号本身） | 属于下游动态 keep |

---

## 8. Telemetry 字段

| Telemetry 字段 | 源码位置 | 类型 | 含义 | 是否在 PADI-OFT 中保留 |
|---|---|---|---|---|
| `precise_active` | `update_padi_fsm` payload | bool | 当前是否处于精细交互态 | 建议保留 |
| `transit_score` | `update_padi_fsm` payload | float | 当前运输倾向分数 | 建议保留 |
| `geometry_risk` | `update_padi_fsm` payload | float | 当前几何约束风险分数 | 建议保留 |
| `precise_candidate` | `update_padi_fsm` payload | bool | precise 候选触发 | 建议保留（debug） |
| `startup_*`, `stationary_veto_*` | `update_padi_fsm` payload | bool/int | 早期抑制诊断 | 建议保留（debug） |
| `recent_mean_speed`, `recent_mean_disp`, `d_z`, `curvature` | `update_padi_fsm` payload | float | 上游特征诊断 | 建议保留（debug） |
| `geometry_risk_raw/smooth/...` | `fastv_config`/`last_inference_stats` 分支 | float/list | geometry dynamic keep 下游诊断 | 可选（仅启用 keep 时） |

---

## 9. 最小可复用 API 建议

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class PadiPhysicsConfig:
    precise_ratio_thresh: float
    precise_speed_thresh: float
    precise_total_speed_thresh: float
    precise_curvature_thresh: float
    precise_entry_steps: int
    precise_exit_steps: int
    local_total_speed_thresh: float
    local_xy_speed_thresh: float
    local_z_speed_thresh: float
    gripper_close_score_thresh: float
    gripper_engage_steps: int
    gripper_open_score_thresh: float
    gripper_open_steps: int
    startup_guard_steps: int
    startup_guard_min_release_steps: int
    startup_guard_release_mean_speed: float
    startup_guard_release_total_speed: float
    startup_guard_motion_confirm_steps: int
    startup_precise_min_speed: float
    startup_precise_min_total_speed: float
    startup_precise_confirm_steps: int
    early_stationary_veto_steps: int
    stationary_veto_mean_speed: float
    stationary_veto_total_speed: float
    stationary_veto_confirm_steps: int

@dataclass
class PadiPhysicsState:
    precise_active: bool = False
    transit_score: float = 0.0
    geometry_risk: float = 0.0
    # + counters/windows needed for hysteresis and feature extraction

@dataclass
class PadiSignalOutput:
    geometry_risk: float
    precise_active: bool
    transit_score: float
    debug: Dict[str, Any]

class PadiPhysicsAwareRuntime:
    def reset(self) -> None: ...

    def update(
        self,
        eef_pos: np.ndarray,
        gripper_value: Optional[float] = None,
        gripper_cmd: Optional[float] = None,
        step_idx: Optional[int] = None,
        action: Optional[np.ndarray] = None,
        obs: Optional[Dict[str, Any]] = None,
    ) -> PadiSignalOutput: ...
```

约束建议：`action`, `attention`, `model forward internals` 不能成为这三个信号的必需输入。

---

## 10. 直接复用 / 需要改写分类

| 组件 | 源码位置 | 分类 | 原因 |
|---|---|---|---|
| `compute_proprio_features` | `padi_runtime.py` | 直接复用 | 纯观测+状态计算，与模型无关 |
| `compute_precise_active` / `compute_precise_active_with_startup_guard` / `apply_stationary_veto` | `padi_runtime.py` | 直接复用 | 纯规则状态机逻辑 |
| `compute_transit_score` | `padi_runtime.py` | 直接复用 | 纯启发式分数函数 |
| `compute_geometry_risk` | `padi_runtime.py` | 直接复用 | 纯启发式分数函数 |
| `update_padi_fsm` 中与 action memory/goal prior/phase 兼容段 | `modeling_prismatic.py` | runtime-only 迁移中排除 | 非本任务范围 |
| `update_padi_step` 封装接口 | `modeling_prismatic.py` | 小幅适配 | OFT eval loop 的入参/返回格式可能不同 |
| geometry dynamic keep | `modeling_prismatic.py` + `configuration_prismatic.py` | runtime-only 迁移中排除（本阶段） | 属于 pruning/keep 下游消费，不是三信号核心计算 |

---

## 11. 风险与不确定点

1. `padi_*` 阈值参数在 `modeling_prismatic.py` 初始化路径中读取，但默认值分散在 runtime 常量与模型构造逻辑，迁移时需核对“最终生效值”来源。  
2. `gripper_cmd` 已在 `update_padi_fsm` 中 `del`，但接口仍保留该参数；迁移时若删参可能影响调用兼容。  
3. `geometry_risk` 同时有“原始值”和“geometry_dynamic_keep 平滑值”两套路径，必须区分：本文三信号以 `PadiPhysicsState.geometry_risk` 为准。  
4. telemetry 字段在不同输出层（payload / `fastv_config` / `last_inference_stats`）命名存在重叠，迁移时需固定单一 schema。  
5. `phase` / `carry_*` 仍被计算用于兼容，但与本三信号核心目标不完全一致，迁移时建议标为非必需。

---

## 12. 给下一个 Codex 的迁移 checklist

- [ ] 在 PADI-OFT 新建 runtime 模块文件（例如 `padi_physics_runtime.py`）与 state/config dataclass。  
- [ ] 从 PADI-VLA 复制并适配（不改语义）以下逻辑：`compute_proprio_features`、`compute_precise_active*`、`apply_stationary_veto`、`compute_transit_score`、`compute_geometry_risk`。  
- [ ] 在 OFT eval loop 每 step 提供输入：`eef_pos`、`gripper_value`、`step_idx`（`gripper_cmd`可选兼容）。  
- [ ] 输出并记录 telemetry：至少 `geometry_risk`, `precise_active`, `transit_score`，建议附带关键 debug 字段。  
- [ ] 编写测试：
  - [ ] 首帧默认值测试（`transit_score=0, geometry_risk=0`）
  - [ ] precise hysteresis 进入/退出测试
  - [ ] startup guard 与 stationary veto 抑制测试
  - [ ] transit/geometry 分数范围 clamp `[0,1]` 测试
- [ ] 保持 invariant：
  - [ ] physics-aware runtime 必须 model-independent
  - [ ] 不允许修改 action
  - [ ] 不允许依赖 attention maps
  - [ ] 不允许依赖 FastV、GSDR、action memory
  - [ ] 模块关闭时与 upstream OpenVLA-OFT 行为一致
  - [ ] 模块开启时本阶段只输出 telemetry signals

---

### 验收标准

1. 只新增或修改 `docs/PADI_PHYSICS_AWARE_SIGNAL_MAPPING.md`。  
2. 文档列出 `geometry_risk`、`precise_active`、`transit_score` 的精确源码路径与符号。  
3. 文档识别所有必需 runtime 输入。  
4. 文档说明每个输入是否 model-independent。  
5. 文档提出最小可复用的 PADI-OFT API。  
6. 文档明确排除 FastV、GSDR、action memory、goal prior、transformer 改动和 pruning。  
7. 不发生任何功能性代码修改。
