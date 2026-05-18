"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from padi_oft.runtime.physics_runtime import PadiPhysicsAwareRuntime, PadiPhysicsConfig
from padi_oft.runtime.gsdr_controller import PadiGSDRConfig, PadiGSDRController

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from padi_oft.runtime.video_overlay import overlay_fastv_pruning_on_views, overlay_padi_scores_on_frame

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    use_padi_runtime: bool = False                   # Enable PADI phase-1 physics runtime telemetry
    padi_debug: bool = False                         # Print per-step PADI signals when runtime is enabled
    padi_video_overlay: bool = False                # Overlay PADI scores on rollout MP4 frames when enabled
    padi_overlay_position: str = "top_left"         # Overlay position: top_left/top_right/bottom_left/bottom_right

    gsdr: bool = False                               # Enable GSDR geometry-risk-only budget telemetry
    gsdr_debug: bool = False                         # Print per-step GSDR budget telemetry
    gsdr_base_keep_ratio: float = 0.25              # Base keep ratio under low geometry risk
    gsdr_max_keep_ratio: float = 0.75               # Max keep ratio before entering no-prune mode
    gsdr_g_low: float = 0.25                        # Risk <= g_low uses gsdr_base_keep_ratio
    gsdr_g_no_prune: float = 0.95                   # Risk >= g_no_prune forces no-prune
    gsdr_geometry_ema_alpha: float = 0.65           # EMA alpha for geometry risk smoothing
    gsdr_token_quantum: int = 2                     # Keep-token quantization quantum
    gsdr_num_patches_per_image: int = 256           # Simulated per-image vision patch token count before FastV integration

    use_fastv: bool = False                         # Enable FastV Stage-2 call chain
    fastv_k: int = 3                                # FastV pruning layer index
    fastv_r: float = 0.5                            # FastV prune ratio inside each image segment
    fastv_image_token_start_index: int = 1          # FastV vision token block start index
    fastv_image_token_length: int = 512             # FastV total vision patch tokens (exclude proprio/diffusion)
    fastv_patches_per_image: int = 256              # FastV per-image patch count
    fastv_score_source: str = "text_mean"           # FastV score source: text_mean/last_query/action_mean
    fastv_debug: bool = False                       # Print one-line FastV pruning summaries
    fastv_sanity_assert: bool = False               # Raise RuntimeError on FastV sanity check failures
    fastv_video_overlay: bool = False               # Overlay FastV pruning mask on dual-view rollout frames
    fastv_overlay_alpha: int = 170                  # Dark mask strength for pruned FastV patches
    fastv_overlay_label: bool = True                # Show tiny "Global/Wrist/FastV mask" labels

    measure_latency: bool = False                    # Enable latency telemetry (policy query + LLM forward)
    latency_warmup_queries: int = 3                 # Number of initial policy queries excluded from summary
    latency_log_per_query: bool = False             # Print one-line latency per policy query

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.gsdr:
        if not cfg.use_padi_runtime:
            logger.warning("[GSDR] --gsdr=True automatically enables --use_padi_runtime=True")
            cfg.use_padi_runtime = True
        if not cfg.use_fastv:
            logger.warning("[GSDR] --gsdr=True automatically enables --use_fastv=True")
            cfg.use_fastv = True
        cfg.fastv_r = 1.0 - float(cfg.gsdr_base_keep_ratio)
        assert 0.0 < cfg.gsdr_base_keep_ratio <= 1.0
        assert cfg.gsdr_base_keep_ratio <= cfg.gsdr_max_keep_ratio <= 1.0
        assert 0.0 <= cfg.gsdr_g_low < cfg.gsdr_g_no_prune <= 1.0
        assert 0.0 < cfg.gsdr_geometry_ema_alpha <= 1.0
        assert cfg.gsdr_token_quantum >= 1
        assert cfg.gsdr_num_patches_per_image > 0

    assert 0.0 <= cfg.fastv_r < 1.0, "fastv_r must satisfy 0.0 <= fastv_r < 1.0"
    assert cfg.fastv_k >= 0, "fastv_k must be >= 0"
    assert cfg.fastv_image_token_start_index >= 0, "fastv_image_token_start_index must be >= 0"
    assert cfg.fastv_image_token_length > 0, "fastv_image_token_length must be > 0"
    assert cfg.fastv_patches_per_image > 0, "fastv_patches_per_image must be > 0"
    assert cfg.fastv_score_source in ["text_mean", "last_query", "action_mean"], "invalid fastv_score_source"
    if cfg.use_fastv:
        expected_len = cfg.num_images_in_input * cfg.fastv_patches_per_image
        if cfg.fastv_image_token_length != expected_len:
            logger.warning(
                "[PADI-OFT FastV] recommended fastv_image_token_length == num_images_in_input * fastv_patches_per_image "
                f"(got {cfg.fastv_image_token_length} vs {expected_len})"
            )


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def log_run_configuration(cfg: GenerateConfig, log_file=None):
    action_mode = "l1_regression" if cfg.use_l1_regression else ("diffusion" if cfg.use_diffusion else "other")
    log_message(
        f"[PADI-OFT Config] task_suite={cfg.task_suite_name} checkpoint={cfg.pretrained_checkpoint} "
        f"trials_per_task={cfg.num_trials_per_task} seed={cfg.seed}",
        log_file,
    )
    log_message(
        f"[PADI-OFT Config] action_mode={action_mode} use_diffusion={cfg.use_diffusion} use_film={cfg.use_film} "
        f"num_images={cfg.num_images_in_input} use_proprio={cfg.use_proprio} center_crop={cfg.center_crop}",
        log_file,
    )

    active_modules = []
    if cfg.use_padi_runtime:
        active_modules.append("padi")
    if cfg.use_fastv:
        active_modules.append("fastv")
    if cfg.gsdr:
        active_modules.append("gsdr")
    if cfg.measure_latency:
        active_modules.append("latency")
    if cfg.fastv_video_overlay:
        active_modules.append("fastv_video_overlay")
    if cfg.padi_video_overlay:
        active_modules.append("padi_video_overlay")
    log_message(
        f"[PADI-OFT Config] active_modules={','.join(active_modules) if active_modules else 'baseline'}",
        log_file,
    )

    if cfg.use_padi_runtime:
        log_message(
            f"[PADI-OFT Config][PADI] profile=final_oft_calibrated debug={cfg.padi_debug} "
            f"video_overlay={cfg.padi_video_overlay} overlay_position={cfg.padi_overlay_position}",
            log_file,
        )

    if cfg.use_fastv:
        if cfg.gsdr:
            initial_fastv_r = 1.0 - float(cfg.gsdr_base_keep_ratio)
            log_message(
                f"[PADI-OFT Config][FastV] mode=dynamic_by_gsdr fastv_k={cfg.fastv_k} "
                f"initial_fastv_r={initial_fastv_r:.4f} image_start={cfg.fastv_image_token_start_index} "
                f"image_length={cfg.fastv_image_token_length} patches_per_image={cfg.fastv_patches_per_image} "
                f"num_images={cfg.num_images_in_input} score_source={cfg.fastv_score_source} "
                f"debug={cfg.fastv_debug} sanity_assert={cfg.fastv_sanity_assert}",
                log_file,
            )
        else:
            log_message(
                f"[PADI-OFT Config][FastV] mode=fixed fastv_k={cfg.fastv_k} fastv_r={cfg.fastv_r:.4f} "
                f"image_start={cfg.fastv_image_token_start_index} image_length={cfg.fastv_image_token_length} "
                f"patches_per_image={cfg.fastv_patches_per_image} num_images={cfg.num_images_in_input} "
                f"score_source={cfg.fastv_score_source} debug={cfg.fastv_debug} "
                f"sanity_assert={cfg.fastv_sanity_assert}",
                log_file,
            )

    if cfg.gsdr:
        log_message(
            f"[PADI-OFT Config][GSDR] base_keep={cfg.gsdr_base_keep_ratio:.4f} max_keep={cfg.gsdr_max_keep_ratio:.4f} "
            f"g_low={cfg.gsdr_g_low:.4f} g_no_prune={cfg.gsdr_g_no_prune:.4f} "
            f"ema_alpha={cfg.gsdr_geometry_ema_alpha:.4f} token_quantum={cfg.gsdr_token_quantum} "
            f"num_vision_tokens={cfg.fastv_image_token_length} debug={cfg.gsdr_debug}",
            log_file,
        )

    if cfg.measure_latency:
        log_message(
            f"[PADI-OFT Config][Latency] enabled=True warmup_queries={cfg.latency_warmup_queries} "
            f"log_per_query={cfg.latency_log_per_query} mode={_get_latency_mode(cfg)}",
            log_file,
        )

    if cfg.fastv_video_overlay and cfg.padi_video_overlay:
        log_message(
            "[PADI-OFT Config][Video] fastv_video_overlay=True padi_video_overlay=True effective=fastv",
            log_file,
        )
    elif cfg.fastv_video_overlay:
        log_message(
            f"[PADI-OFT Config][FastV-Overlay] alpha={cfg.fastv_overlay_alpha} label={cfg.fastv_overlay_label}",
            log_file,
        )
    elif cfg.padi_video_overlay:
        log_message(f"[PADI-OFT Config][PADI-Overlay] position={cfg.padi_overlay_position}", log_file)


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay





def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def _get_latency_mode(cfg: GenerateConfig) -> str:
    modes = []
    if getattr(cfg, "gsdr", False):
        modes.append(f"fastv_dynamic_gsdr_base{cfg.gsdr_base_keep_ratio}_max{cfg.gsdr_max_keep_ratio}_k{cfg.fastv_k}")
    elif getattr(cfg, "use_fastv", False):
        modes.append(f"fastv_r{cfg.fastv_r}_k{cfg.fastv_k}")
    if getattr(cfg, "use_padi_runtime", False):
        modes.append("padi_runtime")
    if not modes:
        return "baseline"
    return "+".join(modes)




def _set_model_fastv_runtime_budget(model, prune_ratio, keep_ratio, source, gsdr_out=None, num_vision_tokens=None):
    prune_ratio = float(np.clip(prune_ratio, 0.0, 1.0))
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))

    model.fastv_r = prune_ratio
    model.runtime_fastv_r = prune_ratio
    model.runtime_fastv_keep_ratio = keep_ratio
    model.runtime_fastv_budget_source = source

    if hasattr(model, "config"):
        model.config.fastv_r = prune_ratio
        model.config.runtime_fastv_r = prune_ratio
        model.config.runtime_fastv_keep_ratio = keep_ratio
        model.config.runtime_fastv_budget_source = source

    if gsdr_out is not None:
        model.last_gsdr_out = gsdr_out
        model.last_gsdr_apply_info = {
            "source": source,
            "keep_ratio": keep_ratio,
            "prune_ratio": prune_ratio,
            "keep_tokens": int(gsdr_out.keep_tokens),
            "num_vision_tokens": int(gsdr_out.num_vision_tokens),
            "no_prune": bool(gsdr_out.no_prune),
            "geometry_risk_smooth": float(gsdr_out.geometry_risk_smooth),
            "g_raw": float(gsdr_out.g_raw),
        }
    else:
        model.last_gsdr_out = None
        num_vision_tokens = 0 if num_vision_tokens is None else int(num_vision_tokens)
        keep_tokens = int(round(float(num_vision_tokens) * keep_ratio))
        model.last_gsdr_apply_info = {
            "source": source,
            "keep_ratio": keep_ratio,
            "prune_ratio": prune_ratio,
            "keep_tokens": keep_tokens,
            "num_vision_tokens": num_vision_tokens,
            "no_prune": bool(keep_ratio >= 1.0),
        }


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    padi_runtime=None,
    gsdr_controller=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    model.last_gsdr_out = None
    model.last_gsdr_apply_info = None
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    padi_episode_telemetry = []
    latest_fastv_pruning_info = None
    latest_gsdr_out = None
    latest_gsdr_apply_info = None
    latency_records = []
    policy_query_idx = 0

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, _ = prepare_observation(obs, resize_size)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                if cfg.gsdr:
                    if latest_gsdr_out is None:
                        keep_ratio = float(cfg.gsdr_base_keep_ratio)
                        source = "gsdr_initial_base"
                        gsdr_out_for_apply = None
                    else:
                        keep_ratio = float(latest_gsdr_out.keep_ratio)
                        source = "gsdr"
                        gsdr_out_for_apply = latest_gsdr_out

                    dynamic_fastv_r = 1.0 - keep_ratio
                    _set_model_fastv_runtime_budget(
                        model,
                        prune_ratio=dynamic_fastv_r,
                        keep_ratio=keep_ratio,
                        source=source,
                        gsdr_out=gsdr_out_for_apply,
                        num_vision_tokens=cfg.fastv_image_token_length,
                    )
                    latest_gsdr_apply_info = getattr(model, "last_gsdr_apply_info", None)
                    if cfg.gsdr_debug:
                        if gsdr_out_for_apply is None:
                            log_message(
                                f"[PADI-OFT GSDR-Apply] step={t} source={source} "
                                f"keep_ratio={keep_ratio:.4f} prune_ratio={dynamic_fastv_r:.4f} "
                                f"max_keep_ratio={cfg.gsdr_max_keep_ratio:.4f}",
                                log_file,
                            )
                        else:
                            log_message(
                                f"[PADI-OFT GSDR-Apply] step={t} source={source} "
                                f"keep_ratio={keep_ratio:.4f} prune_ratio={dynamic_fastv_r:.4f} "
                                f"keep_tokens={int(gsdr_out_for_apply.keep_tokens)} "
                                f"num_vision_tokens={int(gsdr_out_for_apply.num_vision_tokens)} "
                                f"no_prune={bool(gsdr_out_for_apply.no_prune)} "
                                f"g_raw={float(gsdr_out_for_apply.g_raw):.4f} "
                                f"geometry_risk_smooth={float(gsdr_out_for_apply.geometry_risk_smooth):.4f} "
                                f"max_keep_ratio={cfg.gsdr_max_keep_ratio:.4f}",
                                log_file,
                            )

                # Query model to get action
                if cfg.measure_latency:
                    _cuda_sync_if_available()
                    policy_t0 = time.perf_counter()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                if cfg.measure_latency:
                    _cuda_sync_if_available()
                    policy_t1 = time.perf_counter()
                    policy_query_latency_ms = float((policy_t1 - policy_t0) * 1000.0)
                else:
                    policy_query_latency_ms = None
                latest_fastv_pruning_info = getattr(model, "last_pruning_info", None)
                llm_latency_ms = getattr(model, "last_llm_latency_ms", None)
                if cfg.measure_latency:
                    record = {
                        "query_idx": policy_query_idx,
                        "step": t,
                        "policy_query_latency_ms": policy_query_latency_ms,
                        "llm_forward_latency_ms": llm_latency_ms,
                        "use_fastv": cfg.use_fastv,
                        "fastv_r": getattr(model, "fastv_r", cfg.fastv_r) if cfg.use_fastv else None,
                        "fastv_k": cfg.fastv_k if cfg.use_fastv else None,
                    }
                    if latest_fastv_pruning_info is not None:
                        for k in [
                            "original_seq_length",
                            "kept_seq_length",
                            "pruned_count",
                            "num_keep_per_image",
                            "skipped",
                            "skip_reason",
                            "pruned_indices",
                        ]:
                            if k in latest_fastv_pruning_info:
                                record[k] = latest_fastv_pruning_info[k]
                    gsdr_apply_info = getattr(model, "last_gsdr_apply_info", None)
                    if gsdr_apply_info is not None:
                        record.update({
                            "gsdr_keep_ratio": gsdr_apply_info.get("keep_ratio"),
                            "gsdr_prune_ratio": gsdr_apply_info.get("prune_ratio"),
                            "gsdr_keep_tokens": gsdr_apply_info.get("keep_tokens"),
                            "gsdr_num_vision_tokens": gsdr_apply_info.get("num_vision_tokens"),
                            "gsdr_no_prune": gsdr_apply_info.get("no_prune"),
                            "gsdr_geometry_risk_smooth": gsdr_apply_info.get("geometry_risk_smooth"),
                            "gsdr_g_raw": gsdr_apply_info.get("g_raw"),
                            "gsdr_source": gsdr_apply_info.get("source"),
                        })
                    latency_records.append(record)
                    if cfg.latency_log_per_query:
                        kept = record.get("kept_seq_length", None)
                        pruned = record.get("pruned_count", None)
                        log_message(
                            f"[PADI-OFT Latency-Query] step={t} query={policy_query_idx} "
                            f"policy_ms={policy_query_latency_ms:.3f} llm_ms={llm_latency_ms} "
                            f"use_fastv={cfg.use_fastv} kept={kept} pruned={pruned}",
                            log_file,
                        )
                    policy_query_idx += 1
                should_run_fastv_sanity = cfg.use_fastv and (cfg.fastv_debug or cfg.fastv_sanity_assert)
                if should_run_fastv_sanity:
                    pruning_info = getattr(model, "last_pruning_info", None)
                    remap_info = getattr(model, "last_action_remap_info", None)
                    if pruning_info is None:
                        if cfg.fastv_debug:
                            log_message(
                                f"[PADI-OFT FastV-Sanity] step={t} ERROR pruning_info=None; FastV may not have been called.",
                                log_file,
                            )
                        if cfg.fastv_sanity_assert:
                            raise RuntimeError("FastV sanity assert failed: pruning_info is None")
                    else:
                        skipped = bool(pruning_info.get("skipped", True))
                        skip_reason = pruning_info.get("skip_reason", None)
                        original_seq_length = int(pruning_info.get("original_seq_length", -1))
                        kept_seq_length = int(pruning_info.get("kept_seq_length", -1))
                        fastv_k = pruning_info.get("fastv_k", None)
                        fastv_r = pruning_info.get("fastv_r", None)
                        vision_start = int(pruning_info.get("image_token_start_index", 0))
                        vision_len = int(pruning_info.get("image_token_length", 0))
                        vision_end = vision_start + vision_len
                        num_images_in_input = pruning_info.get("num_images_in_input", None)
                        patches_per_image = pruning_info.get("patches_per_image", None)
                        num_keep_per_image = pruning_info.get("num_keep_per_image", [])
                        pruned_indices = pruning_info.get("pruned_indices", [])
                        kept_indices = pruning_info.get("kept_indices", [])
                        pruned_list = pruned_indices.tolist() if hasattr(pruned_indices, "tolist") else list(pruned_indices)
                        kept_list = kept_indices.tolist() if hasattr(kept_indices, "tolist") else list(kept_indices)
                        pruned_count = len(pruned_list)
                        expected_pruned_count = int(vision_len - sum(num_keep_per_image)) if num_keep_per_image else 0
                        expected_kept_seq_length = int(original_seq_length - expected_pruned_count)
                        pruned_within_vision = all(vision_start <= idx < vision_end for idx in pruned_list)
                        bos_preserved = 0 in kept_list
                        post_tokens_preserved = all(idx < vision_end for idx in pruned_list)
                        proprio_preserved = vision_end in kept_list
                        score_source = pruning_info.get("score_source", None)
                        score_query_start = pruning_info.get("score_query_start", None)
                        score_query_end = pruning_info.get("score_query_end", None)
                        score_query_count = pruning_info.get("score_query_count", None)
                        fastv_query_type = pruning_info.get("fastv_query_type", None)
                        action_query_start = pruning_info.get("action_query_start", None)
                        action_query_end = pruning_info.get("action_query_end", None)
                        stop_token_index = pruning_info.get("stop_token_index", None)
                        proprio_start = pruning_info.get("proprio_start", None)
                        proprio_end = pruning_info.get("proprio_end", None)
                        layout = pruning_info.get("layout", None)
                        shape_ok = kept_seq_length == expected_kept_seq_length
                        if cfg.fastv_debug:
                            log_message(
                                f"[PADI-OFT FastV-Sanity] step={t} skipped={skipped} skip_reason={skip_reason} "
                                f"original={original_seq_length} kept={kept_seq_length} pruned={pruned_count} "
                                f"expected_pruned={expected_pruned_count} expected_kept={expected_kept_seq_length} "
                                f"shape_ok={shape_ok} pruned_within_vision={pruned_within_vision} "
                                f"bos_preserved={bos_preserved} proprio_preserved={proprio_preserved} "
                                f"post_tokens_preserved={post_tokens_preserved} fastv_k={fastv_k} fastv_r={fastv_r} "
                                f"image_token_start_index={vision_start} image_token_length={vision_len} image_end={vision_end} "
                                f"num_images_in_input={num_images_in_input} patches_per_image={patches_per_image} "
                                f"num_keep_per_image={num_keep_per_image} score_source={score_source} "
                                f"score_query_range=({score_query_start}, {score_query_end}) "
                                f"score_query_count={score_query_count} fastv_query_type={fastv_query_type} "
                                f"action_range=({action_query_start}, {action_query_end}) stop_token_index={stop_token_index} "
                                f"proprio_range=({proprio_start}, {proprio_end}) layout={layout}",
                                log_file,
                            )
                            if remap_info is not None:
                                log_message(
                                    f"[PADI-OFT FastV-ActionRemap] step={t} "
                                    f"original_start={remap_info.get('original_action_start')} "
                                    f"new_start={remap_info.get('new_action_start')} "
                                    f"shift={remap_info.get('action_shift')} "
                                    f"count={remap_info.get('action_token_count')} "
                                    f"preserved={remap_info.get('action_positions_preserved')} "
                                    f"contiguous={remap_info.get('action_new_positions_contiguous')} "
                                    f"fallback={remap_info.get('fallback', False)}",
                                    log_file,
                                )
                        if cfg.fastv_sanity_assert:
                            if skipped:
                                raise RuntimeError(f"FastV sanity assert failed: skipped=True reason={skip_reason}")
                            if not shape_ok:
                                raise RuntimeError("FastV sanity assert failed: shape_ok=False")
                            if not pruned_within_vision:
                                raise RuntimeError("FastV sanity assert failed: pruned_within_vision=False")
                            if not bos_preserved:
                                raise RuntimeError("FastV sanity assert failed: bos_preserved=False")
                            if not post_tokens_preserved:
                                raise RuntimeError("FastV sanity assert failed: post_tokens_preserved=False")
                            if cfg.use_proprio and not proprio_preserved:
                                raise RuntimeError("FastV sanity assert failed: proprio_preserved=False")
                            if remap_info is not None and not remap_info.get("action_positions_preserved", False):
                                raise RuntimeError("FastV sanity assert failed: action_positions_preserved=False")
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())

            padi_out = None
            eef_pos = None
            if padi_runtime is not None:
                eef_pos = obs.get("robot0_eef_pos") if isinstance(obs, dict) else None
                if eef_pos is None:
                    if cfg.padi_debug:
                        log_message(f"[PADI] warning: missing robot0_eef_pos at step={t}, skipping update", log_file)
                else:
                    gripper_raw = obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
                    gripper_value = None if gripper_raw is None else float(np.asarray(gripper_raw).reshape(-1).mean())
                    padi_out = padi_runtime.update(eef_pos=eef_pos, gripper_value=gripper_value, step_idx=t)
                    padi_episode_telemetry.append(
                        {
                            "step_idx": t,
                            "geometry_risk": padi_out.geometry_risk,
                            "precise_active": padi_out.precise_active,
                            "transit_score": padi_out.transit_score,
                            "debug": padi_out.debug,
                        }
                    )
                    if cfg.padi_debug:
                        dbg = padi_out.debug
                        log_message(
                            f"[PADI] step={t} geometry_risk={padi_out.geometry_risk:.4f} "
                            f"precise_active={padi_out.precise_active} transit_score={padi_out.transit_score:.4f} "
                            f"gripper_value={gripper_value} gripper_engaged={dbg.get('gripper_engaged')} "
                            f"gripper_stably_closed={dbg.get('gripper_stably_closed')} "
                            f"holding_confidence={dbg.get('holding_confidence', 0.0):.4f} "
                            f"gripper_score={dbg.get('gripper_score', 0.0):.4f} "
                            f"motion_score={dbg.get('motion_score', 0.0):.4f} "
                            f"not_precise_score={dbg.get('not_precise_score', 1.0):.4f} "
                            f"recent_mean_speed={dbg.get('recent_mean_speed', 0.0):.4f} "
                            f"recent_mean_disp={dbg.get('recent_mean_disp', 0.0):.4f} "
                            f"total_speed={dbg.get('total_speed', 0.0):.4f} "
                            f"xy_speed={dbg.get('xy_speed', 0.0):.4f} z_speed={dbg.get('z_speed', 0.0):.4f} "
                            f"curvature={dbg.get('curvature', 0.0):.4f} d_z={dbg.get('d_z', 0.0):.4f} "
                            f"u_s={dbg.get('u_s', 0.0):.4f} u_xy={dbg.get('u_xy', 0.0):.4f} "
                            f"dz_term={dbg.get('dz_term', 0.0):.4f} curve_term={dbg.get('curve_term', 0.0):.4f} "
                            f"slow_term={dbg.get('slow_term', 0.0):.4f} precise_term={dbg.get('precise_term', 0.0):.4f} "
                            f"local_term={dbg.get('local_term', 0.0):.4f} local_interaction_candidate={dbg.get('local_interaction_candidate')} "
                            f"precise_candidate={dbg.get('precise_candidate')}",
                            log_file,
                        )

            gsdr_out = None
            if gsdr_controller is not None and padi_out is not None:
                expected_vision_tokens = cfg.gsdr_num_patches_per_image * cfg.num_images_in_input
                num_vision_tokens = cfg.fastv_image_token_length
                if expected_vision_tokens != num_vision_tokens:
                    logger.warning(
                        "[GSDR] expected_vision_tokens (gsdr_num_patches_per_image*num_images_in_input) "
                        f"!= fastv_image_token_length ({expected_vision_tokens} vs {num_vision_tokens}); "
                        "using fastv_image_token_length for GSDR update."
                    )
                gsdr_out = gsdr_controller.update(
                    geometry_risk=padi_out.geometry_risk,
                    base_keep_ratio=cfg.gsdr_base_keep_ratio,
                    num_vision_tokens=num_vision_tokens,
                )
                latest_gsdr_out = gsdr_out
                if cfg.gsdr_debug:
                    log_message(
                        f"[GSDR] step={t} g_raw={gsdr_out.g_raw:.4f} "
                        f"geometry_risk_smooth={gsdr_out.geometry_risk_smooth:.4f} "
                        f"keep_ratio_cont={gsdr_out.keep_ratio_cont:.4f} raw_keep_tokens={gsdr_out.raw_keep_tokens:.1f} "
                        f"base_keep_tokens={gsdr_out.base_keep_tokens} keep_ratio={gsdr_out.keep_ratio:.4f} "
                        f"keep_tokens={gsdr_out.keep_tokens} no_prune={gsdr_out.no_prune} "
                        f"max_keep_ratio={gsdr_out.debug.get('max_keep_ratio')} "
                        f"effective_max_keep_ratio={gsdr_out.debug.get('effective_max_keep_ratio')}",
                        log_file,
                    )

            if cfg.fastv_video_overlay:
                # FastV overlay takes priority over PADI overlay to keep visuals clean.
                frame_for_video = overlay_fastv_pruning_on_views(
                    observation["full_image"],
                    observation["wrist_image"],
                    latest_fastv_pruning_info,
                    alpha=cfg.fastv_overlay_alpha,
                    show_label=cfg.fastv_overlay_label,
                )
            else:
                frame_for_video = get_libero_image(obs)

            if (not cfg.fastv_video_overlay) and cfg.use_padi_runtime and cfg.padi_video_overlay:
                frame_for_video = overlay_padi_scores_on_frame(
                    frame_for_video, padi_out if eef_pos is not None else None, t, position=cfg.padi_overlay_position
                )
            replay_images.append(frame_for_video)

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images, padi_episode_telemetry, latency_records


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    padi_runtime=None,
    gsdr_controller=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    task_latency_records = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        if padi_runtime is not None:
            padi_runtime.reset()
        if gsdr_controller is not None:
            gsdr_controller.reset()

        success, replay_images, padi_episode_telemetry, latency_records = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            padi_runtime,
            gsdr_controller,
        )

        if cfg.measure_latency:
            task_latency_records.extend(latency_records)

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    if cfg.measure_latency and task_latency_records:
        warmup = max(int(cfg.latency_warmup_queries), 0)
        valid_records = [r for r in task_latency_records if int(r.get("query_idx", 0)) >= warmup]
        warmup_fallback = False
        if len(valid_records) == 0:
            valid_records = task_latency_records
            warmup_fallback = True

        policy_vals = [r.get("policy_query_latency_ms") for r in valid_records if r.get("policy_query_latency_ms") is not None]
        llm_vals = [r.get("llm_forward_latency_ms") for r in valid_records if r.get("llm_forward_latency_ms") is not None]

        policy_mean = float(np.mean(policy_vals)) if policy_vals else float("nan")
        policy_p50 = float(np.median(policy_vals)) if policy_vals else float("nan")
        policy_p90 = float(np.percentile(policy_vals, 90)) if policy_vals else float("nan")
        llm_mean = float(np.mean(llm_vals)) if llm_vals else float("nan")
        llm_p50 = float(np.median(llm_vals)) if llm_vals else float("nan")
        llm_p90 = float(np.percentile(llm_vals, 90)) if llm_vals else float("nan")

        token_orig_vals, token_kept_vals, token_pruned_vals, keep_ratio_vals = [], [], [], []
        for r in valid_records:
            original = r.get("original_seq_length")
            kept = r.get("kept_seq_length")
            if original is None or kept is None:
                continue
            try:
                original_f = float(original)
                kept_f = float(kept)
            except (TypeError, ValueError):
                continue
            if original_f <= 0:
                continue
            pruned = r.get("pruned_count")
            if pruned is None and r.get("pruned_indices") is not None:
                pruned_indices = r.get("pruned_indices")
                pruned = len(pruned_indices.tolist()) if hasattr(pruned_indices, "tolist") else len(pruned_indices)
            if pruned is None:
                pruned = original_f - kept_f
            token_orig_vals.append(original_f)
            token_kept_vals.append(kept_f)
            token_pruned_vals.append(float(pruned))
            keep_ratio_vals.append(kept_f / original_f)

        if token_orig_vals and token_kept_vals and token_pruned_vals and keep_ratio_vals:
            token_orig = f"{float(np.mean(token_orig_vals)):.3f}"
            token_kept = f"{float(np.mean(token_kept_vals)):.3f}"
            token_pruned = f"{float(np.mean(token_pruned_vals)):.3f}"
            token_keep_ratio = f"{float(np.mean(keep_ratio_vals)):.4f}"
        else:
            token_orig = "NA"
            token_kept = "NA"
            token_pruned = "NA"
            token_keep_ratio = "NA"

        gsdr_keep_ratio_vals = [float(r["gsdr_keep_ratio"]) for r in valid_records if r.get("gsdr_keep_ratio") is not None]
        gsdr_prune_ratio_vals = [float(r["gsdr_prune_ratio"]) for r in valid_records if r.get("gsdr_prune_ratio") is not None]
        gsdr_no_prune_vals = [1.0 if bool(r.get("gsdr_no_prune")) else 0.0 for r in valid_records if r.get("gsdr_no_prune") is not None]
        gsdr_summary = ""
        if gsdr_keep_ratio_vals:
            gsdr_keep_ratio_mean = float(np.mean(gsdr_keep_ratio_vals))
            gsdr_prune_ratio_mean = float(np.mean(gsdr_prune_ratio_vals)) if gsdr_prune_ratio_vals else float("nan")
            gsdr_no_prune_rate = float(np.mean(gsdr_no_prune_vals)) if gsdr_no_prune_vals else float("nan")
            gsdr_summary = (
                f" gsdr_keep_ratio_mean={gsdr_keep_ratio_mean:.4f}"
                f" gsdr_prune_ratio_mean={gsdr_prune_ratio_mean:.4f}"
                f" gsdr_no_prune_rate={gsdr_no_prune_rate:.4f}"
            )

        log_message(
            "[PADI-OFT Latency] "
            f"scope=task task_id={task_id} task=\"{task_description}\" mode={_get_latency_mode(cfg)} episodes={task_episodes} "
            f"queries={len(task_latency_records)} measured={len(valid_records)} warmup={warmup} "
            f"warmup_fallback={warmup_fallback} "
            f"policy_ms_mean={policy_mean:.3f} policy_ms_p50={policy_p50:.3f} policy_ms_p90={policy_p90:.3f} "
            f"llm_ms_mean={llm_mean:.3f} llm_ms_p50={llm_p50:.3f} llm_ms_p90={llm_p90:.3f} "
            f"token_orig={token_orig} token_kept={token_kept} token_pruned={token_pruned} token_keep_ratio={token_keep_ratio}"
            f"{gsdr_summary}",
            log_file,
        )

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    padi_runtime = None
    if cfg.use_padi_runtime:
        padi_config = PadiPhysicsConfig()
        padi_runtime = PadiPhysicsAwareRuntime(padi_config)
        logger.info("[PADI] using final OFT-calibrated physics profile")
    if cfg.padi_debug and not cfg.use_padi_runtime:
        logger.warning("--padi_debug=True but --use_padi_runtime=False; PADI telemetry disabled.")

    gsdr_controller = None
    if cfg.gsdr:
        gsdr_controller = PadiGSDRController(
            PadiGSDRConfig(
                g_low=cfg.gsdr_g_low,
                g_no_prune=cfg.gsdr_g_no_prune,
                max_keep_ratio=cfg.gsdr_max_keep_ratio,
                geometry_ema_alpha=cfg.gsdr_geometry_ema_alpha,
                token_quantum=cfg.gsdr_token_quantum,
            )
        )
        logger.info("[GSDR] enabled: geometry-risk-only budget controller telemetry")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_run_configuration(cfg, log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            padi_runtime,
            gsdr_controller,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
