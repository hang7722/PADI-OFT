from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay_padi_scores_on_frame(frame, padi_out, step_idx, position="top_left", panel_scale=0.82):
    if frame is None:
        return frame
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = arr.shape[0], arr.shape[1]
    panel_w = int(min(max(170, w * 0.42), 220) * panel_scale)
    pad = max(6, int(10 * panel_scale))
    line_h = max(12, int(16 * panel_scale))
    bar_w = max(70, int(96 * panel_scale))
    bar_h = max(6, int(8 * panel_scale))
    box_h = pad * 2 + line_h * 5
    margin = 10
    if position not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        position = "top_left"
    x0 = margin if "left" in position else max(0, w - panel_w - margin)
    y0 = margin if "top" in position else max(0, h - box_h - margin)

    g = 0.0 if padi_out is None else float(padi_out.geometry_risk)
    p = 0 if padi_out is None else int(bool(padi_out.precise_active))
    t = 0.0 if padi_out is None else float(padi_out.transit_score)
    debug = {} if padi_out is None else (padi_out.debug or {})
    startup = bool(debug.get("startup_guard_active", False)) and not bool(p)
    local = bool(debug.get("local_interaction_candidate", False))
    if padi_out is None:
        phase = "NO SIGNAL"
    elif startup:
        phase = "STARTUP"
    elif p:
        phase = "PRECISE"
    elif t >= 0.6 and g < 0.3:
        phase = "TRANSIT"
    elif local:
        phase = "LOCAL"
    else:
        phase = "IDLE"

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(11, int(12 * panel_scale)))
    except Exception:
        font = ImageFont.load_default()

    bg = (0, 0, 0, 175)
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + box_h), radius=max(6, int(8 * panel_scale)), fill=bg)
    else:
        draw.rectangle((x0, y0, x0 + panel_w, y0 + box_h), fill=bg)
    if phase == "PRECISE":
        draw.rectangle((x0, y0, x0 + panel_w, y0 + line_h + pad), fill=(120, 20, 20, 190))

    draw.text((x0 + pad, y0 + pad), f"PADI  |  {phase}", fill=(255, 255, 255, 255), font=font)
    g_label = f"G* {g:.3f}" if startup else f"G  {g:.3f}"
    draw.text((x0 + pad, y0 + pad + line_h), g_label, fill=(210, 235, 255, 255), font=font)
    draw.text((x0 + pad, y0 + pad + 2 * line_h), f"P  {'ON' if p else 'OFF'}", fill=(255, 220, 180, 255), font=font)
    draw.text((x0 + pad, y0 + pad + 3 * line_h), f"T  {t:.3f}", fill=(200, 255, 200, 255), font=font)
    draw.text((x0 + pad, y0 + pad + 4 * line_h), f"step {step_idx}", fill=(190, 190, 190, 255), font=font)

    bar_x = x0 + panel_w - pad - bar_w
    g_bar_y = y0 + pad + line_h + 2
    t_bar_y = y0 + pad + 3 * line_h + 2
    draw.rectangle((bar_x, g_bar_y, bar_x + bar_w, g_bar_y + bar_h), fill=(60, 60, 60, 170))
    draw.rectangle((bar_x, t_bar_y, bar_x + bar_w, t_bar_y + bar_h), fill=(60, 60, 60, 170))
    draw.rectangle((bar_x, g_bar_y, bar_x + int(bar_w * max(0.0, min(1.0, g))), g_bar_y + bar_h), fill=(80, 170, 255, 230))
    draw.rectangle((bar_x, t_bar_y, bar_x + int(bar_w * max(0.0, min(1.0, t))), t_bar_y + bar_h), fill=(100, 240, 120, 230))
    if startup:
        draw.text((x0 + pad, y0 + box_h - line_h), "startup/local baseline", fill=(255, 200, 120, 240), font=font)
    return np.asarray(img, dtype=np.uint8)


def _to_rgb_uint8(frame):
    if isinstance(frame, Image.Image):
        img = frame.convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _compute_grid(patches_per_image: int):
    side = int(np.sqrt(max(1, patches_per_image)))
    if side * side == patches_per_image:
        return side, side
    grid_h = 16
    grid_w = int(np.ceil(max(1, patches_per_image) / grid_h))
    return grid_h, grid_w


def _overlay_pruned_patches(frame_rgb, local_pruned, patches_per_image, alpha):
    out = frame_rgb.copy()
    h, w = out.shape[0], out.shape[1]
    grid_h, grid_w = _compute_grid(patches_per_image)
    strength = float(np.clip(alpha / 255.0, 0.0, 0.90))
    dark_value = 0.03

    for local_idx in local_pruned:
        row = int(local_idx) // grid_w
        col = int(local_idx) % grid_w
        if row < 0 or row >= grid_h:
            continue
        x0 = int(round(col * w / grid_w))
        y0 = int(round(row * h / grid_h))
        x1 = int(round((col + 1) * w / grid_w))
        y1 = int(round((row + 1) * h / grid_h))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = out[y0:y1, x0:x1, :]
        patch_f = patch.astype(np.float32) / 255.0
        dark = np.ones_like(patch_f) * dark_value
        masked = patch_f * (1.0 - strength) + dark * strength
        out[y0:y1, x0:x1, :] = np.clip(masked * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def overlay_fastv_pruning_on_views(global_frame, wrist_frame, pruning_info, alpha=170, show_label=True):
    global_rgb = _to_rgb_uint8(global_frame)
    wrist_rgb = _to_rgb_uint8(wrist_frame)

    gh, gw = global_rgb.shape[:2]
    wh, ww = wrist_rgb.shape[:2]
    if wh != gh:
        new_w = max(1, int(round(ww * (gh / max(1, wh)))))
        wrist_rgb = np.asarray(Image.fromarray(wrist_rgb, mode="RGB").resize((new_w, gh), Image.BILINEAR), dtype=np.uint8)

    global_img = Image.fromarray(global_rgb, mode="RGB")
    wrist_img = Image.fromarray(wrist_rgb, mode="RGB")
    has_mask = pruning_info is not None and not bool(pruning_info.get("skipped", True))
    if has_mask:
        image_token_start_index = int(pruning_info.get("image_token_start_index", 1))
        num_images_in_input = int(pruning_info.get("num_images_in_input", 2))
        patches_per_image = int(pruning_info.get("patches_per_image", 256))
        pruned_indices = pruning_info.get("pruned_indices", [])
        pruned = np.array(pruned_indices.tolist() if hasattr(pruned_indices, "tolist") else list(pruned_indices), dtype=np.int64)
        g_start = image_token_start_index
        g_end = g_start + patches_per_image
        global_local = pruned[(pruned >= g_start) & (pruned < g_end)] - g_start
        global_img = _overlay_pruned_patches(np.asarray(global_img), global_local, patches_per_image, alpha)
        if num_images_in_input >= 2:
            w_start = image_token_start_index + patches_per_image
            w_end = w_start + patches_per_image
            wrist_local = pruned[(pruned >= w_start) & (pruned < w_end)] - w_start
            wrist_img = _overlay_pruned_patches(np.asarray(wrist_img), wrist_local, patches_per_image, alpha)

    panel_h = global_img.height
    panel_w_left = global_img.width
    panel_w_right = wrist_img.width
    top_margin = max(24, int(panel_h * 0.08))
    bottom_margin = max(54, int(panel_h * 0.18))
    divider_w = 4
    merged_w = panel_w_left + divider_w + panel_w_right
    merged_h = top_margin + panel_h + bottom_margin
    merged = Image.new("RGB", (merged_w, merged_h), color=(0, 0, 0))
    merged.paste(global_img, (0, top_margin))
    merged.paste(wrist_img, (panel_w_left + divider_w, top_margin))

    if show_label:
        draw = ImageDraw.Draw(merged)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(22, int(panel_h * 0.08)))
        except Exception:
            font = ImageFont.load_default()
        left_label = "Global"
        right_label = "Wrist"
        left_bbox = draw.textbbox((0, 0), left_label, font=font)
        right_bbox = draw.textbbox((0, 0), right_label, font=font)
        left_w = left_bbox[2] - left_bbox[0]
        right_w = right_bbox[2] - right_bbox[0]
        y_lbl = top_margin + panel_h + max(8, int(bottom_margin * 0.22))
        left_x = (panel_w_left - left_w) // 2
        right_x = panel_w_left + divider_w + (panel_w_right - right_w) // 2
        draw.text((left_x, y_lbl), left_label, fill=(245, 245, 245), font=font)
        draw.text((right_x, y_lbl), right_label, fill=(245, 245, 245), font=font)

    return np.asarray(merged, dtype=np.uint8)
