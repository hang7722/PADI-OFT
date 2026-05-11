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
