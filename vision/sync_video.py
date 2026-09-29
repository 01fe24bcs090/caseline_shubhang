"""
CaseLine — Synchronized Multi-Camera Video Compositer.

Stitches multiple annotated camera videos side-by-side into a single
composite video for synchronized review. All videos are frame-aligned
and padded to the length of the longest input.
"""

import os
import math
from typing import List, Optional

import cv2
import numpy as np


def create_sync_video(
    video_paths: List[str],
    output_path: str,
    target_height: int = 480,
    max_columns: int = 2,
    fps: Optional[float] = None,
    codec: str = "VP80",
) -> str:
    """
    Create a synchronized side-by-side composite video from multiple camera feeds.

    Parameters
    ----------
    video_paths : list of str
        Paths to the annotated video files to synchronize.
    output_path : str
        Output file path for the composite video.
    target_height : int
        Height to resize each individual video panel to (maintains aspect ratio).
    max_columns : int
        Maximum number of video panels per row in the grid.
    fps : float or None
        Output FPS. If None, uses the FPS of the first video.
    codec : str
        FourCC codec string. Default 'VP80' for .webm output.

    Returns
    -------
    str
        Path to the generated composite video.
    """
    if len(video_paths) < 2:
        raise ValueError("At least 2 video paths are required for sync playback.")

    # Open all video captures
    caps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        caps.append(cap)

    # Determine output FPS from the first video if not specified
    if fps is None:
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

    # Calculate panel dimensions (maintain aspect ratio, normalise height)
    panel_sizes = []
    for cap in caps:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = target_height / h if h > 0 else 1.0
        panel_w = int(w * scale)
        panel_h = target_height
        panel_sizes.append((panel_w, panel_h))

    # Calculate grid layout
    n_videos = len(video_paths)
    n_cols = min(n_videos, max_columns)
    n_rows = math.ceil(n_videos / n_cols)

    # Uniform panel width = max across all panels (for clean grid)
    uniform_w = max(pw for pw, _ in panel_sizes)
    uniform_h = target_height

    # Canvas dimensions
    canvas_w = uniform_w * n_cols
    canvas_h = uniform_h * n_rows

    # Extract camera labels from filenames
    labels = []
    for path in video_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        cam_id = name.split("_")[0] if "_" in name else name
        labels.append(cam_id.upper())

    # Prepare output writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    if not out.isOpened():
        # Fallback to mp4
        fallback = output_path.replace(".webm", ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(fallback, fourcc, fps, (canvas_w, canvas_h))
        output_path = fallback

    frame_count = 0
    print(f"[SyncVideo] Compositing {n_videos} videos → {n_cols}×{n_rows} grid "
          f"({canvas_w}×{canvas_h} @ {fps:.1f} FPS)")

    while True:
        panels = []
        any_active = False

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                any_active = True
                # Resize to uniform panel size
                panel = cv2.resize(frame, (uniform_w, uniform_h))
            else:
                # Video ended — show black panel with "END" label
                panel = np.zeros((uniform_h, uniform_w, 3), dtype=np.uint8)
                cv2.putText(
                    panel, f"{labels[i]} — END",
                    (uniform_w // 4, uniform_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2,
                )

            # Draw camera label on each panel
            label = labels[i]
            cv2.rectangle(panel, (0, 0), (len(label) * 18 + 20, 35), (0, 0, 0), -1)
            cv2.putText(
                panel, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )

            # Draw thin border around panel
            cv2.rectangle(panel, (0, 0), (uniform_w - 1, uniform_h - 1), (40, 40, 40), 1)

            panels.append(panel)

        if not any_active:
            break

        # Assemble grid canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for i, panel in enumerate(panels):
            row = i // n_cols
            col = i % n_cols
            y_off = row * uniform_h
            x_off = col * uniform_w
            canvas[y_off:y_off + uniform_h, x_off:x_off + uniform_w] = panel

        # Draw "SYNCHRONIZED PLAYBACK" header
        cv2.rectangle(canvas, (canvas_w // 2 - 180, 0), (canvas_w // 2 + 180, 30), (0, 0, 0), -1)
        cv2.putText(
            canvas, "SYNCHRONIZED PLAYBACK",
            (canvas_w // 2 - 170, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA,
        )

        out.write(canvas)
        frame_count += 1

    # Clean up
    for cap in caps:
        cap.release()
    out.release()

    print(f"[SyncVideo] Saved {frame_count} frames → {output_path}")
    return output_path
