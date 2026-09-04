import io
import math
import zipfile
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Calligraphy Writing & Sword Animator (Batch Mode)",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✒️ Calligraphy Writing & Sword Animator")

st.write(
    "Upload multiple calligraphy artworks at once. Configure default parameters, "
    "process them automatically one by one, and download the entire result folder as a ZIP package."
)


# ============================================================
# IMAGE / MASK UTILITIES
# ============================================================

def resize_image(img, max_size=900):
    h, w = img.shape[:2]

    if max(h, w) <= max_size:
        return img

    scale = max_size / float(max(h, w))

    return cv2.resize(
        img,
        (
            max(1, int(w * scale)),
            max(1, int(h * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def create_clean_paper_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)


def segment_components_by_color_bands(image):
    """
    HSV Band Thresholding to isolate individual color strokes.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    min_area = max(20, int(h * w * 0.00008))

    color_ranges = [
        # Purple / Violet
        {
            "name": "Purple/Violet",
            "ranges": [(np.array([110, 25, 25]), np.array([155, 255, 255]))],
            "role": "Sword"
        },
        # Pink / Magenta
        {
            "name": "Pink/Magenta",
            "ranges": [
                (np.array([156, 25, 25]), np.array([180, 255, 255])),
                (np.array([0, 25, 25]), np.array([12, 255, 255]))
            ],
            "role": "Writing"
        },
        # Gold / Yellow
        {
            "name": "Gold/Yellow",
            "ranges": [(np.array([13, 25, 25]), np.array([35, 255, 255]))],
            "role": "Writing"
        },
        # Blue / Cyan
        {
            "name": "Blue/Cyan",
            "ranges": [(np.array([85, 25, 25]), np.array([109, 255, 255]))],
            "role": "Writing"
        },
        # Green
        {
            "name": "Green",
            "ranges": [(np.array([36, 25, 25]), np.array([84, 255, 255]))],
            "role": "Writing"
        },
        # Dark Ink
        {
            "name": "Dark Ink",
            "ranges": [(np.array([0, 0, 0]), np.array([180, 255, 130]))],
            "role": "Writing"
        }
    ]

    components = []
    processed_mask = np.zeros((h, w), dtype=np.uint8)

    for band in color_ranges:
        band_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in band["ranges"]:
            m = cv2.inRange(hsv, lower, upper)
            band_mask = cv2.bitwise_or(band_mask, m)

        band_mask = cv2.bitwise_and(band_mask, cv2.bitwise_not(processed_mask))

        kernel_close = np.ones((3, 3), np.uint8)
        band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
            band_mask, connectivity=8
        )

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            cx = int(stats[label, cv2.CC_STAT_LEFT])
            cy = int(stats[label, cv2.CC_STAT_TOP])
            cw = int(stats[label, cv2.CC_STAT_WIDTH])
            ch = int(stats[label, cv2.CC_STAT_HEIGHT])

            comp_mask = (cc_labels == label)
            mean_hsv = cv2.mean(hsv, mask=comp_mask.astype(np.uint8))

            components.append({
                "id": len(components) + 1,
                "mask": comp_mask,
                "bbox": (cx, cy, cx + cw - 1, cy + ch - 1),
                "center": (float(centroids[label][0]), float(centroids[label][1])),
                "area": area,
                "hue": mean_hsv[0],
                "sat": mean_hsv[1],
                "color_name": band["name"],
                "default_role": band["role"]
            })

            processed_mask = cv2.bitwise_or(processed_mask, (comp_mask.astype(np.uint8) * 255))

    components.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))
    return components


# ============================================================
# PATH & SKELETON UTILITIES
# ============================================================

def morphological_skeleton(binary):
    binary = (binary.astype(np.uint8) * 255)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    max_iterations = max(20, int(max(binary.shape[0], binary.shape[1])))

    for _ in range(max_iterations):
        eroded = cv2.erode(current, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        current = eroded
        if cv2.countNonZero(current) == 0:
            break

    return skeleton > 0


def skeleton_neighbors(point, skeleton):
    y, x = point
    h, w = skeleton.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors


def find_skeleton_endpoints(skeleton):
    endpoints = []
    ys, xs = np.where(skeleton)
    for y, x in zip(ys, xs):
        if len(skeleton_neighbors((y, x), skeleton)) == 1:
            endpoints.append((int(y), int(x)))
    return endpoints


def choose_start_point(skeleton, direction="Left -> Right"):
    endpoints = find_skeleton_endpoints(skeleton)
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return None

    if len(endpoints) == 0:
        points = list(zip(ys.astype(int), xs.astype(int)))
        if direction == "Top -> Bottom": return min(points, key=lambda p: p[0])
        if direction == "Bottom -> Top": return max(points, key=lambda p: p[0])
        if direction == "Left -> Right": return min(points, key=lambda p: p[1])
        if direction == "Right -> Left": return max(points, key=lambda p: p[1])
        return points[0]

    if direction == "Top -> Bottom": return min(endpoints, key=lambda p: (p[0], p[1]))
    if direction == "Bottom -> Top": return max(endpoints, key=lambda p: (p[0], -p[1]))
    if direction == "Left -> Right": return min(endpoints, key=lambda p: (p[1], p[0]))
    if direction == "Right -> Left": return max(endpoints, key=lambda p: (p[1], -p[0]))
    return min(endpoints, key=lambda p: (p[0] + p[1]))


def order_skeleton_path(skeleton, start):
    if start is None:
        return []

    remaining = set(zip(*np.where(skeleton)))
    if not remaining:
        return []

    if start not in remaining:
        start = min(remaining, key=lambda p: ((p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2))

    path = [start]
    remaining.remove(start)
    current = start
    max_points = min(len(path) + len(remaining), 20000)

    while remaining and len(path) < max_points:
        cy, cx = current
        candidates = [(((py - cy) ** 2 + (px - cx) ** 2), (py, px)) for py, px in remaining]
        candidates.sort(key=lambda x: x[0])
        _, next_point = candidates[0]
        path.append(next_point)
        remaining.remove(next_point)
        current = next_point

    return path


def smooth_path(path, smoothing_window=7):
    if len(path) < 3:
        return path

    pts = np.array([[p[1], p[0]] for p in path], dtype=np.float32)
    window = max(3, int(smoothing_window))
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / float(window)
    x = np.convolve(pts[:, 0], kernel, mode="same")
    y = np.convolve(pts[:, 1], kernel, mode="same")

    half = window // 2
    x[:half], x[-half:] = pts[:half, 0], pts[-half:, 0]
    y[:half], y[-half:] = pts[:half, 1], pts[-half:, 1]

    return [(int(round(py)), int(round(px))) for px, py in zip(x, y)]


def build_component_path(component, direction):
    x1, y1, x2, y2 = component["bbox"]
    local = component["mask"][y1:y2 + 1, x1:x2 + 1]
    skeleton = morphological_skeleton(local)

    if cv2.countNonZero(skeleton.astype(np.uint8)) < 2:
        ys, xs = np.where(local)
        if len(xs) == 0:
            return []
        order = np.argsort(xs + ys)
        return [(int(ys[i] + y1), int(xs[i] + x1)) for i in order[::max(1, len(order) // 300)]]

    start = choose_start_point(skeleton, direction)
    local_path = order_skeleton_path(skeleton, start)
    local_path = smooth_path(local_path, smoothing_window=7)

    return [(y + y1, x + x1) for y, x in local_path]


def build_path_reveal_mask(component_mask, path):
    h, w = component_mask.shape
    ys, xs = np.where(component_mask)

    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.float32)

    if not path:
        result = np.zeros((h, w), dtype=np.float32)
        result[component_mask] = 1.0
        return result

    path_arr = np.array([[p[1], p[0]] for p in path], dtype=np.float32)
    if len(path_arr) > 500:
        indices = np.linspace(0, len(path_arr) - 1, 500).astype(int)
        path_arr = path_arr[indices]

    progress = np.zeros((h, w), dtype=np.float32)
    chunk_size = 5000
    pixel_points = np.column_stack((xs, ys)).astype(np.float32)

    for start in range(0, len(pixel_points), chunk_size):
        end = min(start + chunk_size, len(pixel_points))
        chunk = pixel_points[start:end]
        distances = np.sqrt(((chunk[:, None, :] - path_arr[None, :, :]) ** 2).sum(axis=2))
        nearest = np.argmin(distances, axis=1)
        progress[ys[start:end], xs[start:end]] = nearest.astype(np.float32) / max(1, len(path_arr) - 1)

    return progress


def prepare_component(component, image, direction):
    path = build_component_path(component, direction)
    progress_map = build_path_reveal_mask(component["mask"], path)
    component["path"] = path
    component["progress_map"] = progress_map
    return component


# ============================================================
# RENDERING FUNCTIONS
# ============================================================

def render_writing_component(canvas, original, component, progress):
    mask = component["mask"]
    progress_map = component["progress_map"]
    reveal_width = 0.035
    threshold = progress + reveal_width
    active = mask & (progress_map <= threshold)
    canvas[active] = original[active]
    return active


def draw_single_pen_tip(canvas, current_point, color=(40, 40, 40)):
    if current_point is None:
        return
    y, x = current_point
    cv2.circle(canvas, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (x - 1, y - 1), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def direction_vector(direction):
    vectors = {
        "Top": (0, -1), "Bottom": (0, 1), "Left": (-1, 0), "Right": (1, 0),
        "Top-Left": (-1, -1), "Top-Right": (1, -1), "Bottom-Left": (-1, 1), "Bottom-Right": (1, 1)
    }
    return vectors.get(direction, (1, 0))


def normalize_vector(x, y):
    length = math.sqrt(x * x + y * y)
    if length == 0:
        return 1.0, 0.0
    return x / length, y / length


def shift_mask(mask, dx, dy):
    h, w = mask.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return shifted > 0


def shift_image(image, dx, dy):
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def render_sword(canvas, original, component, progress, entry_direction, penetration, angle_degrees, ref_bbox=None):
    mask = component["mask"]
    bbox = ref_bbox if ref_bbox is not None else component["bbox"]
    x1, y1, x2, y2 = bbox

    vw, vh = direction_vector(entry_direction)
    vw, vh = normalize_vector(vw, vh)

    angle_rad = math.radians(angle_degrees)
    rotated_x = vw * math.cos(angle_rad) - vh * math.sin(angle_rad)
    rotated_y = vw * math.sin(angle_rad) + vh * math.cos(angle_rad)
    rotated_x, rotated_y = normalize_vector(rotated_x, rotated_y)

    object_size = max(x2 - x1, y2 - y1)
    entry_distance = object_size * 1.6 + 80
    travel = (1.0 - progress) * entry_distance

    dx = rotated_x * travel
    dy = rotated_y * travel

    penetration_distance = (penetration / 100.0) * object_size * 0.75
    if progress > 0.72 and progress < 1.0:
        pen_phase = (progress - 0.72) / 0.28
        dx += rotated_x * penetration_distance * pen_phase
        dy += rotated_y * penetration_distance * pen_phase

    shifted = shift_image(original, int(round(dx)), int(round(dy)))
    shifted_mask = shift_mask(mask, int(round(dx)), int(round(dy)))
    canvas[shifted_mask] = shifted[shifted_mask]


def render_static_component(canvas, original, component):
    mask = component["mask"]
    canvas[mask] = original[mask]


def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def render_animation_frame(
    original, base_background, components, assignments, frame_index, total_frames,
    show_pen, group_bboxes, enable_fade_in, fade_in_frames
):
    canvas = base_background.copy()
    global_progress = 1.0 if total_frames <= 1 else frame_index / float(total_frames - 1)

    active_pen_point = None
    latest_active_time = -1.0

    # Pass 1: Static elements
    for idx, component in enumerate(components):
        if assignments.get(idx, {}).get("role") == "Static":
            render_static_component(canvas, original, component)

    # Pass 2: Writing elements
    for idx, component in enumerate(components):
        config = assignments.get(idx, {})
        if config.get("role", "Writing") != "Writing":
            continue
        start, duration = config.get("start", 0.0), config.get("duration", 1.0)
        if global_progress < start:
            continue
        local = max(0.0, min(1.0, (global_progress - start) / max(duration, 0.001)))
        local = ease_in_out(local)
        render_writing_component(canvas, original, component, local)

        if show_pen and 0.0 < local < 1.0 and start >= latest_active_time:
            path = component.get("path", [])
            if path:
                p_idx = max(0, min(int(local * (len(path) - 1)), len(path) - 1))
                active_pen_point = path[p_idx]
                latest_active_time = start

    # Pass 3: Sword / Weapon elements
    for idx, component in enumerate(components):
        config = assignments.get(idx, {})
        if config.get("role") != "Sword":
            continue
        group_id = config.get("group_id", "Standalone")
        ref_bbox = group_bboxes.get(group_id, component["bbox"]) if group_id != "Standalone" else component["bbox"]

        start, duration = config.get("start", 0.0), config.get("duration", 1.0)
        if global_progress < start:
            continue
        local = max(0.0, min(1.0, (global_progress - start) / max(duration, 0.001)))
        local = ease_out(local)
        render_sword(
            canvas, original, component, local,
            config.get("entry_direction", "Top-Right"),
            config.get("penetration", 0),
            config.get("angle", 0),
            ref_bbox=ref_bbox
        )

    if show_pen and active_pen_point is not None:
        draw_single_pen_tip(canvas, active_pen_point)

    if enable_fade_in and fade_in_frames > 0:
        fade_start_frame = max(0, total_frames - fade_in_frames)
        if frame_index >= fade_start_frame:
            alpha = (frame_index - fade_start_frame) / float(max(1, fade_in_frames - 1))
            alpha = max(0.0, min(1.0, alpha))
            canvas = cv2.addWeighted(canvas, 1.0 - alpha, original, alpha, 0)

    return canvas


def build_gif(frames, duration=60):
    if not frames:
        return b""
    prepared = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    buf = io.BytesIO()
    prepared[0].save(
        buf, format="GIF", save_all=True, append_images=prepared[1:],
        duration=duration, loop=0, optimize=False
    )
    return buf.getvalue()


# ============================================================
# BATCH PROCESSOR PIPELINE
# ============================================================

def process_single_image(img, bg_mode, sequence_strategy, writing_default_direction, show_pen, enable_fade_in, fade_in_percent, total_frames):
    img = resize_image(img, max_size=900)

    if bg_mode == "Pure White Canvas":
        base_background = np.full_like(img, 255, dtype=np.uint8)
    elif bg_mode == "Original Paper Background":
        base_background = img.copy()
    else:
        base_background = create_clean_paper_background(img)

    parts = segment_components_by_color_bands(img)
    if not parts:
        return None

    fade_in_frames = int((fade_in_percent / 100.0) * total_frames) if enable_fade_in else 0

    for component in parts:
        component["direction"] = writing_default_direction
        prepare_component(component, img, writing_default_direction)

    assignments = {}
    for idx, component in enumerate(parts):
        role = component.get("default_role", "Writing")
        if sequence_strategy == "Letters Written First, Then Sword Enters":
            default_write_start = int((idx / max(1, len(parts))) * 40)
            default_write_dur = 40
            default_sword_start = 50
            default_sword_dur = 35
        else:
            default_sword_start = 0
            default_sword_dur = 35
            default_write_start = int(40 + (idx / max(1, len(parts))) * 40)
            default_write_dur = 40

        if role == "Writing":
            assignments[idx] = {
                "role": "Writing", "group_id": "Standalone",
                "start": default_write_start / 100.0, "duration": default_write_dur / 100.0
            }
        elif role == "Sword":
            assignments[idx] = {
                "role": "Sword", "group_id": "Standalone",
                "start": default_sword_start / 100.0, "duration": default_sword_dur / 100.0,
                "entry_direction": "Top-Right", "penetration": 0, "angle": 0
            }
        else:
            assignments[idx] = {"role": "Static", "group_id": "Standalone", "start": 0.0, "duration": 1.0}

    frames = []
    for frame_index in range(total_frames):
        frame = render_animation_frame(
            img, base_background, parts, assignments, frame_index, total_frames,
            show_pen, {}, enable_fade_in, fade_in_frames
        )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(np.ascontiguousarray(rgb_frame)))

    return build_gif(frames, duration=60)


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Calligraphy Images (Multiple Selection Supported)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

st.sidebar.header("⚙️ Batch Render Settings")

bg_mode = st.sidebar.selectbox(
    "Background Canvas Style",
    ["Pure White Canvas", "Original Paper Background", "Cleaned Paper Texture"],
    index=0
)

sequence_strategy = st.sidebar.radio(
    "Animation Sequence Strategy",
    ["Letters Written First, Then Sword Enters", "Sword Enters First, Then Letters Written"],
    index=0
)

writing_default_direction = st.sidebar.selectbox(
    "Default Writing Direction",
    ["Left -> Right", "Top -> Bottom", "Bottom -> Top", "Right -> Left"],
    index=0
)

show_pen = st.sidebar.checkbox("Single Pen Tip Tracking All Writing Strokes", value=True)
enable_fade_in = st.sidebar.checkbox("Enable Final Artwork Fade-In Reveal", value=True)
fade_in_percent = st.sidebar.slider("Fade-In Duration %", 5, 40, 20) if enable_fade_in else 0
total_frames = st.sidebar.slider("Animation Frames", 30, 150, 60, step=5)
gif_duration = st.sidebar.slider("Frame Duration (ms)", 30, 150, 60, step=10)

if uploaded_files:
    st.info(f"📂 {len(uploaded_files)} image(s) loaded for batch animation processing.")

    if st.button("🚀 Process All Images & Generate Zip Folder", type="primary"):
        batch_progress = st.progress(0)
        status_text = st.empty()
        
        # In-Memory ZIP Buffer
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, file in enumerate(uploaded_files):
                status_text.write(f"Processing ({idx + 1}/{len(uploaded_files)}): **{file.name}**...")
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is not None:
                    gif_bytes = process_single_image(
                        img, bg_mode, sequence_strategy, writing_default_direction,
                        show_pen, enable_fade_in, fade_in_percent, total_frames
                    )

                    if gif_bytes:
                        out_filename = f"animated_{file.name.rsplit('.', 1)[0]}.gif"
                        zip_file.writestr(out_filename, gif_bytes)

                batch_progress.progress((idx + 1) / len(uploaded_files))

        status_text.success("🎉 All images processed successfully!")
        batch_progress.empty()

        zip_buffer.seek(0)
        st.download_button(
            "📦 Download All Animated GIFs (.ZIP Folder)",
            data=zip_buffer.getvalue(),
            file_name="calligraphy_animations_folder.zip",
            mime="application/zip"
        )
