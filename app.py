import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Calligraphy & Character Animator",
    page_icon="🗡️",
    layout="wide",
)

st.title("🗡️ Calligraphy & Character Penetration Animator")
st.write(
    "Upload calligraphy art. Assign dedicated motions like **Sword Penetration** "
    "to selected elements to watch them slide in and pierce the artwork dynamically."
)

# ============================================================
# ACCURATE HSV COLOR BOUNDS
# ============================================================

COLOR_RANGES = {
    "Black / Dark Ink": [(np.array([0, 0, 0]), np.array([180, 255, 65]))],
    "Red": [
        (np.array([0, 70, 50]), np.array([8, 255, 255])),
        (np.array([172, 70, 50]), np.array([180, 255, 255])),
    ],
    "Yellow / Gold": [(np.array([15, 70, 50]), np.array([34, 255, 255]))],
    "Green": [(np.array([35, 50, 40]), np.array([84, 255, 255]))],
    "Blue": [(np.array([106, 50, 40]), np.array([130, 255, 255]))],
    "Purple / Violet": [(np.array([131, 50, 40]), np.array([155, 255, 255]))],
}

ALL_MOTIONS = [
    "None (Static)",
    "Sword Penetration",
    "Advanced Walk",
    "Belly Laugh",
    "Natural Sway",
    "Playful Bounce",
    "Dynamic Wave",
    "Breathing Pulse",
]


def extract_accurate_colors(image):
    """Scans image using defined HSV ranges to isolate distinct strokes/ink."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]
    detected = []

    for name, ranges in COLOR_RANGES.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        pixel_count = cv2.countNonZero(combined_mask)
        coverage = (pixel_count / total_pixels) * 100

        if coverage > 0.05:
            detected.append({
                "name": name,
                "coverage": coverage,
                "mask": combined_mask,
            })

    return detected


def extract_independent_parts(detected_colors):
    """Separates connected component strokes for fine-grained animation control."""
    color_parts_map = {}

    for color_info in detected_colors:
        color_name = color_info["name"]
        mask = color_info["mask"]

        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        parts = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 30:
                continue

            comp = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(comp > 0)
            if len(xs) < 10:
                continue

            parts.append({
                "mask": comp > 0,
                "bbox": (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))),
                "center": (float(np.mean(xs)), float(np.mean(ys))),
                "area": area,
            })

        if parts:
            color_parts_map[color_name] = parts

    return color_parts_map


def animate_penetration(original, sword_part, background_img, frame_index, total_frames):
    """
    Renders a progressive vertical penetration effect:
    The element translates down from above while being clipped progressively
    to simulate entering/piercing into the artwork.
    """
    h, w = original.shape[:2]
    progress = float(frame_index) / float(total_frames - 1)

    min_x, min_y, max_x, max_y = sword_part["bbox"]
    sword_height = max_y - min_y

    # Calculate vertical movement (starts off-screen/above and slides down)
    y_offset = int((1.0 - progress) * (sword_height + 20))

    # Create displacement matrices
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y + y_offset  # Pull pixels from higher up

    # Warp the full image to move the sword
    warped_full = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Build progressive reveal mask (cut-off line moves down)
    reveal_line = min_y + int(progress * (sword_height + 30))
    reveal_mask = grid_y <= reveal_line

    # Combine shifted sword pixels onto static background using masks
    sword_mask = sword_part["mask"]
    active_mask = sword_mask & reveal_mask

    result = background_img.copy()
    result[active_mask] = warped_full[active_mask]

    return result


def animate_custom_frame(original, color_parts_map, assignments, frame_index, total_frames, intensity):
    h, w = original.shape[:2]

    # Create static background base image (fill sword areas with surrounding paper color)
    background_img = original.copy()
    for color_name, motion in assignments.items():
        if motion == "Sword Penetration":
            for part in color_parts_map[color_name]:
                # Inpaint/erase sword stroke to build clean background paper
                mask_uint = (part["mask"]).astype(np.uint8) * 255
                background_img = cv2.inpaint(background_img, mask_uint, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    frame = background_img.copy()

    # Process each color group based on assigned motion
    for color_name, parts in color_parts_map.items():
        motion = assignments.get(color_name, "None (Static)")

        if motion == "Sword Penetration":
            for part in parts:
                frame = animate_penetration(original, part, background_img, frame_index, total_frames)

        elif motion != "None (Static)":
            grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            map_x, map_y = grid_x.copy(), grid_y.copy()
            phase = 2.0 * math.pi * float(frame_index) / float(total_frames)

            for part in parts:
                mask = part["mask"]
                cx, cy = part["center"]
                dx = math.sin(phase) * intensity
                map_x[mask] -= dx

            warped = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # Blend distorted part onto composite frame
            for part in parts:
                frame[part["mask"]] = warped[part["mask"]]

    return frame


def build_gif(frames, duration=60):
    buf = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(buf, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0)
    return buf.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader("Upload Calligraphy Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image format.")
            st.stop()

        h, w = img.shape[:2]
        if max(h, w) > 900:
            scale = 900.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        detected_colors = extract_accurate_colors(img)
        color_parts_map = extract_independent_parts(detected_colors)

        st.sidebar.header("⚙️ Penetration Controls")
        assignments = {}

        if color_parts_map:
            st.sidebar.write("Select **'Sword Penetration'** for the sword ink stroke:")

            for idx, (color_name, parts) in enumerate(color_parts_map.items()):
                # Default dark ink to penetration for quick testing
                default_idx = 1 if "Black" in color_name else 0
                selected = st.sidebar.selectbox(
                    f"Stroke Layer: {color_name} ({len(parts)} segments)",
                    ALL_MOTIONS,
                    index=default_idx,
                    key=f"motion_{color_name}",
                )
                assignments[color_name] = selected
        else:
            st.sidebar.warning("No distinct ink strokes detected.")

        frame_count = st.sidebar.slider("Animation Resolution (Frames)", 12, 36, 24)
        intensity = float(st.sidebar.slider("Motion Speed / Intensity", 4, 30, 12))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Calligraphy")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Stroke Segments")
            preview = img.copy()
            for color_name, parts in color_parts_map.items():
                for p in parts:
                    x1, y1, x2, y2 = p["bbox"]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")

        if st.button("✨ Render Penetration Animation", type="primary", disabled=(len(color_parts_map) == 0)):
            with st.spinner("Rendering progressive piercing animation..."):
                frames = []
                for i in range(frame_count):
                    frame = animate_custom_frame(img, color_parts_map, assignments, i, frame_count, intensity)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(Image.fromarray(rgb_frame))

                gif_bytes = build_gif(frames, duration=70)

            st.subheader("🎬 Penetration Result")
            st.image(gif_bytes, width=450)

            st.download_button(
                "Download Piercing GIF",
                data=gif_bytes,
                file_name="sword_penetration_calligraphy.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
