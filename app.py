import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Calligraphy Write & Penetrate Animator",
    page_icon="✒️",
    layout="wide",
)

st.title("✒️ Calligraphy Writing & Custom Penetration Animator")
st.write(
    "Upload your calligraphy artwork. The app creates a clean, pure-white sheet, "
    "literally writes out selected letters step-by-step from your chosen direction, "
    "and animates the sword penetrating along your desired angle!"
)


# ============================================================
# SEGMENTATION & SOLID CANVAS GENERATION
# ============================================================

def segment_and_create_white_canvas(image):
    """
    Isolates calligraphy strokes and creates a pure white background canvas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold ink strokes (both black ink and colored fills)
    _, ink_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 50]), np.array([35, 255, 255]))
    
    combined_mask = cv2.bitwise_or(ink_mask, yellow_mask)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Pure white background
    white_canvas = np.full_like(image, 255, dtype=np.uint8)

    # Extract connected stroke components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    parts = []
    h, w = image.shape[:2]
    min_area = max(50, int(h * w * 0.0002))

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        comp_mask = (labels == i)
        ys, xs = np.where(comp_mask)
        if len(xs) < 15:
            continue

        min_x, max_x = int(np.min(xs)), int(np.max(xs))
        min_y, max_y = int(np.min(ys)), int(np.max(ys))

        parts.append({
            "id": i,
            "mask": comp_mask,
            "bbox": (min_x, min_y, max_x, max_y),
            "center": (float(centroids[i][0]), float(centroids[i][1])),
            "area": area,
        })

    # Sort left-to-right spatially
    parts.sort(key=lambda p: p["bbox"][0])
    return parts, white_canvas


def render_writing_stroke(canvas, original, mask, bbox, progress, draw_direction):
    """
    Simulates actual ink writing across a stroke component from various directions.
    """
    min_x, min_y, max_x, max_y = bbox
    w_box = max_x - min_x
    h_box = max_y - min_y
    
    grid_y, grid_x = np.ogrid[:canvas.shape[0], :canvas.shape[1]]

    if draw_direction == "Top -> Bottom":
        reveal_cutoff = min_y + int(progress * h_box)
        active_reveal = (grid_y <= reveal_cutoff) & mask

    elif draw_direction == "Bottom -> Top":
        reveal_cutoff = max_y - int(progress * h_box)
        active_reveal = (grid_y >= reveal_cutoff) & mask

    elif draw_direction == "Left -> Right":
        reveal_cutoff = min_x + int(progress * w_box)
        active_reveal = (grid_x <= reveal_cutoff) & mask

    elif draw_direction == "Right -> Left":
        reveal_cutoff = max_x - int(progress * w_box)
        active_reveal = (grid_x >= reveal_cutoff) & mask

    else:
        active_reveal = mask

    canvas[active_reveal] = original[active_reveal]


def render_sword_penetration(canvas, original, mask, bbox, progress, trajectory):
    """
    Animates the sword sliding and penetrating into position from custom trajectory angles.
    """
    h, w = canvas.shape[:2]
    min_x, min_y, max_x, max_y = bbox
    sword_w = max_x - min_x
    sword_h = max_y - min_y

    grid_x_mat, grid_y_mat = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid_y_og, grid_x_og = np.ogrid[:h, :w]

    # Calculate shift vector & dynamic mask
    if trajectory == "Top -> Bottom":
        y_shift = int((1.0 - progress) * (sword_h + 40))
        x_shift = 0
        reveal_cutoff = min_y + int(progress * (sword_h + 50))
        active_mask = (grid_y_og <= reveal_cutoff) & mask

    elif trajectory == "Bottom -> Top":
        y_shift = -int((1.0 - progress) * (sword_h + 40))
        x_shift = 0
        reveal_cutoff = max_y - int(progress * (sword_h + 50))
        active_mask = (grid_y_og >= reveal_cutoff) & mask

    elif trajectory == "Top-Left Diagonal":
        y_shift = int((1.0 - progress) * (sword_h + 30))
        x_shift = int((1.0 - progress) * (sword_w + 30))
        reveal_cutoff_y = min_y + int(progress * (sword_h + 40))
        reveal_cutoff_x = min_x + int(progress * (sword_w + 40))
        active_mask = (grid_y_og <= reveal_cutoff_y) & (grid_x_og <= reveal_cutoff_x) & mask

    elif trajectory == "Bottom-Right Diagonal":
        y_shift = -int((1.0 - progress) * (sword_h + 30))
        x_shift = -int((1.0 - progress) * (sword_w + 30))
        reveal_cutoff_y = max_y - int(progress * (sword_h + 40))
        reveal_cutoff_x = max_x - int(progress * (sword_w + 40))
        active_mask = (grid_y_og >= reveal_cutoff_y) & (grid_x_og >= reveal_cutoff_x) & mask

    # Warp pixel map to slide the sword
    map_x = grid_x_mat + x_shift
    map_y = grid_y_mat + y_shift
    warped_sword = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    canvas[active_mask] = warped_sword[active_mask]


def render_animation_frame(original, white_canvas, parts, assignments, frame_index, total_frames):
    h, w = original.shape[:2]
    canvas = white_canvas.copy()

    max_step = max([a["step"] for a in assignments.values()]) if assignments else 1
    step_duration = total_frames / float(max_step)
    current_step = int(frame_index / step_duration) + 1
    step_progress = (frame_index % step_duration) / step_duration

    for idx, part in enumerate(parts):
        config = assignments.get(idx, {"step": 1, "effect": "Reveal / Write", "direction": "Top -> Bottom"})
        step = config["step"]
        effect = config["effect"]
        direction = config["direction"]
        part_mask = part["mask"]

        # Component not reached yet -> stays pure white
        if current_step < step:
            continue

        # Component already completed in earlier step -> fully visible
        elif current_step > step:
            canvas[part_mask] = original[part_mask]

        # Active animation step
        else:
            if effect == "Sword Penetration":
                render_sword_penetration(canvas, original, part_mask, part["bbox"], step_progress, direction)
            else:
                render_writing_stroke(canvas, original, part_mask, part["bbox"], step_progress, direction)

    return canvas


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

        parts, white_canvas = segment_and_create_white_canvas(img)

        st.sidebar.header("⚙️ Sequence & Trajectory Controls")
        assignments = {}

        if parts:
            st.sidebar.write("Configure step order and drawing direction for each stroke:")

            writing_directions = ["Top -> Bottom", "Bottom -> Top", "Left -> Right", "Right -> Left"]
            penetration_trajectories = ["Top -> Bottom", "Bottom -> Top", "Top-Left Diagonal", "Bottom-Right Diagonal"]

            for idx, part in enumerate(parts):
                st.sidebar.markdown(f"**Stroke Part P{idx+1}**")
                
                # Smart defaults: P1 (N) -> Step 1, P2 (Sword) -> Step 3, P3 (O) -> Step 2
                default_step = 3 if idx == 1 else (1 if idx == 0 else 2)
                default_effect = "Sword Penetration" if idx == 1 else "Reveal / Write"

                col1, col2, col3 = st.sidebar.columns([1, 1.5, 1.5])
                
                with col1:
                    step = st.number_input(f"Step", min_value=1, max_value=10, value=default_step, key=f"step_{idx}")
                with col2:
                    effect = st.selectbox(
                        f"Type",
                        ["Reveal / Write", "Sword Penetration"],
                        index=1 if default_effect == "Sword Penetration" else 0,
                        key=f"effect_{idx}",
                    )
                with col3:
                    dir_options = penetration_trajectories if effect == "Sword Penetration" else writing_directions
                    direction = st.selectbox(
                        f"Direction",
                        dir_options,
                        index=0,
                        key=f"dir_{idx}",
                    )

                assignments[idx] = {"step": step, "effect": effect, "direction": direction}
                st.sidebar.markdown("---")
        else:
            st.sidebar.warning("No distinct ink strokes detected.")

        total_frames = st.sidebar.slider("Total Animation Duration (Frames)", 20, 80, 40)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Artwork")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Strokes & Blank Canvas Preview")
            preview = img.copy()
            for idx, p in enumerate(parts):
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(preview, f"P{idx+1}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("✨ Render Pure-White Calligraphy Animation", type="primary", disabled=(len(parts) == 0)):
            with st.spinner("Writing calligraphy on pure white page & animating custom penetration..."):
                frames = []
                for i in range(total_frames):
                    frame = render_animation_frame(img, white_canvas, parts, assignments, i, total_frames)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(Image.fromarray(rgb_frame))

                gif_bytes = build_gif(frames, duration=70)

            st.subheader("🎬 Final Animated Calligraphy Result")
            st.image(gif_bytes, width=480)

            st.download_button(
                "Download Custom GIF",
                data=gif_bytes,
                file_name="pure_white_calligraphy_penetration.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
