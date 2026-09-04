import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Sequential Calligraphy Animator",
    page_icon="🖋️",
    layout="wide",
)

st.title("🖋️ Sequential Calligraphy & Penetration Animator")
st.write(
    "Upload your calligraphy artwork. Assign an **Appearance Order** to each detected "
    "stroke so elements reveal step-by-step from a blank canvas before the final penetration effect."
)


# ============================================================
# COMPONENT SEGMENTATION & INPAINTING
# ============================================================

def segment_calligraphy_elements(image):
    """
    Separates dark ink strokes and colored fills into distinct spatial components
    and generates a clean, stroke-free paper background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold dark ink strokes from light paper background
    _, ink_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    # Also capture yellow/gold colored fills inside characters
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR_HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 50]), np.array([35, 255, 255]))
    
    combined_mask = cv2.bitwise_or(ink_mask, yellow_mask)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Generate clean paper background by erasing all ink
    background_paper = cv2.inpaint(image, clean_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

    # Separate into connected components
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

    # Sort left-to-right based on bounding box
    parts.sort(key=lambda p: p["bbox"][0])
    return parts, background_paper


def render_sequential_frame(original, background_paper, parts, assignments, frame_index, total_frames):
    """
    Renders frames sequentially based on assigned appearance steps.
    """
    h, w = original.shape[:2]
    canvas = background_paper.copy()

    # Determine max sequence step
    max_step = max([a["step"] for a in assignments.values()]) if assignments else 1
    
    # Calculate global timeline phase
    step_duration = total_frames / float(max_step)
    current_step = int(frame_index / step_duration) + 1
    step_progress = (frame_index % step_duration) / step_duration

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    for idx, part in enumerate(parts):
        part_config = assignments.get(idx, {"step": 1, "effect": "Reveal / Write"})
        step = part_config["step"]
        effect = part_config["effect"]
        part_mask = part["mask"]

        # 1. Component hasn't appeared yet -> Stay blank paper
        if current_step < step:
            continue

        # 2. Component is fully revealed in an earlier step -> Draw static
        elif current_step > step:
            canvas[part_mask] = original[part_mask]

        # 3. Component is currently animating in this active step
        else:
            min_x, min_y, max_x, max_y = part["bbox"]
            
            if effect == "Sword Penetration":
                sword_h = max_y - min_y
                # Vertical translation: slides down from above
                y_shift = int((1.0 - step_progress) * (sword_h + 30))
                
                map_x = grid_x.copy()
                map_y = grid_y + y_shift
                warped_original = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                # Progressive linear penetration reveal
                reveal_y = min_y + int(step_progress * (sword_h + 40))
                active_reveal = (grid_y <= reveal_y) & part_mask

                canvas[active_reveal] = warped_original[active_reveal]

            elif effect == "Reveal / Write":
                # Brush write effect: progressive top-to-bottom stroke reveal
                part_h = max_y - min_y
                reveal_y = min_y + int(step_progress * part_h)
                active_reveal = (grid_y <= reveal_y) & part_mask

                canvas[active_reveal] = original[active_reveal]

            elif effect == "Fade In":
                # Alpha blending fade in
                alpha = step_progress
                canvas[part_mask] = cv2.addWeighted(original, alpha, background_paper, 1.0 - alpha, 0)[part_mask]

            else:  # Instant Appearance
                canvas[part_mask] = original[part_mask]

    return canvas


def build_gif(frames, duration=60):
    buf = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(buf, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0)
    return buf.getvalue()


# ============================================================
# STREAMLIT INTERFACE
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

        parts, background_paper = segment_calligraphy_elements(img)

        st.sidebar.header("⚙️ Sequence & Animation Timeline")
        assignments = {}

        if parts:
            st.sidebar.write("Set appearance order and entry effects:")

            for idx, part in enumerate(parts):
                col_s, col_e = st.sidebar.columns([1, 2])
                
                # Provide smart defaults based on spatial order (N -> O -> Sword)
                default_step = idx + 1
                default_effect = "Sword Penetration" if idx == 1 else "Reveal / Write"

                with col_s:
                    step = st.number_input(f"Part {idx+1} Step", min_value=1, max_value=10, value=default_step, key=f"step_{idx}")
                with col_e:
                    effect = st.selectbox(
                        f"Effect",
                        ["Reveal / Write", "Sword Penetration", "Fade In", "Instant"],
                        index=["Reveal / Write", "Sword Penetration", "Fade In", "Instant"].index(default_effect),
                        key=f"effect_{idx}",
                    )

                assignments[idx] = {"step": step, "effect": effect}
        else:
            st.sidebar.warning("No distinct ink strokes detected.")

        total_frames = st.sidebar.slider("Total Animation Duration (Frames)", 18, 60, 36)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Artwork")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Components & Clean Paper")
            preview = img.copy()
            for idx, p in enumerate(parts):
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(preview, f"P{idx+1}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")

        if st.button("✨ Render Sequential Animation", type="primary", disabled=(len(parts) == 0)):
            with st.spinner("Erasing artwork to blank paper & rendering sequential timeline..."):
                frames = []
                for i in range(total_frames):
                    frame = render_sequential_frame(img, background_paper, parts, assignments, i, total_frames)
                    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(Image.fromarray(rgb_frame))

                gif_bytes = build_gif(frames, duration=70)

            st.subheader("🎬 Final Sequential Penetration GIF")
            st.image(gif_bytes, width=450)

            st.download_button(
                "Download Animation GIF",
                data=gif_bytes,
                file_name="sequential_sword_calligraphy.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
