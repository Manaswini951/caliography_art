import io
import math
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Calligraphy Writing & Sword Animator",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✒️ Calligraphy Writing & Sword Animator")

st.write(
    "Upload your calligraphy artwork. The app starts from a pure-white sheet, "
    "draws the selected components progressively like handwriting, and can "
    "animate weapons entering or penetrating the finished letter."
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


def create_ink_mask(image):
    """
    Creates a mask of the foreground artwork.

    Designed to preserve:
    - black ink
    - dark colors
    - colored artwork
    - yellow/gold
    - moderately bright colored strokes

    while rejecting a normal white paper background.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Basic darkness detection.
    dark_mask = cv2.inRange(
        gray,
        0,
        235,
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Colored pixels.
    color_mask = (
        (saturation > 35)
        & (value < 250)
    ).astype(np.uint8) * 255

    # Yellow / gold preservation.
    yellow_mask = cv2.inRange(
        hsv,
        np.array([10, 35, 40]),
        np.array([40, 255, 255]),
    )

    combined = cv2.bitwise_or(
        dark_mask,
        color_mask,
    )

    combined = cv2.bitwise_or(
        combined,
        yellow_mask,
    )

    # Remove tiny isolated camera/scanner noise.
    kernel_small = np.ones((3, 3), np.uint8)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel_small,
    )

    # Connect tiny gaps inside strokes.
    kernel_close = np.ones((3, 3), np.uint8)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel_close,
    )

    return combined


def segment_components(image):
    """
    Finds individual connected artwork components.
    """

    mask = create_ink_mask(image)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    h, w = image.shape[:2]

    min_area = max(
        40,
        int(h * w * 0.00015),
    )

    components = []

    for label in range(1, num_labels):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])

        if cw < 3 or ch < 3:
            continue

        component_mask = (
            labels == label
        )

        components.append(
            {
                "id": label,
                "mask": component_mask,
                "bbox": (
                    x,
                    y,
                    x + cw - 1,
                    y + ch - 1,
                ),
                "center": (
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ),
                "area": area,
            }
        )

    # Sort from left to right.
    components.sort(
        key=lambda p: (
            p["bbox"][0],
            p["bbox"][1],
        )
    )

    return components


# ============================================================
# SKELETONIZATION
# ============================================================

def morphological_skeleton(binary):
    """
    OpenCV-only morphological skeletonization.

    Does not require cv2.ximgproc.
    """

    binary = (
        binary.astype(np.uint8) * 255
    )

    skeleton = np.zeros_like(binary)

    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3),
    )

    current = binary.copy()

    max_iterations = max(
        20,
        int(
            max(
                binary.shape[0],
                binary.shape[1],
            )
        ),
    )

    for _ in range(max_iterations):

        eroded = cv2.erode(
            current,
            element,
        )

        opened = cv2.morphologyEx(
            eroded,
            cv2.MORPH_OPEN,
            element,
        )

        temp = cv2.subtract(
            eroded,
            opened,
        )

        skeleton = cv2.bitwise_or(
            skeleton,
            temp,
        )

        current = eroded

        if cv2.countNonZero(current) == 0:
            break

    return skeleton > 0


# ============================================================
# SKELETON PATH EXTRACTION
# ============================================================

def skeleton_neighbors(point, skeleton):
    """
    Returns 8-connected neighboring skeleton pixels.
    """

    y, x = point

    h, w = skeleton.shape

    neighbors = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):

            if dx == 0 and dy == 0:
                continue

            ny = y + dy
            nx = x + dx

            if (
                0 <= ny < h
                and 0 <= nx < w
                and skeleton[ny, nx]
            ):
                neighbors.append(
                    (ny, nx)
                )

    return neighbors


def find_skeleton_endpoints(skeleton):
    endpoints = []

    ys, xs = np.where(skeleton)

    for y, x in zip(ys, xs):

        neighbors = skeleton_neighbors(
            (y, x),
            skeleton,
        )

        if len(neighbors) == 1:
            endpoints.append(
                (int(y), int(x))
            )

    return endpoints


def choose_start_point(
    skeleton,
    direction="Auto",
):
    """
    Chooses a natural start point.

    If endpoints exist, chooses an endpoint based
    on the requested direction.
    """

    endpoints = find_skeleton_endpoints(
        skeleton
    )

    ys, xs = np.where(skeleton)

    if len(xs) == 0:
        return None

    if len(endpoints) == 0:

        points = list(
            zip(
                ys.astype(int),
                xs.astype(int),
            )
        )

        if direction == "Top -> Bottom":
            return min(
                points,
                key=lambda p: p[0],
            )

        if direction == "Bottom -> Top":
            return max(
                points,
                key=lambda p: p[0],
            )

        if direction == "Left -> Right":
            return min(
                points,
                key=lambda p: p[1],
            )

        if direction == "Right -> Left":
            return max(
                points,
                key=lambda p: p[1],
            )

        return points[0]

    if direction == "Top -> Bottom":
        return min(
            endpoints,
            key=lambda p: (p[0], p[1]),
        )

    if direction == "Bottom -> Top":
        return max(
            endpoints,
            key=lambda p: (p[0], -p[1]),
        )

    if direction == "Left -> Right":
        return min(
            endpoints,
            key=lambda p: (p[1], p[0]),
        )

    if direction == "Right -> Left":
        return max(
            endpoints,
            key=lambda p: (p[1], -p[0]),
        )

    # Auto:
    # choose upper-left-ish endpoint.
    return min(
        endpoints,
        key=lambda p: (
            p[0] + p[1],
        ),
    )


def order_skeleton_path(
    skeleton,
    start,
):
    """
    Follows the skeleton from the selected
    starting point.

    A greedy traversal is used so the path follows
    the actual shape instead of a rectangular wipe.
    """

    if start is None:
        return []

    remaining = set(
        zip(
            *np.where(skeleton)
        )
    )

    if not remaining:
        return []

    # Make sure start belongs to skeleton.
    if start not in remaining:

        start = min(
            remaining,
            key=lambda p: (
                (p[0] - start[0]) ** 2
                + (p[1] - start[1]) ** 2
            ),
        )

    path = [start]

    remaining.remove(start)

    current = start

    max_points = min(
        len(path) + len(remaining),
        20000,
    )

    while remaining and len(path) < max_points:

        cy, cx = current

        # Prefer nearby skeleton points.
        candidates = []

        for point in remaining:

            py, px = point

            distance = (
                (py - cy) ** 2
                + (px - cx) ** 2
            )

            candidates.append(
                (
                    distance,
                    point,
                )
            )

        candidates.sort(
            key=lambda x: x[0]
        )

        # Usually use nearest point.
        _, next_point = candidates[0]

        path.append(next_point)

        remaining.remove(
            next_point
        )

        current = next_point

        # If path gets disconnected, jump to the
        # closest remaining region.
        if remaining:

            near_count = 0

            for point in remaining:

                py, px = point

                d = math.sqrt(
                    (py - cy) ** 2
                    + (px - cx) ** 2
                )

                if d < 8:
                    near_count += 1

            if near_count == 0:

                _, jump_point = min(
                    (
                        (
                            (p[0] - cy) ** 2
                            + (p[1] - cx) ** 2,
                            p,
                        )
                        for p in remaining
                    ),
                    key=lambda x: x[0],
                )

                path.append(
                    jump_point
                )

                remaining.remove(
                    jump_point
                )

                current = jump_point

    return path


def smooth_path(path, smoothing_window=7):
    """
    Smooths skeleton coordinates.
    """

    if len(path) < 3:
        return path

    pts = np.array(
        [
            [p[1], p[0]]
            for p in path
        ],
        dtype=np.float32,
    )

    window = max(
        3,
        int(smoothing_window),
    )

    if window % 2 == 0:
        window += 1

    kernel = np.ones(
        window,
        dtype=np.float32,
    ) / float(window)

    x = np.convolve(
        pts[:, 0],
        kernel,
        mode="same",
    )

    y = np.convolve(
        pts[:, 1],
        kernel,
        mode="same",
    )

    half = window // 2

    x[:half] = pts[:half, 0]
    x[-half:] = pts[-half:, 0]

    y[:half] = pts[:half, 1]
    y[-half:] = pts[-half:, 1]

    result = []

    for px, py in zip(x, y):
        result.append(
            (
                int(round(py)),
                int(round(px)),
            )
        )

    return result


# ============================================================
# BUILD COMPONENT WRITING PATH
# ============================================================

def build_component_path(
    component,
    direction,
):
    """
    Generates a writing path for one component.
    """

    x1, y1, x2, y2 = component["bbox"]

    component_mask = component["mask"]

    # Crop.
    local = component_mask[
        y1:y2 + 1,
        x1:x2 + 1,
    ]

    # Skeleton.
    skeleton = morphological_skeleton(
        local
    )

    # If skeleton failed, fall back to a
    # boundary/center approximation.
    if cv2.countNonZero(
        skeleton.astype(np.uint8)
    ) < 2:

        ys, xs = np.where(local)

        if len(xs) == 0:
            return []

        order = np.argsort(
            xs + ys
        )

        path = [
            (
                int(ys[i] + y1),
                int(xs[i] + x1),
            )
            for i in order[
                ::max(1, len(order) // 300)
            ]
        ]

        return path

    start = choose_start_point(
        skeleton,
        direction,
    )

    local_path = order_skeleton_path(
        skeleton,
        start,
    )

    local_path = smooth_path(
        local_path,
        smoothing_window=7,
    )

    global_path = [
        (
            y + y1,
            x + x1,
        )
        for y, x in local_path
    ]

    return global_path


# ============================================================
# BUILD REVEAL MASKS
# ============================================================

def build_path_reveal_mask(
    component_mask,
    path,
):
    """
    Creates a mapping between path progress and
    component pixels.

    Each artwork pixel receives a value indicating
    how far along the writing path it is.

    This is what allows us to reveal ink progressively.
    """

    h, w = component_mask.shape

    ys, xs = np.where(
        component_mask
    )

    if len(xs) == 0:
        return np.zeros(
            (h, w),
            dtype=np.float32,
        )

    if not path:
        result = np.zeros(
            (h, w),
            dtype=np.float32,
        )
        result[component_mask] = 1.0
        return result

    path_arr = np.array(
        [
            [p[1], p[0]]
            for p in path
        ],
        dtype=np.float32,
    )

    # To keep processing manageable for large images,
    # downsample very long paths.
    max_path_points = 500

    if len(path_arr) > max_path_points:

        indices = np.linspace(
            0,
            len(path_arr) - 1,
            max_path_points,
        ).astype(int)

        path_arr = path_arr[
            indices
        ]

    # Distance of every ink pixel to nearest
    # point on the writing path.
    progress = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    # Chunked calculation avoids excessive RAM use.
    chunk_size = 5000

    pixel_points = np.column_stack(
        (
            xs,
            ys,
        )
    ).astype(np.float32)

    for start in range(
        0,
        len(pixel_points),
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            len(pixel_points),
        )

        chunk = pixel_points[
            start:end
        ]

        distances = np.sqrt(
            (
                (
                    chunk[:, None, :]
                    - path_arr[None, :, :]
                ) ** 2
            ).sum(axis=2)
        )

        nearest = np.argmin(
            distances,
            axis=1,
        )

        progress[
            ys[start:end],
            xs[start:end],
        ] = (
            nearest.astype(
                np.float32
            )
            / max(
                1,
                len(path_arr) - 1,
            )
        )

    # Expand the path index according to
    # actual path geometry.
    path_distances = [0.0]

    for i in range(
        1,
        len(path_arr),
    ):

        dx = (
            path_arr[i, 0]
            - path_arr[i - 1, 0]
        )

        dy = (
            path_arr[i, 1]
            - path_arr[i - 1, 1]
        )

        path_distances.append(
            path_distances[-1]
            + math.sqrt(
                dx * dx + dy * dy
            )
        )

    total_distance = max(
        path_distances[-1],
        1.0,
    )

    # Convert nearest path index to actual
    # normalized distance.
    normalized = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    valid_y, valid_x = np.where(
        component_mask
    )

    nearest_indices = np.rint(
        progress[
            valid_y,
            valid_x,
        ]
        * (
            len(path_arr) - 1
        )
    ).astype(int)

    nearest_indices = np.clip(
        nearest_indices,
        0,
        len(path_arr) - 1,
    )

    normalized[
        valid_y,
        valid_x,
    ] = np.array(
        [
            path_distances[i]
            / total_distance
            for i in nearest_indices
        ],
        dtype=np.float32,
    )

    return normalized


# ============================================================
# COMPONENT PREPARATION
# ============================================================

def prepare_component(component, image, direction):
    """
    Prepares a component for animation.
    """

    path = build_component_path(
        component,
        direction,
    )

    progress_map = build_path_reveal_mask(
        component["mask"],
        path,
    )

    component["path"] = path
    component["progress_map"] = progress_map

    return component


# ============================================================
# WRITING ANIMATION
# ============================================================

def render_writing_component(
    canvas,
    original,
    component,
    progress,
):
    """
    Reveals artwork progressively according to
    the reconstructed writing path.
    """

    mask = component["mask"]
    progress_map = component[
        "progress_map"
    ]

    # Slightly soften the transition so the
    # writing does not appear pixel-sharp.
    reveal_width = 0.035

    threshold = (
        progress
        + reveal_width
    )

    active = (
        mask
        & (
            progress_map
            <= threshold
        )
    )

    canvas[active] = original[
        active
    ]

    return active


# ============================================================
# PEN TIP
# ============================================================

def draw_pen_tip(
    canvas,
    component,
    progress,
    color=(40, 40, 40),
):
    """
    Draws a small pen/calligraphy nib following
    the reconstructed writing path.
    """

    path = component.get(
        "path",
        [],
    )

    if not path:
        return

    index = int(
        progress
        * (
            len(path) - 1
        )
    )

    index = max(
        0,
        min(
            index,
            len(path) - 1,
        ),
    )

    y, x = path[index]

    # Outer nib.
    cv2.circle(
        canvas,
        (x, y),
        5,
        color,
        -1,
        lineType=cv2.LINE_AA,
    )

    # Small highlight.
    cv2.circle(
        canvas,
        (x - 1, y - 1),
        2,
        (255, 255, 255),
        -1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# SWORD UTILITIES
# ============================================================

def direction_vector(direction):
    vectors = {
        "Top": (0, -1),
        "Bottom": (0, 1),
        "Left": (-1, 0),
        "Right": (1, 0),
        "Top-Left": (-1, -1),
        "Top-Right": (1, -1),
        "Bottom-Left": (-1, 1),
        "Bottom-Right": (1, 1),
    }

    return vectors.get(
        direction,
        (1, 0),
    )


def normalize_vector(x, y):
    length = math.sqrt(
        x * x + y * y
    )

    if length == 0:
        return 1.0, 0.0

    return (
        x / length,
        y / length,
    )


def shift_mask(mask, dx, dy):
    """
    Shifts a binary mask.
    """

    h, w = mask.shape

    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )

    shifted = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return shifted > 0


def shift_image(
    image,
    dx,
    dy,
):
    h, w = image.shape[:2]

    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )

    shifted = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return shifted


def render_sword(
    canvas,
    original,
    component,
    progress,
    entry_direction,
    penetration,
    angle_degrees,
):
    """
    Moves the selected weapon component from outside
    the page toward its original position.

    The final position corresponds to the original
    artwork position.

    This makes the weapon appear to physically enter
    the letter.
    """

    mask = component["mask"]

    h, w = canvas.shape[:2]

    x1, y1, x2, y2 = component["bbox"]

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    vw, vh = direction_vector(
        entry_direction
    )

    vw, vh = normalize_vector(
        vw,
        vh,
    )

    # Diagonal / angle adjustment.
    angle_rad = math.radians(
        angle_degrees
    )

    rotated_x = (
        vw * math.cos(angle_rad)
        - vh * math.sin(angle_rad)
    )

    rotated_y = (
        vw * math.sin(angle_rad)
        + vh * math.cos(angle_rad)
    )

    rotated_x, rotated_y = normalize_vector(
        rotated_x,
        rotated_y,
    )

    # Distance outside the artwork.
    object_size = max(
        x2 - x1,
        y2 - y1,
    )

    entry_distance = (
        object_size * 1.6
        + 80
    )

    # At progress 0:
    # sword is outside.
    #
    # At progress 1:
    # sword reaches original location.
    travel = (
        1.0 - progress
    ) * entry_distance

    dx = (
        rotated_x
        * travel
    )

    dy = (
        rotated_y
        * travel
    )

    # Apply penetration extension.
    #
    # penetration = 0:
    # stop at normal location
    #
    # penetration = 100:
    # continue through the target.
    penetration_distance = (
        penetration
        / 100.0
        * object_size
        * 0.75
    )

    if progress > 0.72:
        pen_phase = (
            progress - 0.72
        ) / 0.28

        dx += (
            rotated_x
            * penetration_distance
            * pen_phase
        )

        dy += (
            rotated_y
            * penetration_distance
            * pen_phase
        )

    shifted = shift_image(
        original,
        int(round(dx)),
        int(round(dy)),
    )

    shifted_mask = shift_mask(
        mask,
        int(round(dx)),
        int(round(dy)),
    )

    canvas[shifted_mask] = shifted[
        shifted_mask
    ]


# ============================================================
# STATIC COMPONENT
# ============================================================

def render_static_component(
    canvas,
    original,
    component,
):
    mask = component["mask"]

    canvas[mask] = original[
        mask
    ]


# ============================================================
# EASING
# ============================================================

def ease_in_out(t):
    """
    Smooth cinematic motion.
    """

    t = max(
        0.0,
        min(
            1.0,
            t,
        ),
    )

    return (
        t
        * t
        * (
            3.0
            - 2.0 * t
        )
    )


def ease_out(t):
    t = max(
        0.0,
        min(
            1.0,
            t,
        ),
    )

    return 1.0 - (
        1.0 - t
    ) ** 3


# ============================================================
# ANIMATION FRAME
# ============================================================

def render_animation_frame(
    original,
    white_canvas,
    components,
    assignments,
    frame_index,
    total_frames,
    show_pen,
):
    """
    Main animation compositor.
    """

    canvas = white_canvas.copy()

    # Convert global frame to normalized time.
    if total_frames <= 1:
        global_progress = 1.0
    else:
        global_progress = (
            frame_index
            / float(
                total_frames - 1
            )
        )

    # --------------------------------------------------------
    # FIRST PASS
    # Static elements
    # --------------------------------------------------------

    for idx, component in enumerate(
        components
    ):

        config = assignments.get(
            idx,
            {},
        )

        role = config.get(
            "role",
            "Writing",
        )

        if role == "Static":

            render_static_component(
                canvas,
                original,
                component,
            )

    # --------------------------------------------------------
    # SECOND PASS
    # Writing
    # --------------------------------------------------------

    for idx, component in enumerate(
        components
    ):

        config = assignments.get(
            idx,
            {},
        )

        role = config.get(
            "role",
            "Writing",
        )

        if role != "Writing":
            continue

        start = config.get(
            "start",
            0.0,
        )

        duration = config.get(
            "duration",
            1.0,
        )

        if global_progress < start:
            continue

        local = (
            global_progress
            - start
        ) / max(
            duration,
            0.001,
        )

        local = max(
            0.0,
            min(
                1.0,
                local,
            ),
        )

        local = ease_in_out(
            local
        )

        render_writing_component(
            canvas,
            original,
            component,
            local,
        )

        if (
            show_pen
            and local > 0.0
            and local < 1.0
        ):
            draw_pen_tip(
                canvas,
                component,
                local,
            )

    # --------------------------------------------------------
    # THIRD PASS
    # Sword / weapon
    # --------------------------------------------------------

    for idx, component in enumerate(
        components
    ):

        config = assignments.get(
            idx,
            {},
        )

        role = config.get(
            "role",
            "Writing",
        )

        if role != "Sword":
            continue

        start = config.get(
            "start",
            0.0,
        )

        duration = config.get(
            "duration",
            1.0,
        )

        if global_progress < start:
            continue

        local = (
            global_progress
            - start
        ) / max(
            duration,
            0.001,
        )

        local = max(
            0.0,
            min(
                1.0,
                local,
            ),
        )

        local = ease_out(
            local
        )

        render_sword(
            canvas,
            original,
            component,
            local,
            config.get(
                "entry_direction",
                "Right",
            ),
            config.get(
                "penetration",
                25,
            ),
            config.get(
                "angle",
                0,
            ),
        )

    return canvas


# ============================================================
# GIF GENERATION
# ============================================================

def build_gif(
    frames,
    duration=60,
):
    """
    Builds optimized GIF.
    """

    if not frames:
        return b""

    prepared = []

    for frame in frames:

        rgb = frame.convert(
            "RGB"
        )

        prepared.append(
            rgb.convert(
                "P",
                palette=Image.ADAPTIVE,
                colors=256,
            )
        )

    buf = io.BytesIO()

    prepared[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    return buf.getvalue()


# ============================================================
# PREVIEW COMPONENT BOXES
# ============================================================

def create_component_preview(
    image,
    components,
    assignments=None,
):
    preview = image.copy()

    for idx, component in enumerate(
        components
    ):

        x1, y1, x2, y2 = component[
            "bbox"
        ]

        if assignments:

            role = assignments.get(
                idx,
                {},
            ).get(
                "role",
                "Writing",
            )

            if role == "Sword":
                rectangle_color = (
                    255,
                    0,
                    0,
                )
            elif role == "Static":
                rectangle_color = (
                    128,
                    128,
                    128,
                )
            else:
                rectangle_color = (
                    0,
                    200,
                    0,
                )

        else:
            rectangle_color = (
                0,
                255,
                0,
            )

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            rectangle_color,
            2,
        )

        cv2.putText(
            preview,
            f"P{idx + 1}",
            (
                x1,
                max(
                    20,
                    y1 - 7,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    return preview


# ============================================================
# SIDEBAR HELP
# ============================================================

def role_description(role):
    if role == "Writing":
        return "✒️ Draw this component progressively."

    if role == "Sword":
        return "⚔️ Move this component from outside into the artwork."

    return "📌 Show this component immediately."


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Calligraphy Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8,
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if img is None:

            st.error(
                "Invalid image format."
            )

            st.stop()

        img = resize_image(
            img,
            max_size=900,
        )

        h, w = img.shape[:2]

        # ----------------------------------------------------
        # SEGMENT
        # ----------------------------------------------------

        with st.spinner(
            "Detecting calligraphy components..."
        ):

            parts = segment_components(
                img
            )

        # Pure white canvas.
        white_canvas = np.full_like(
            img,
            255,
            dtype=np.uint8,
        )

        if not parts:

            st.error(
                "No artwork components were detected."
            )

            st.stop()

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        st.sidebar.header(
            "⚙️ Animation Controls"
        )

        st.sidebar.subheader(
            "Global Settings"
        )

        writing_default_direction = st.sidebar.selectbox(
            "Default Writing Direction",
            [
                "Top -> Bottom",
                "Bottom -> Top",
                "Left -> Right",
                "Right -> Left",
            ],
            index=2,
        )

        show_pen = st.sidebar.checkbox(
            "Show Pen / Brush Tip",
            value=True,
        )

        total_frames = st.sidebar.slider(
            "Animation Frames",
            min_value=30,
            max_value=120,
            value=60,
            step=5,
        )

        gif_duration = st.sidebar.slider(
            "Frame Duration (ms)",
            min_value=30,
            max_value=150,
            value=60,
            step=10,
        )

        # ----------------------------------------------------
        # PREPARE COMPONENTS
        # ----------------------------------------------------

        with st.spinner(
            "Building handwriting paths..."
        ):

            for component in parts:

                component[
                    "direction"
                ] = writing_default_direction

                prepare_component(
                    component,
                    img,
                    writing_default_direction,
                )

        # ----------------------------------------------------
        # COMPONENT ASSIGNMENTS
        # ----------------------------------------------------

        st.sidebar.markdown("---")

        st.sidebar.subheader(
            "🎬 Component Animation"
        )

        st.sidebar.caption(
            "Writing components are drawn progressively. "
            "Sword components enter from outside the artwork."
        )

        assignments = {}

        for idx, component in enumerate(
            parts
        ):

            x1, y1, x2, y2 = component[
                "bbox"
            ]

            st.sidebar.markdown(
                f"### P{idx + 1}"
            )

            st.sidebar.caption(
                f"Size: {x2 - x1 + 1} × "
                f"{y2 - y1 + 1} px"
            )

            # Intelligent initial default:
            # first components writing,
            # small/medium component can be manually changed.
            default_role = (
                "Writing"
            )

            role = st.sidebar.selectbox(
                "Role",
                [
                    "Writing",
                    "Sword",
                    "Static",
                ],
                index=[
                    "Writing",
                    "Sword",
                    "Static",
                ].index(
                    default_role
                ),
                key=f"role_{idx}",
            )

            st.sidebar.caption(
                role_description(role)
            )

            if role == "Writing":

                direction = st.sidebar.selectbox(
                    "Writing Start",
                    [
                        "Top -> Bottom",
                        "Bottom -> Top",
                        "Left -> Right",
                        "Right -> Left",
                    ],
                    index=[
                        "Top -> Bottom",
                        "Bottom -> Top",
                        "Left -> Right",
                        "Right -> Left",
                    ].index(
                        writing_default_direction
                    ),
                    key=f"writing_dir_{idx}",
                )

                start_percent = st.sidebar.slider(
                    "Start Time %",
                    0,
                    90,
                    int(
                        idx
                        * (
                            60
                            / max(
                                1,
                                len(parts),
                            )
                        )
                    ),
                    key=f"start_write_{idx}",
                )

                duration_percent = st.sidebar.slider(
                    "Writing Duration %",
                    5,
                    100,
                    35,
                    key=f"duration_write_{idx}",
                )

                # Rebuild path if direction changed.
                if (
                    direction
                    != component.get(
                        "direction"
                    )
                ):

                    component[
                        "direction"
                    ] = direction

                    prepare_component(
                        component,
                        img,
                        direction,
                    )

                assignments[
                    idx
                ] = {
                    "role": "Writing",
                    "start": start_percent
                    / 100.0,
                    "duration": duration_percent
                    / 100.0,
                }

            elif role == "Sword":

                entry_direction = st.sidebar.selectbox(
                    "Sword Entry",
                    [
                        "Top",
                        "Bottom",
                        "Left",
                        "Right",
                        "Top-Left",
                        "Top-Right",
                        "Bottom-Left",
                        "Bottom-Right",
                    ],
                    index=3,
                    key=f"sword_entry_{idx}",
                )

                start_percent = st.sidebar.slider(
                    "Sword Start %",
                    0,
                    95,
                    55,
                    key=f"sword_start_{idx}",
                )

                duration_percent = st.sidebar.slider(
                    "Sword Travel %",
                    5,
                    70,
                    25,
                    key=f"sword_duration_{idx}",
                )

                penetration = st.sidebar.slider(
                    "Penetration Depth %",
                    0,
                    100,
                    25,
                    key=f"penetration_{idx}",
                )

                angle = st.sidebar.slider(
                    "Sword Angle",
                    -180,
                    180,
                    0,
                    key=f"sword_angle_{idx}",
                )

                assignments[
                    idx
                ] = {
                    "role": "Sword",
                    "start": start_percent
                    / 100.0,
                    "duration": duration_percent
                    / 100.0,
                    "entry_direction": entry_direction,
                    "penetration": penetration,
                    "angle": angle,
                }

            else:

                assignments[
                    idx
                ] = {
                    "role": "Static",
                    "start": 0.0,
                    "duration": 1.0,
                }

            st.sidebar.markdown(
                "---"
            )

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.subheader(
                "🖼️ Original Artwork"
            )

            st.image(
                cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with col2:

            st.subheader(
                "🔍 Detected Components"
            )

            preview = create_component_preview(
                img,
                parts,
                assignments,
            )

            st.image(
                cv2.cvtColor(
                    preview,
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        writing_count = sum(
            1
            for a in assignments.values()
            if a.get("role")
            == "Writing"
        )

        sword_count = sum(
            1
            for a in assignments.values()
            if a.get("role")
            == "Sword"
        )

        static_count = sum(
            1
            for a in assignments.values()
            if a.get("role")
            == "Static"
        )

        st.info(
            f"Detected {len(parts)} components — "
            f"✒️ Writing: {writing_count} | "
            f"⚔️ Sword: {sword_count} | "
            f"📌 Static: {static_count}"
        )

        # ----------------------------------------------------
        # RENDER
        # ----------------------------------------------------

        st.markdown("---")

        if st.button(
            "✨ Render Handwritten Calligraphy Animation",
            type="primary",
        ):

            progress_bar = st.progress(
                0
            )

            status = st.empty()

            frames = []

            for frame_index in range(
                total_frames
            ):

                status.write(
                    f"Rendering frame "
                    f"{frame_index + 1} / "
                    f"{total_frames}"
                )

                frame = render_animation_frame(
                    img,
                    white_canvas,
                    parts,
                    assignments,
                    frame_index,
                    total_frames,
                    show_pen,
                )

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frames.append(
                    Image.fromarray(
                        np.ascontiguousarray(
                            rgb_frame
                        )
                    )
                )

                progress_bar.progress(
                    (
                        frame_index + 1
                    )
                    / total_frames
                )

            status.write(
                "Building GIF..."
            )

            gif_bytes = build_gif(
                frames,
                duration=gif_duration,
            )

            progress_bar.empty()

            status.success(
                "Animation completed!"
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader(
                "🎬 Animated Result"
            )

            st.image(
                gif_bytes,
                width=600,
            )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.download_button(
                "⬇️ Download GIF",
                data=gif_bytes,
                file_name=(
                    "calligraphy_"
                    "handwriting_sword.gif"
                ),
                mime="image/gif",
            )

    except Exception as e:

        st.error(
            f"Error encountered: {e}"
        )

        st.exception(e) look at the updated script we were working earlier ... is there any more requirement to be added in requirement.txt
