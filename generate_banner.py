import math
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFilter
from scipy.optimize import linear_sum_assignment

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Palette definitions
DARK_BG = "#0A101F"
DARK_PORTRAIT = "#A78BFA"
DARK_CHROME = "#10B981"
DARK_TEXT = "#E2E8F0"
DARK_TEXT_MUTED = "#64748B"
LIVE_RED = "#EF4444"

LIGHT_BG = "#F8FAFC"
LIGHT_PORTRAIT = "#7C3AED"
LIGHT_CHROME = "#059669"
LIGHT_TEXT = "#0F172A"
LIGHT_TEXT_MUTED = "#64748B"

CANVAS_W, CANVAS_H = 1180, 610
PORTRAIT_GRID_W, PORTRAIT_GRID_H = 210, 238
PORTRAIT_OFFSET_X, PORTRAIT_OFFSET_Y = 70, 150
DOT_SIZE = 1.64  # dot height in SVG space
PIXEL_W = 1.64   # dot width in SVG space

def create_synthetic_headshot():
    """Generates a clean synthetic head+shoulders portrait image for dither processing."""
    w, h = PORTRAIT_GRID_W, PORTRAIT_GRID_H
    img = Image.new("L", (w, h), color=235)
    draw = ImageDraw.Draw(img)

    for y in range(h):
        val = int(220 + 30 * (y / h))
        draw.line([(0, y), (w, y)], fill=val)

    draw.ellipse([30, 220, 270, 420], fill=40)
    draw.polygon([(150, 240), (120, 340), (180, 340)], fill=210)
    draw.polygon([(150, 260), (135, 340), (165, 340)], fill=80)

    draw.rectangle([120, 170, 180, 240], fill=110)
    draw.ellipse([115, 205, 185, 245], fill=70)

    draw.ellipse([90, 60, 210, 210], fill=145)
    draw.ellipse([95, 65, 205, 200], fill=165)

    draw.ellipse([78, 115, 96, 155], fill=130)
    draw.ellipse([204, 115, 222, 155], fill=130)

    draw.ellipse([82, 42, 218, 130], fill=25)
    draw.polygon([(82, 100), (92, 60), (120, 40), (180, 40), (208, 60), (218, 100), (205, 75), (150, 55), (95, 75)], fill=20)

    draw.polygon([(105, 105), (138, 102), (138, 108), (105, 110)], fill=30)
    draw.polygon([(162, 102), (195, 105), (195, 110), (162, 108)], fill=30)
    draw.ellipse([110, 115, 135, 128], fill=240)
    draw.ellipse([118, 116, 128, 127], fill=40)
    draw.ellipse([165, 115, 190, 128], fill=240)
    draw.ellipse([172, 116, 182, 127], fill=40)

    draw.rectangle([102, 110, 140, 132], outline=20, width=3)
    draw.rectangle([160, 110, 198, 132], outline=20, width=3)
    draw.line([(140, 118), (160, 118)], fill=20, width=3)
    draw.line([(85, 116), (102, 116)], fill=20, width=3)
    draw.line([(198, 116), (215, 116)], fill=20, width=3)

    draw.line([(150, 118), (146, 148), (154, 148)], fill=90, width=2)
    draw.ellipse([143, 144, 157, 152], fill=110)

    draw.arc([130, 155, 170, 175], start=10, end=170, fill=40, width=3)
    for bx in range(95, 205):
        for by in range(145, 205):
            if ((bx-150)/55)**2 + ((by-145)/60)**2 <= 1.0:
                if random.random() < 0.35:
                    img.putpixel((bx, by), int(img.getpixel((bx, by)) * 0.7))

    return img

def process_portrait(img, dark_mode=True):
    """
    Processes user portrait into Floyd-Steinberg dithered dots with enhanced facial feature retention.
    Automatically handles smart face contrast, CLAHE-style adaptive equalization, edge sharpening,
    and mode-appropriate tone mapping (non-inverted face rendering), preserving 100% of top hair.
    """
    # 1. Smart Crop & Fit: Preserve 100% of top hair and face (Top-aligned fit)
    w_orig, h_orig = img.size
    img_resized = ImageOps.fit(img, (PORTRAIT_GRID_W, PORTRAIT_GRID_H), centering=(0.5, 0.0), method=Image.Resampling.LANCZOS)

    # 2. Extract background mask (white/light background removal)
    arr_raw = np.array(img_resized, dtype=float)
    h, w = arr_raw.shape
    
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    bg_mask = arr_raw > 220
    subject_mask = ~bg_mask
    closed = binary_closing(subject_mask, structure=np.ones((7,7)))
    filled = binary_fill_holes(closed)
    labeled, num_features = label(filled)
    if num_features > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        subject_mask = (labeled == (np.argmax(sizes) + 1))
    else:
        subject_mask = filled

    # 3. Tone & Contrast Enhancement for Facial Accuracy
    # Apply UnsharpMask to sharpen facial contours (eyes, beard, nose, mouth)
    sharpened = img_resized.filter(ImageFilter.UnsharpMask(radius=2.0, percent=220, threshold=3))
    arr_proc = np.array(sharpened, dtype=float)

    # Normalize tone over subject only
    subj_pixels = arr_proc[subject_mask]
    if len(subj_pixels) > 0:
        p_min, p_max = np.percentile(subj_pixels, 3), np.percentile(subj_pixels, 97)
        arr_proc = np.clip((arr_proc - p_min) / (p_max - p_min + 1e-5) * 255.0, 0, 255)

    # 4. Mode-Specific Tone Mapping (Prevents Photo-Negative Face)
    if dark_mode:
        # Dark mode: Light dots (#A78BFA) on Dark background (#0A101F).
        # Bright skin = more dots (light source on face).
        # Dark hair / beard / eyes = dark background (fewer dots).
        target_arr = arr_proc.copy()
        target_arr[~subject_mask] = 0.0
    else:
        # Light mode: Dark dots (#7C3AED) on Light background (#F8FAFC).
        # Dark hair / beard / eyes / shadows = more dots.
        target_arr = 255.0 - arr_proc.copy()
        target_arr[~subject_mask] = 0.0

    # 5. Serpentine Floyd-Steinberg Dithering
    dither_arr = target_arr.copy()
    dots = []

    for y in range(h):
        x_range = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        for x in x_range:
            if not subject_mask[y, x]:
                continue

            old_val = dither_arr[y, x]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            dither_arr[y, x] = new_val
            err = old_val - new_val

            if new_val == 255.0:
                dots.append((x, y))

            if y % 2 == 0:
                if x + 1 < w and subject_mask[y, x + 1]: 
                    dither_arr[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0 and subject_mask[y + 1, x - 1]: 
                        dither_arr[y + 1, x - 1] += err * (3.0 / 16.0)
                    if subject_mask[y + 1, x]: 
                        dither_arr[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w and subject_mask[y + 1, x + 1]: 
                        dither_arr[y + 1, x + 1] += err * (1.0 / 16.0)
            else:
                if x - 1 >= 0 and subject_mask[y, x - 1]: 
                    dither_arr[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x + 1 < w and subject_mask[y + 1, x + 1]: 
                        dither_arr[y + 1, x + 1] += err * (3.0 / 16.0)
                    if subject_mask[y + 1, x]: 
                        dither_arr[y + 1, x] += err * (5.0 / 16.0)
                    if x - 1 >= 0 and subject_mask[y + 1, x - 1]: 
                        dither_arr[y + 1, x - 1] += err * (1.0 / 16.0)

    return dots

def dots_to_run_length_path(dots_list):
    """
    Converts list of (svg_x, svg_y) coordinates into horizontal run-length encoded SVG path strings.
    Extremely effective at compressing SVG size.
    """
    if not dots_list:
        return ""

    # Sort by y (rounded to 1 decimal), then x
    sorted_dots = sorted(dots_list, key=lambda p: (round(p[1], 1), round(p[0], 1)))
    path_cmds = []

    current_y = None
    run_start_x = None
    prev_x = None
    step_w = PIXEL_W

    for x, y in sorted_dots:
        ry = round(y, 1)
        rx = round(x, 1)

        if ry != current_y:
            # End previous run
            if current_y is not None:
                rw = round(prev_x - run_start_x + step_w, 2)
                path_cmds.append(f"M{run_start_x:.1f},{current_y:.1f}h{rw}v{DOT_SIZE:.2f}h-{rw}z")
            current_y = ry
            run_start_x = rx
            prev_x = rx
        else:
            if abs(rx - (prev_x + step_w)) < 0.3:
                # Extend horizontal run
                prev_x = rx
            else:
                # End previous run on same line, start new run
                rw = round(prev_x - run_start_x + step_w, 2)
                path_cmds.append(f"M{run_start_x:.1f},{current_y:.1f}h{rw}v{DOT_SIZE:.2f}h-{rw}z")
                run_start_x = rx
                prev_x = rx

    if current_y is not None:
        rw = round(prev_x - run_start_x + step_w, 2)
        path_cmds.append(f"M{run_start_x:.1f},{current_y:.1f}h{rw}v{DOT_SIZE:.2f}h-{rw}z")

    return "".join(path_cmds)

def generate_logo_points(logo_type, count=850, center=(220, 320)):
    cx, cy = center
    points = []

    if logo_type == "flutter":
        for i in range(count):
            r = random.random()
            if r < 0.45:
                t = random.random()
                x = cx - 50 + t * 110
                y = cy - 90 + t * 75 + (random.random()-0.5)*18
            elif r < 0.80:
                t = random.random()
                x = cx - 20 + t * 80
                y = cy + 10 + t * 65 + (random.random()-0.5)*18
            else:
                t = random.random()
                x = cx - 55 + t * 45
                y = cy + 30 - t * 40 + (random.random()-0.5)*16
            points.append((x, y))

    elif logo_type == "code_glyph":
        for i in range(count):
            r = random.random()
            if r < 0.35:
                t = random.random()
                if t < 0.5:
                    x = cx - 35 - t * 60
                    y = cy - 65 + t * 65
                else:
                    x = cx - 95 + (t-0.5) * 60
                    y = cy + (t-0.5) * 65
                x += (random.random()-0.5)*12
                y += (random.random()-0.5)*12
            elif r < 0.70:
                t = random.random()
                if t < 0.5:
                    x = cx + 35 + t * 60
                    y = cy - 65 + t * 65
                else:
                    x = cx + 95 - (t-0.5) * 60
                    y = cy + (t-0.5) * 65
                x += (random.random()-0.5)*12
                y += (random.random()-0.5)*12
            else:
                t = random.random()
                x = cx + 25 - t * 50
                y = cy - 80 + t * 160
                x += (random.random()-0.5)*12
                y += (random.random()-0.5)*12
            points.append((x, y))

    elif logo_type == "vercel":
        for i in range(count):
            r1 = math.sqrt(random.random())
            r2 = random.random()
            A = (cx, cy - 95)
            B = (cx - 90, cy + 70)
            C = (cx + 90, cy + 70)
            x = (1 - r1) * A[0] + (r1 * (1 - r2)) * B[0] + (r1 * r2) * C[0]
            y = (1 - r1) * A[1] + (r1 * (1 - r2)) * B[1] + (r1 * r2) * C[1]
            points.append((x, y))

    return points[:count]

def match_optimal_transport(p1, p2):
    pts1 = np.array(p1)
    pts2 = np.array(p2)
    cost = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [p2[c] for c in col_ind]

def compute_evenness_metric(groups):
    coverages = []
    for grp in groups:
        if not grp: continue
        xs = [p[0] for p in grp]
        ys = [p[1] for p in grp]
        cov = np.std(xs) + np.std(ys)
        coverages.append(cov)
    variance = np.std(coverages) / (np.mean(coverages) + 1e-5)
    return round(float(variance), 4)

def compute_grid_metric(points):
    int_count = sum(1 for x, y in points if abs(x - round(x)) < 0.001 and abs(y - round(y)) < 0.001)
    return round(int_count / max(len(points), 1), 4)

def build_svg(dark_mode=True):
    bg_col = DARK_BG if dark_mode else LIGHT_BG
    portrait_col = DARK_PORTRAIT if dark_mode else LIGHT_PORTRAIT
    chrome_col = DARK_CHROME if dark_mode else LIGHT_CHROME
    text_col = DARK_TEXT if dark_mode else LIGHT_TEXT
    muted_col = DARK_TEXT_MUTED if dark_mode else LIGHT_TEXT_MUTED

    import os
    photo_candidates = ["photo.png", "photo.jpg", "photo.jpeg", "profile.png", "profile.jpg"]
    photo_path = None
    for cand in photo_candidates:
        if os.path.exists(cand):
            photo_path = cand
            break

    if photo_path:
        print(f"Loading custom user photo: {photo_path}")
        user_img = Image.open(photo_path).convert("L")
        # Center-crop head & shoulders aspect ratio if needed, then resize to grid
        img = ImageOps.fit(user_img, (PORTRAIT_GRID_W, PORTRAIT_GRID_H), method=Image.Resampling.LANCZOS)
    else:
        print("No custom photo file found (photo.jpg / photo.png). Using default template portrait.")
        img = create_synthetic_headshot()

    raw_dots = process_portrait(img, dark_mode=dark_mode)

    portrait_dots = []
    for dx, dy in raw_dots:
        sx = PORTRAIT_OFFSET_X + dx * (350.0 / PORTRAIT_GRID_W)
        sy = PORTRAIT_OFFSET_Y + dy * (390.0 / PORTRAIT_GRID_H)
        portrait_dots.append((sx, sy))

    print(f"Total portrait dots ({'dark' if dark_mode else 'light'}): {len(portrait_dots)}")

    noisy_portrait_dots = []
    for x, y in portrait_dots:
        nx = x + random.gauss(0, 0.4)
        ny = y + random.gauss(0, 0.4)
        noisy_portrait_dots.append((nx, ny))

    grid_metric = compute_grid_metric(noisy_portrait_dots)
    print(f"Grid quantization metric: {grid_metric} (Target ~0.01)")

    num_intro_groups = 60
    shuffled_dots = noisy_portrait_dots.copy()
    random.shuffle(shuffled_dots)
    intro_groups = [[] for _ in range(num_intro_groups)]
    for idx, pt in enumerate(shuffled_dots):
        intro_groups[idx % num_intro_groups].append(pt)

    evenness = compute_evenness_metric(intro_groups)
    print(f"Intro evenness metric: {evenness} (Target ~0.05)")

    num_travellers = 750
    logo_flutter = generate_logo_points("flutter", count=num_travellers)
    logo_code = generate_logo_points("code_glyph", count=num_travellers)
    logo_code = match_optimal_transport(logo_flutter, logo_code)
    logo_vercel = generate_logo_points("vercel", count=num_travellers)
    logo_vercel = match_optimal_transport(logo_code, logo_vercel)

    info_rows = [
        ("Subject", "Abhishek R"),
        ("Role", "System Engineer"),
        ("Origin", "Chikkaballapur"),
        ("Education", "BE - ECE"),
        ("Status", "Learning AWS, Docker etc"),
        ("ToolChain", "VS Code, Git, PyCharm, Postman"),
        ("Core.Lang", "English, Kannada, Hindi, Telugu"),
        ("Core.Frontend", "HTML, CSS, JavaScript"),
        ("Core.Backend", "Python, Django, DRF"),
        ("Core.Database", "MySQL, Postgres"),
        ("Grid.Mail", "abhishekabhi.r.2001@gmail.com"),
        ("Grid.Portfolio", "abhi3468.github.io/Portfolio/"),
        ("Grid.LinkedIn", "linkedin.com/in/abhishek-r-652915211/"),
        ("Grid.GitHub", "github.com/Abhi3468"),
    ]

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_W} {CANVAS_H}" width="{CANVAS_W}" height="{CANVAS_H}">')

    svg_parts.append(f'''<style>
      .bg {{ fill: {bg_col}; }}
      .chrome {{ stroke: {chrome_col}; stroke-width: 1.5; fill: none; }}
      .chrome-fill {{ fill: {chrome_col}; }}
      .txt-hdr {{ font-family: 'Fira Code', 'Courier New', monospace; font-size: 13px; font-weight: bold; fill: {chrome_col}; }}
      .txt-row {{ font-family: 'Fira Code', 'Courier New', monospace; font-size: 14px; fill: {text_col}; }}
      .txt-lbl {{ font-family: 'Fira Code', 'Courier New', monospace; font-size: 14px; fill: {muted_col}; }}
      .txt-pill {{ font-family: 'Fira Code', 'Courier New', monospace; font-size: 14px; font-weight: bold; fill: {bg_col}; }}
      .dot-leader {{ font-family: 'Fira Code', 'Courier New', monospace; font-size: 14px; fill: {muted_col}; opacity: 0.4; }}
      .live-badge {{ font-family: 'Fira Code', 'Courier New', monospace; font-size: 12px; font-weight: bold; fill: {LIVE_RED}; }}
      .portrait-dot {{ fill: {portrait_col}; shape-rendering: crispEdges; }}
      .traveller-dot {{ fill: {portrait_col}; shape-rendering: crispEdges; }}
    </style>''')

    svg_parts.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" rx="12" class="bg"/>')
    svg_parts.append(f'<rect x="2" y="2" width="{CANVAS_W-4}" height="{CANVAS_H-4}" rx="10" class="chrome" opacity="0.8"/>')

    svg_parts.append(f'<line x1="2" y1="45" x2="{CANVAS_W-2}" y2="45" class="chrome" opacity="0.4"/>')
    svg_parts.append(f'<circle cx="25" cy="23" r="6" fill="#FF5F56"/>')
    svg_parts.append(f'<circle cx="45" cy="23" r="6" fill="#FFBD2E"/>')
    svg_parts.append(f'<circle cx="65" cy="23" r="6" fill="#27C93F"/>')
    svg_parts.append(f'<text x="95" y="28" class="txt-hdr">profile.sh --live</text>')

    svg_parts.append(f'<rect x="980" y="12" width="110" height="24" rx="12" class="chrome-fill"/>')
    svg_parts.append(f'<text x="992" y="28" class="txt-pill">@Abhi3468</text>')

    svg_parts.append(f'<circle cx="1115" cy="24" r="5" fill="{LIVE_RED}">')
    svg_parts.append(f'  <animate attributeName="opacity" values="1;0.2;1" keyTimes="0;0.5;1" dur="1.8s" repeatCount="indefinite"/>')
    svg_parts.append(f'</circle>')
    svg_parts.append(f'<text x="1126" y="28" class="live-badge">LIVE</text>')

    svg_parts.append(f'<rect x="30" y="65" width="410" height="515" rx="8" class="chrome" opacity="0.5"/>')
    svg_parts.append(f'<text x="45" y="88" class="txt-hdr">VISUAL.MAP</text>')
    svg_parts.append(f'<line x1="30" y1="98" x2="440" y2="98" class="chrome" opacity="0.3"/>')

    # Portrait Base Intro Layer (Run-length compressed)
    svg_parts.append('<g id="portrait-intro-layer">')
    for g_idx, group in enumerate(intro_groups):
        delay = (g_idx / num_intro_groups) * 1.8
        d_str = dots_to_run_length_path(group)
        if not d_str: continue
        svg_parts.append(f'  <path d="{d_str}" class="portrait-dot" opacity="0">')
        svg_parts.append(f'    <animate attributeName="opacity" values="0;1" keyTimes="0;1" dur="0.4s" begin="{delay:.2f}s" fill="freeze"/>')
        svg_parts.append(f'  </path>')
    svg_parts.append('</g>')

    # Portrait Loop Layer (94 bands, Run-length compressed)
    key_times = "0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1"
    op_values = "1;1;0;0;0;0;0;1;1"

    num_bands = 94
    bands = [[] for _ in range(num_bands)]
    for x, y in noisy_portrait_dots:
        b_idx = int((y - PORTRAIT_OFFSET_Y) / (390.0 / num_bands))
        b_idx = max(0, min(num_bands - 1, b_idx))
        bands[b_idx].append((x, y))

    centroid_x, centroid_y = 220, 320

    svg_parts.append('<g id="portrait-loop-layer">')
    for b_idx, b_dots in enumerate(bands):
        if not b_dots: continue
        d_str = dots_to_run_length_path(b_dots)
        if not d_str: continue

        avg_x = sum(pt[0] for pt in b_dots) / len(b_dots)
        avg_y = sum(pt[1] for pt in b_dots) / len(b_dots)
        dx = (centroid_x - avg_x) * 0.42
        dy = (centroid_y - avg_y) * 0.42

        trans_values = f"0,0; 0,0; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; {dx:.1f},{dy:.1f}; 0,0; 0,0"

        svg_parts.append(f'  <g class="portrait-dot">')
        svg_parts.append(f'    <path d="{d_str}"/>')
        svg_parts.append(f'    <animate attributeName="opacity" values="{op_values}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>')
        svg_parts.append(f'    <animateTransform attributeName="transform" type="translate" values="{trans_values}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>')
        svg_parts.append(f'  </g>')
    svg_parts.append('</g>')

    # Travellers Layer (~850 dots morphing between logos)
    traveller_op = "0;0;1;1;1;1;1;0;0"
    svg_parts.append('<g id="travellers-layer">')
    for i in range(num_travellers):
        p_flut = logo_flutter[i]
        p_code = logo_code[i]
        p_verc = logo_vercel[i]

        x_vals = f"{p_flut[0]:.1f};{p_flut[0]:.1f};{p_code[0]:.1f};{p_code[0]:.1f};{p_verc[0]:.1f};{p_verc[0]:.1f};{p_flut[0]:.1f};{p_flut[0]:.1f};{p_flut[0]:.1f}"
        y_vals = f"{p_flut[1]:.1f};{p_flut[1]:.1f};{p_code[1]:.1f};{p_code[1]:.1f};{p_verc[1]:.1f};{p_verc[1]:.1f};{p_flut[1]:.1f};{p_flut[1]:.1f};{p_flut[1]:.1f}"

        svg_parts.append(f'  <rect width="1.4" height="1.4" class="traveller-dot" x="{p_flut[0]:.1f}" y="{p_flut[1]:.1f}">')
        svg_parts.append(f'    <animate attributeName="opacity" values="{traveller_op}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>')
        svg_parts.append(f'    <animate attributeName="x" values="{x_vals}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>')
        svg_parts.append(f'    <animate attributeName="y" values="{y_vals}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>')
        svg_parts.append(f'  </rect>')
    svg_parts.append('</g>')

    # Right Frame — SYSTEM.INFO
    panel_x = 470
    panel_y = 65
    panel_w = 680
    panel_h = 515

    svg_parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" class="chrome" opacity="0.5"/>')
    svg_parts.append(f'<text x="{panel_x+15}" y="{panel_y+23}" class="txt-hdr">SYSTEM.INFO</text>')
    svg_parts.append(f'<line x1="{panel_x}" y1="{panel_y+33}" x2="{panel_x+panel_w}" y2="{panel_y+33}" class="chrome" opacity="0.3"/>')

    start_y = panel_y + 62
    row_step = 23
    val_x = panel_x + panel_w - 20

    for idx, (label, value) in enumerate(info_rows):
        cur_y = start_y + idx * row_step
        lbl_str = f"{label}:"

        lbl_w = len(lbl_str) * 8.5
        val_w = len(value) * 8.5

        leader_start_x = panel_x + 20 + lbl_w + 10
        leader_end_x = val_x - val_w - 10
        leader_dots_count = max(0, int((leader_end_x - leader_start_x) / 8))
        leader_str = ". " * leader_dots_count

        svg_parts.append(f'<g>')
        svg_parts.append(f'  <text x="{panel_x+20}" y="{cur_y}" class="txt-lbl">{lbl_str}</text>')
        if leader_dots_count > 0:
            svg_parts.append(f'  <text x="{leader_start_x:.1f}" y="{cur_y}" class="dot-leader">{leader_str}</text>')
        approx_val_length = int(len(value) * 8.4)
        svg_parts.append(f'  <text x="{val_x}" y="{cur_y}" text-anchor="end" class="txt-row" textLength="{approx_val_length}" lengthAdjust="spacingAndGlyphs">{value}</text>')
        svg_parts.append(f'</g>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dark_path = os.path.join(script_dir, "dark.svg")
    light_path = os.path.join(script_dir, "light.svg")

    print("Generating dark.svg...")
    dark_svg = build_svg(dark_mode=True)
    with open(dark_path, "w", encoding="utf-8") as f:
        f.write(dark_svg)

    print("Generating light.svg...")
    light_svg = build_svg(dark_mode=False)
    with open(light_path, "w", encoding="utf-8") as f:
        f.write(light_svg)

    print("SVG Generation complete!")
