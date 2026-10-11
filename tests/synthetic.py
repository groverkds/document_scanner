"""Synthetic document photos with known page corners, for testing page detection."""

import cv2
import numpy as np

WIDTH, HEIGHT = 1200, 900


def text_page(width=600, height=800, seed=0):
    """A light page covered in dark text-like lines, with a red marker in its top-left corner."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), 245, np.uint8)
    for y in range(60, height - 60, 30):
        x = 50
        while x < width - 80:
            length = int(rng.integers(20, 90))
            cv2.rectangle(page, (x, y), (min(x + length, width - 50), y + 10), (40, 40, 40), -1)
            x += length + 12
    cv2.rectangle(page, (30, 25), (110, 45), (0, 0, 255), -1)
    return page


def background(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "dark":
        return np.full((HEIGHT, WIDTH, 3), (40, 45, 50), np.uint8)
    if kind == "light":
        # Low contrast: light grey table under a white page
        return np.full((HEIGHT, WIDTH, 3), (190, 195, 200), np.uint8)
    if kind == "wood":
        x = np.arange(WIDTH)[None, :] + 25 * np.sin(np.arange(HEIGHT)[:, None] / 40.0)
        value = 90 + 40 * np.sin(x / 9.0) + rng.normal(0, 12, (HEIGHT, WIDTH))
        return np.clip(np.stack([value * 0.55, value * 0.75, value], -1), 0, 255).astype(np.uint8)
    if kind == "noise":
        return np.clip(rng.normal(90, 45, (HEIGHT, WIDTH, 3)), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def rotated_quad(angle, cx=600, cy=450, width=500, height=620):
    """Corners (top-left, top-right, bottom-right, bottom-left) of a rectangle rotated clockwise by `angle` degrees."""
    corners = np.float32([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    a = np.deg2rad(angle)
    rotation = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return corners @ rotation.T + np.float32([cx, cy])


def photo(quad, kind="dark", seed=0):
    """`text_page` warped onto `quad` over a background."""
    page = text_page(seed=seed)
    h, w = page.shape[:2]
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(source, np.float32(quad))
    warped = cv2.warpPerspective(page, matrix, (WIDTH, HEIGHT))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (WIDTH, HEIGHT))
    out = background(kind, seed)
    out[mask > 0] = warped[mask > 0]
    return out


def corner_error(found, truth):
    """Largest distance between found and true corners. At exactly 45 degrees no corner is
    the top-left one, so any cyclic order of the true corners is accepted."""
    found, truth = np.float32(found), np.float32(truth)
    dx, dy = truth[1] - truth[0]
    shifts = range(4) if abs(abs(np.degrees(np.arctan2(dy, dx))) - 45) < 1 else [0]
    return min(float(np.max(np.linalg.norm(found - np.roll(truth, k, axis=0), axis=1))) for k in shifts)


def hard_cases():
    """(name, image, true corners) for photos the original single Canny pass could not handle."""
    cases = []
    for angle in (-45, 45):
        quad = rotated_quad(angle)
        cases.append((f"rotated{angle:+d}", photo(quad), quad))

    quad = rotated_quad(0)
    cases.append(("low_contrast", photo(quad, "light"), quad))

    quad = rotated_quad(8)
    image = photo(quad)
    bottom_middle = (quad[2] + quad[3]) / 2
    cv2.ellipse(image, (int(bottom_middle[0]), int(bottom_middle[1])), (45, 80), 15, 0, 360, (120, 150, 210), -1)
    cases.append(("thumb_over_edge", image, quad))

    quad = rotated_quad(5)
    image = photo(quad)
    top_right = quad[1]
    cv2.fillConvexPoly(image, np.int32([top_right, top_right + [-45, -5], top_right + [5, 45]]), (40, 45, 50))
    cases.append(("dog_eared_corner", image, quad))

    quad = rotated_quad(10, width=230, height=300)
    cases.append(("small_page", photo(quad), quad))

    # A plain page rotated with warpAffine: its anti-aliased outline breaks up after downscaling
    for angle in (-35, 30):
        canvas = np.full((900, 900, 3), 40, np.uint8)
        canvas[250:650, 300:600] = 255
        matrix = cv2.getRotationMatrix2D((450, 450), angle, 1.0)
        image = cv2.warpAffine(canvas, matrix, (900, 900), borderValue=(40, 40, 40))
        quad = cv2.transform(np.float32([[[300, 250], [599, 250], [599, 649], [300, 649]]]), matrix)[0]
        cases.append((f"broken_outline{angle:+d}", image, quad))
    return cases


def easy_cases():
    """(name, image, true corners) for photos that the original detection already handled."""
    cases = []
    for angle in (-40, -20, 0, 20, 40):
        quad = rotated_quad(angle)
        cases.append((f"rotated{angle:+d}", photo(quad), quad))
    rng = np.random.default_rng(1)
    for i in range(4):
        quad = rotated_quad(rng.uniform(-20, 20)) + rng.uniform(-60, 60, (4, 2)).astype(np.float32)
        cases.append((f"perspective{i}", photo(quad, seed=i), quad))
    for kind in ("wood", "noise"):
        quad = rotated_quad(-15)
        cases.append((f"{kind}_background", photo(quad, kind), quad))

    quad = rotated_quad(-10)
    shadow = photo(quad).astype(np.float32) * np.linspace(0.45, 1.0, WIDTH)[None, :, None]
    cases.append(("shadow", shadow.astype(np.uint8), quad))

    quad = rotated_quad(12)
    cases.append(("blurred", cv2.GaussianBlur(photo(quad), (15, 15), 0), quad))

    quad = rotated_quad(-5)
    image = photo(quad)
    cv2.rectangle(image, (520, 380), (680, 480), (30, 30, 200), 4)
    cases.append(("box_printed_on_page", image, quad))
    return cases


def pageless_images():
    """(name, image) for photos that contain no page."""
    blank = np.full((800, 600, 3), 255, np.uint8)
    small_box = blank.copy()
    cv2.rectangle(small_box, (280, 380), (320, 420), (0, 0, 0), 3)
    images = [
        ("blank", blank),
        ("dark", np.full((800, 600, 3), 40, np.uint8)),
        ("small_box", small_box),
        ("wood", background("wood")),
        ("noise_background", background("noise")),
    ]
    for seed in range(5):
        images.append((f"random_pixels{seed}", np.random.default_rng(seed).integers(0, 256, (800, 600, 3), dtype=np.uint8)))
    return images
