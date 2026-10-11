import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import document_scanner.document_scanner as ds
from tests import synthetic

ROOT = Path(__file__).resolve().parent.parent

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate(image, degrees):
    return (
        image if ROTATIONS[degrees] is None else cv2.rotate(image, ROTATIONS[degrees])
    )


def load(name):
    image = cv2.imread(str(ROOT / "examples" / name))
    assert image is not None, name
    return image


def photo_of_page(angle):
    """A 300x400 white page with a red marker in its top-left corner, tilted by `angle` on a dark background."""
    page = np.full((400, 300, 3), 255, np.uint8)
    cv2.rectangle(page, (20, 20), (80, 60), (0, 0, 255), -1)
    canvas = np.full((900, 900, 3), 40, np.uint8)
    canvas[250:650, 300:600] = page
    matrix = cv2.getRotationMatrix2D((450, 450), angle, 1.0)
    return cv2.warpAffine(canvas, matrix, (900, 900), borderValue=(40, 40, 40))


def blank_photo():
    return np.full((800, 600, 3), 255, np.uint8)


# resize_to_height


def test_resize_to_height_keeps_aspect_ratio():
    out = ds.resize_to_height(np.zeros((1000, 600, 3), np.uint8), 500)
    assert out.shape == (500, 300, 3)


# adaptive_threshold


def test_adaptive_threshold_works_for_every_height():
    # blockSize used to be height//10+1, which OpenCV rejects when it is even
    rng = np.random.default_rng(0)
    for height in range(30, 4000):
        gray = rng.integers(0, 256, (height, 8), dtype=np.uint8)
        assert ds.adaptive_threshold(gray).shape == gray.shape


@pytest.mark.parametrize("height", [1200, 1400, 2624])
def test_adaptive_threshold_unchanged_where_old_block_size_was_valid(height):
    gray = np.random.default_rng(height).integers(0, 256, (height, 50), dtype=np.uint8)
    expected = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        height // 10 + 1,
        height // 100 + 1,
    )
    assert (ds.adaptive_threshold(gray) == expected).all()


# order_corners

TL, TR, BR, BL = [10, 12], [200, 20], [210, 300], [5, 290]


@pytest.mark.parametrize("start", range(4))
@pytest.mark.parametrize("clockwise", [True, False])
def test_order_corners_independent_of_start_and_winding(start, clockwise):
    corners = [TL, TR, BR, BL] if clockwise else [TL, BL, BR, TR]
    corners = corners[start:] + corners[:start]
    approx = np.array(corners).reshape(4, 1, 2)  # the shape approxPolyDP returns
    assert ds.order_corners(approx).tolist() == [TL, TR, BR, BL]


def test_order_corners_at_45_degrees_returns_four_distinct_corners_clockwise():
    # A diamond: the top and left corners tie for the smallest x + y
    diamond = np.array([[100, 0], [200, 100], [100, 200], [0, 100]])
    for start in range(4):
        ordered = ds.order_corners(np.roll(diamond, start, axis=0))
        assert len({tuple(c) for c in ordered.tolist()}) == 4
        assert ordered.tolist() in [
            np.roll(diamond, k, axis=0).tolist() for k in range(4)
        ]


# detect_page


@pytest.mark.parametrize(
    "image, truth",
    [
        pytest.param(image, truth, id=name)
        for name, image, truth in synthetic.easy_cases() + synthetic.hard_cases()
    ],
)
def test_detect_page_finds_corners(image, truth):
    assert synthetic.corner_error(ds.DocumentScanner().detect_page(image), truth) < 15


@pytest.mark.parametrize(
    "image",
    [pytest.param(image, id=name) for name, image in synthetic.pageless_images()],
)
def test_detect_page_finds_nothing_in_pageless_images(image):
    with pytest.raises(ds.PageNotFoundError):
        ds.DocumentScanner().detect_page(image)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("page.jpg", [[372, 353], [2148, 470], [2350, 2774], [398, 2977]]),
        ("receipt.jpg", [[553, 671], [2277, 539], [2321, 1581], [720, 1836]]),
    ],
)
def test_detect_page_on_samples_matches_original_detection(name, expected):
    # Images the original Canny pass handled must give the same corners as before
    corners = ds.DocumentScanner().detect_page(load(name))
    assert np.max(np.abs(corners - np.array(expected))) <= 2


def test_longest_sides_quad_recovers_cut_off_corner():
    # A 200x300 rectangle with its top-right corner cut off diagonally
    outline = np.array(
        [[0, 0], [170, 0], [200, 30], [200, 300], [0, 300]], np.int32
    ).reshape(-1, 1, 2)
    quad = ds.longest_sides_quad(outline)
    assert (
        synthetic.corner_error(
            ds.order_corners(quad), [[0, 0], [200, 0], [200, 300], [0, 300]]
        )
        < 1
    )


def test_side_support_is_low_for_a_side_without_outline():
    outline = np.full((100, 100), 255, np.uint8)
    cv2.polylines(
        outline, [np.int32([[10, 90], [10, 10], [90, 10], [90, 90]])], False, 0
    )  # bottom side missing
    distance = cv2.distanceTransform(outline, cv2.DIST_L2, 3)
    quad = np.float32([[10, 10], [90, 10], [90, 90], [10, 90]])
    assert ds.side_support(quad, distance) < ds.MIN_SIDE_SUPPORT / 2
    cv2.line(outline, (10, 90), (90, 90), 0)
    assert ds.side_support(quad, cv2.distanceTransform(outline, cv2.DIST_L2, 3)) == 1.0


def test_touches_border():
    assert ds.touches_border(
        np.float32([[1, 1], [399, 1], [399, 499], [1, 499]]), (500, 400)
    )
    assert ds.touches_border(
        np.float32([[50, 1], [300, 2], [300, 400], [50, 400]]), (500, 400)
    )
    assert not ds.touches_border(
        np.float32([[50, 40], [300, 40], [300, 400], [50, 400]]), (500, 400)
    )


# find_page / Scan without a page


def test_find_page_raises_when_there_is_no_page():
    with pytest.raises(ds.PageNotFoundError):
        ds.DocumentScanner().find_page(np.zeros((500, 400), np.uint8))


def test_find_page_ignores_small_rectangles():
    edge_map = np.zeros((500, 400), np.uint8)
    cv2.rectangle(edge_map, (180, 230), (220, 270), 255, 1)
    with pytest.raises(ds.PageNotFoundError):
        ds.DocumentScanner().find_page(edge_map)


def test_scan_returns_original_and_warns_when_no_page(capsys):
    image = blank_photo()
    out = ds.DocumentScanner().Scan(image, perform_threshold=True)
    assert (out == image).all()
    assert "no page outline found" in capsys.readouterr().err


# Scan geometry


@pytest.mark.parametrize("angle", [-40, -20, -10, 0, 10, 20, 40])
def test_scan_straightens_tilted_page_keeping_top_left(angle):
    out = ds.DocumentScanner().Scan(photo_of_page(angle))
    height, width = out.shape[:2]
    assert height > width
    assert abs(height / width - 4 / 3) < 0.05
    ys, xs = np.nonzero((out[:, :, 2] > 150) & (out[:, :, 1] < 100))
    assert ys.mean() < height / 3 and xs.mean() < width / 3


def test_scan_sample_page():
    out = ds.DocumentScanner().Scan(load("page.jpg"), perform_threshold=True)
    assert out.ndim == 2
    assert abs(out.shape[0] - 2624) < 10 and abs(out.shape[1] - 1962) < 10
    assert set(np.unique(out)) <= {0, 255}


# auto_rotate


def fake_osd(monkeypatch, rotate, confidence):
    monkeypatch.setattr(
        ds.pytesseract,
        "image_to_osd",
        lambda *a, **k: {"rotate": rotate, "orientation_conf": confidence},
    )


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_auto_rotate_applies_detected_rotation(monkeypatch, degrees):
    page = np.arange(12, dtype=np.uint8).reshape(3, 4)
    fake_osd(monkeypatch, degrees, 5.0)
    assert (ds.auto_rotate(page) == rotate(page, degrees)).all()


def test_auto_rotate_ignores_low_confidence(monkeypatch):
    page = np.arange(12, dtype=np.uint8).reshape(3, 4)
    fake_osd(monkeypatch, 180, ds.MIN_ORIENTATION_CONFIDENCE / 2)
    assert (ds.auto_rotate(page) == page).all()


def test_auto_rotate_leaves_page_when_tesseract_finds_too_little_text(monkeypatch):
    def raise_error(*args, **kwargs):
        raise ds.pytesseract.TesseractError(1, "Too few characters. Skipping this page")

    monkeypatch.setattr(ds.pytesseract, "image_to_osd", raise_error)
    page = np.zeros((10, 20, 3), np.uint8)
    assert ds.auto_rotate(page) is page


@requires_tesseract
@pytest.mark.parametrize("name, upright_turn", [("page.jpg", 0), ("receipt.jpg", 270)])
@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_scan_auto_rotates_samples_upright(name, upright_turn, degrees):
    scanner = ds.DocumentScanner()
    photo = load(name)
    # receipt.jpg is photographed sideways, so its upright reference is turned 270 degrees clockwise
    reference = rotate(scanner.Scan(photo), upright_turn)
    out = scanner.Scan(rotate(photo, degrees), perform_auto_rotate=True)

    def mean_diff(image):
        resized = cv2.resize(image, (reference.shape[1], reference.shape[0]))
        return np.abs(resized.astype(int) - reference.astype(int)).mean()

    # The correct orientation must match the reference far better than any other orientation
    wrong = min(mean_diff(rotate(out, d)) for d in (90, 180, 270))
    assert mean_diff(out) < wrong * 0.7


# TextConverter


def test_text_converter_thresholds_color_input_and_passes_grayscale_through(
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(
        ds.pytesseract,
        "image_to_string",
        lambda image, lang: seen.append(np.array(image)) or "",
    )
    color = np.full((100, 80, 3), 200, np.uint8)
    gray = np.full((100, 80), 7, np.uint8)
    ds.TextConverter().ConvertImageToText(color)
    ds.TextConverter().ConvertImageToText(gray)
    assert seen[0].ndim == 2 and set(np.unique(seen[0])) <= {0, 255}
    assert (seen[1] == gray).all()


@requires_tesseract
def test_ocr_reads_sample_receipt():
    scanner = ds.DocumentScanner()
    page = scanner.Scan(
        load("receipt.jpg"), perform_threshold=True, perform_auto_rotate=True
    )
    text = ds.TextConverter().ConvertImageToText(page)
    assert "WHOLE FOODS MARKET" in text
