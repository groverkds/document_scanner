import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from document_scanner import cli
from document_scanner import document_scanner as ds
from tests import synthetic

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def photo(tmp_path):
    """A synthetic photo of a page, saved as a file."""
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), synthetic.photo(synthetic.rotated_quad(10)))
    return path


@pytest.fixture
def no_tesseract_calls(monkeypatch):
    """Records auto-rotation and OCR calls instead of running Tesseract."""
    calls = []
    monkeypatch.setattr(cli, "tesseract_available", lambda: True)
    monkeypatch.setattr(ds, "auto_rotate", lambda page: calls.append("rotate") or page)
    monkeypatch.setattr(
        ds.TextConverter,
        "ConvertImageToText",
        lambda self, image: calls.append("ocr") or "TEXT",
    )
    return calls


def test_saves_black_and_white_scan_next_to_image_by_default(photo, no_tesseract_calls):
    assert cli.main([str(photo)]) == 0
    scan = cv2.imread(str(photo.with_name("photo_scanned.png")), cv2.IMREAD_UNCHANGED)
    assert scan.ndim == 2 and set(np.unique(scan)) <= {0, 255}
    assert abs(scan.shape[0] / scan.shape[1] - 620 / 500) < 0.05
    assert no_tesseract_calls == ["rotate"]


def test_output_option_and_color_scan(photo, tmp_path, no_tesseract_calls):
    output = tmp_path / "out" / "scan.jpg"
    output.parent.mkdir()
    assert cli.main([str(photo), "-o", str(output), "--no-threshold"]) == 0
    assert cv2.imread(str(output)).ndim == 3
    assert not photo.with_name("photo_scanned.png").exists()


def test_legacy_image_option_still_works(photo, no_tesseract_calls):
    assert cli.main(["--image", str(photo)]) == 0
    assert photo.with_name("photo_scanned.png").exists()


def test_no_rotate_and_no_save(photo, no_tesseract_calls):
    assert cli.main([str(photo), "--no-rotate", "--no-save"]) == 0
    assert no_tesseract_calls == []
    assert not photo.with_name("photo_scanned.png").exists()


def test_ocr_prints_text_with_headers_for_several_images(
    photo, tmp_path, no_tesseract_calls, capsys
):
    second = tmp_path / "second.jpg"
    shutil.copy(photo, second)
    assert cli.main([str(photo), str(second), "--ocr", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert out == f"==> {photo} <==\nTEXT\n==> {second} <==\nTEXT\n"


def test_ocr_prints_only_text_for_one_image(photo, no_tesseract_calls, capsys):
    cli.main([str(photo), "--ocr", "--no-save"])
    assert capsys.readouterr().out == "TEXT\n"


def test_unreadable_image_is_reported_and_others_still_scanned(
    photo, tmp_path, no_tesseract_calls, capsys
):
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    assert cli.main([str(tmp_path / "missing.jpg"), str(broken), str(photo)]) == 1
    err = capsys.readouterr().err
    assert f"cannot read image {tmp_path / 'missing.jpg'}" in err
    assert f"cannot read image {broken}" in err
    assert photo.with_name("photo_scanned.png").exists()


def test_page_not_found_saves_original_with_warning(
    tmp_path, no_tesseract_calls, capsys
):
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((400, 300, 3), 255, np.uint8))
    assert cli.main([str(blank)]) == 0
    assert "no page outline found" in capsys.readouterr().err
    assert cv2.imread(str(tmp_path / "blank_scanned.png")).shape == (400, 300, 3)


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "no image given"),
        (
            ["a.jpg", "b.jpg", "-o", "x.png"],
            "--output can only be used with a single image",
        ),
        (
            ["a.jpg", "-o", "x.png", "--no-save"],
            "--output and --no-save cannot be used together",
        ),
    ],
)
def test_invalid_arguments(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_without_tesseract_rotation_is_skipped_with_warning(photo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "tesseract_available", lambda: False)
    monkeypatch.setattr(
        ds, "auto_rotate", lambda page: pytest.fail("auto_rotate called")
    )
    assert cli.main([str(photo)]) == 0
    assert "will not be rotated" in capsys.readouterr().err


def test_without_tesseract_ocr_is_an_error(photo, monkeypatch, capsys):
    monkeypatch.setattr(cli, "tesseract_available", lambda: False)
    assert cli.main([str(photo), "--ocr"]) == 2
    assert "--ocr needs the Tesseract OCR engine" in capsys.readouterr().err


def test_show_without_display_warns_instead_of_crashing(
    photo, monkeypatch, no_tesseract_calls, capsys
):
    monkeypatch.setattr(cli, "display_available", lambda: False)
    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("imshow called"))
    assert cli.main([str(photo), "--show", "--no-save"]) == 0
    assert "no display available" in capsys.readouterr().err


def test_show_opens_windows_when_display_available(
    photo, monkeypatch, no_tesseract_calls
):
    shown = []
    monkeypatch.setattr(cli, "display_available", lambda: True)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append(name))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    assert cli.main([str(photo), "--show", "--no-save"]) == 0
    assert shown == ["Photo", "Scan"]


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)
def test_main_script_end_to_end(tmp_path):
    receipt = tmp_path / "receipt.jpg"
    shutil.copy(ROOT / "examples" / "receipt.jpg", receipt)
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), str(receipt), "--ocr"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "WHOLE FOODS MARKET" in result.stdout
    scan = cv2.imread(str(tmp_path / "receipt_scanned.png"), cv2.IMREAD_UNCHANGED)
    assert scan.shape[0] > scan.shape[1]  # rotated upright: the receipt is portrait


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("document-scanner ")


def test_runs_as_python_module():
    result = subprocess.run(
        [sys.executable, "-m", "document_scanner", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: document-scanner")
