import argparse
import os
import sys
from pathlib import Path

import cv2
import pytesseract

import document_scanner as package
from document_scanner import document_scanner

DESCRIPTION = "Scan documents from photographs: find the page, straighten it and optionally extract its text."


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="document-scanner", description=DESCRIPTION)
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + package.__version__
    )
    parser.add_argument(
        "images", nargs="*", metavar="IMAGE", help="photo(s) of a document"
    )
    parser.add_argument(
        "--image",
        dest="legacy_images",
        action="append",
        default=[],
        metavar="IMAGE",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="PATH",
        help="where to save the scan (only with a single image); default: <image>_scanned.png next to each image",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not save the scanned image"
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="keep the scan in color instead of black and white",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="do not turn the page so its text is upright",
    )
    parser.add_argument(
        "--ocr", action="store_true", help="print the text found on the page"
    )
    parser.add_argument(
        "--show", action="store_true", help="show the photo and the scan in windows"
    )
    args = parser.parse_args(argv)

    args.images = [Path(p) for p in args.images + args.legacy_images]
    if not args.images:
        parser.error("no image given")
    if args.output and len(args.images) > 1:
        parser.error("--output can only be used with a single image")
    if args.output and args.no_save:
        parser.error("--output and --no-save cannot be used together")
    return args


def tesseract_available():
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False


def output_path(image_path):
    return image_path.with_name(image_path.stem + "_scanned.png")


def display_available():
    # Without a display, OpenCV's window backend on Linux aborts the process instead of raising an error
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def show(image, scan):
    if not display_available():
        print("Warning: no display available; cannot show windows", file=sys.stderr)
        return
    try:
        cv2.imshow("Photo", document_scanner.resize_to_height(image, 800))
        cv2.imshow(
            "Scan", document_scanner.resize_to_height(scan, min(800, scan.shape[0]))
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print(
            "Warning: cannot open windows (OpenCV was built without GUI support)",
            file=sys.stderr,
        )


def main(argv=None):
    args = parse_args(argv)

    rotate = not args.no_rotate
    if (rotate or args.ocr) and not tesseract_available():
        if args.ocr:
            print(
                "Error: --ocr needs the Tesseract OCR engine, which was not found",
                file=sys.stderr,
            )
            return 2
        print(
            "Warning: Tesseract OCR engine not found; scans will not be rotated upright",
            file=sys.stderr,
        )
        rotate = False

    scanner = document_scanner.DocumentScanner()
    status = 0
    for path in args.images:
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None:
            print("Error: cannot read image %s" % path, file=sys.stderr)
            status = 1
            continue

        scan = scanner.Scan(
            image, perform_threshold=not args.no_threshold, perform_auto_rotate=rotate
        )

        if not args.no_save:
            destination = args.output or output_path(path)
            if not cv2.imwrite(str(destination), scan):
                print("Error: cannot save scan to %s" % destination, file=sys.stderr)
                status = 1
            else:
                print("Saved %s" % destination, file=sys.stderr)

        if args.ocr:
            if len(args.images) > 1:
                print("==> %s <==" % path)
            print(document_scanner.TextConverter().ConvertImageToText(scan))

        if args.show:
            show(image, scan)

    return status
