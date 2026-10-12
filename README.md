# Document Scanner

Scanning documents from photographs using Python and OpenCV.
Thanks to Adrian Rosebrock(www.pyimagesearch.com) for an awesome tutorial which enabled me to write this code.

Steps to run:

Requires Python 3.11 or newer (tested on 3.14) and the [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) engine
(e.g. `sudo apt install tesseract-ocr` or `brew install tesseract`).

Scanned pages are rotated so their text is upright using Tesseract's orientation detection
(`osd.traineddata`, included in the standard Tesseract packages). Pages with too little text are left as photographed.

```
pip install .
document-scanner examples/receipt.jpg
```

This saves the scan as `examples/receipt_scanned.png`, next to the photo.
`python -m document_scanner` and `python3 main.py` (from a checkout) work the same way. Options:

| Option | Effect |
| --- | --- |
| `-o PATH`, `--output PATH` | Save the scan to `PATH` (single image only) |
| `--no-save` | Don't save the scan |
| `--no-threshold` | Keep the scan in color instead of black and white |
| `--no-rotate` | Don't turn the page so its text is upright |
| `--ocr` | Print the text found on the page |
| `--show` | Show the photo and the scan in windows |

Several photos can be scanned at once, e.g. `document-scanner photos/*.jpg --ocr`.
If no page outline is found, the photo is saved unchanged and a warning is printed.
The exit status is 1 if any photo could not be read or saved.

It can also be used as a library:

```python
import cv2
from document_scanner.document_scanner import DocumentScanner, TextConverter

scan = DocumentScanner().Scan(cv2.imread("photo.jpg"), perform_threshold=True, perform_auto_rotate=True)
text = TextConverter().ConvertImageToText(scan)
```

Development:

```
pip install -e ".[dev]"
python -m pytest
black .
```

Tests that need the Tesseract engine are skipped if it is not installed. Code is formatted with [Black](https://black.readthedocs.io/).
