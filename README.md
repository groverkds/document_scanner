# Document Scanner
*** No more being maintained ***

Scanning documents from photographs using Python and OpenCV.
Thanks to Adrian Rosebrock(www.pyimagesearch.com) for an awesome tutorial which enabled me to write this code.

Steps to run:

Requires Python 3.11 or newer (tested on 3.14) and the [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) engine
(e.g. `sudo apt install tesseract-ocr` or `brew install tesseract`).

Scanned pages are rotated so their text is upright using Tesseract's orientation detection
(`osd.traineddata`, included in the standard Tesseract packages). Pages with too little text are left as photographed.

```
pip install -r requirements.txt
python3 main.py receipt.jpg
```

This saves the scan as `receipt_scanned.png` next to the photo. Options:

| Option | Effect |
| --- | --- |
| `-o PATH`, `--output PATH` | Save the scan to `PATH` (single image only) |
| `--no-save` | Don't save the scan |
| `--no-threshold` | Keep the scan in color instead of black and white |
| `--no-rotate` | Don't turn the page so its text is upright |
| `--ocr` | Print the text found on the page |
| `--show` | Show the photo and the scan in windows |

Several photos can be scanned at once, e.g. `python3 main.py *.jpg --ocr`.
If no page outline is found, the photo is saved unchanged and a warning is printed.
The exit status is 1 if any photo could not be read or saved.

Running tests:

```
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Tests that need the Tesseract engine are skipped if it is not installed.
