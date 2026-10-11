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
python3 main.py --image receipt.jpg
```

Running tests:

```
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Tests that need the Tesseract engine are skipped if it is not installed.
