# Fibre Classifier

**Fibre Classifier** is a Python-based GUI application for semi-automated classification of muscle fibres in immunohistochemistry (IHC) images. The tool enables loading TIFF images and fibre outlines, computes quantitative features, supports automatic and manual classification, and exports overlays and tabular results.

## Features

- Load 8-bit grayscale TIFF images
- Load fibre outline coordinates from TXT files
- Segment fibres and compute:
  - Mean and standard deviation of intensity
  - Area in µm²
  - Convexity and roundness
- Manual exclusion of fibres
- Classification options:
  - Auto-thresholding using k-means
  - Manual label assignment via GUI
  - Optional revision of auto labels
- Save classification overlays (PNG) and data (CSV)

## Requirements

- Python 3.9 or higher
- Dependencies:
  - numpy
  - pandas
  - matplotlib
  - scikit-image
  - shapely
  - scikit-learn
  - tifffile
  - tkinter
  - imagej (for pixel calibration via ImageJ/Fiji)

## Installation

Clone the repository:

```bash
git clone https://github.com/frontzekkarl/fibre-classifier.git
cd fibre-classifier
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Launch the script:

```bash
python fibreclassifier.py
```

A GUI will guide you through:

1. Loading an image and outline.
2. Segmenting and excluding fibres.
3. Selecting classification mode.
4. Saving results and overlay images.

## Notes

- The script attempts to detect pixel size using ImageJ's metadata (via pyimagej). If this fails, it falls back to a default value.
- Overlays and CSV results are saved to the working directory.

## License

MIT License

---

Developed by [Your Name]
