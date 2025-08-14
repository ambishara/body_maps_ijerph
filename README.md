# Body-Map PDF → Aligned Images Pipeline

Minimal, reproducible pipeline to:
1) Rasterize specific pages from 5-page body-map PDFs to JPEGs.
2) Align cropped regions (front/back) to a canonical template via ORB+RANSAC.
3) Remove printed template lines to isolate participant-drawn content.
4) Save `_front_result.jpg` and `_back_result.jpg` per input image.

> This repository is generic and intentionally free of institution- or project-specific details.

---

## Quick Start

### 0) Dependencies
- **Python** ≥ 3.10
- Packages: `opencv-python`, `pdf2image`, `pillow`, `numpy`, `python-dotenv`
- **Poppler** (for `pdf2image`):
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - Windows: install Poppler and add `bin/` to `PATH`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install opencv-python pdf2image pillow numpy python-dotenv

