#!/usr/bin/env python3
"""
Process body-map PDFs and images into template-aligned, template-subtracted outputs.

Steps:
1) (Optional) Convert specified PDF pages to JPEGs.
2) For each JPEG, resize to template, crop (front/back), ORB-align to template,
   build a mask from the template lines, and remove those lines from the aligned image.
3) Save _front_result.jpg and _back_result.jpg per input image.

Configuration is via .env (see README and .env.example). CLI flags can override .env.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

from pdf2image import convert_from_path, pdfinfo_from_path

# ----------------------------
# Utilities
# ----------------------------

def parse_box(box_str: str) -> Tuple[int, int, int, int]:
    """Parse a crop box string 'left,top,right,bottom' into a 4-tuple of ints."""
    parts = [p.strip() for p in box_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid crop box: {box_str!r}. Expected 'l,t,r,b'.")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def parse_pages(pages_str: str) -> Tuple[int, int]:
    """
    Parse pages string like '2-4' into (first_page, last_page) (1-based, inclusive).
    """
    if "-" not in pages_str:
        p = int(pages_str.strip())
        return p, p
    a, b = pages_str.split("-", 1)
    return int(a.strip()), int(b.strip())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def count_pdf_pages(pdf_path: Path) -> int:
    info = pdfinfo_from_path(str(pdf_path))
    return int(info.get("Pages", 0))


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot load image: {path}")
    return img


# ----------------------------
# PDF → Images
# ----------------------------

def convert_pdf_pages_to_images(
    pdf_path: Path,
    out_dir: Path,
    first_page: int,
    last_page: int,
    dpi: int,
    expected_pages: Optional[int] = None,
) -> List[Path]:
    """
    Convert a page range from a PDF into JPEGs. Returns list of created image paths.
    Skips PDFs that do not match expected_pages (if provided).
    """
    created: List[Path] = []
    if expected_pages is not None:
        total = count_pdf_pages(pdf_path)
        if total != expected_pages:
            # Skip silently; caller may log if desired
            return created

    images = convert_from_path(
        str(pdf_path),
        first_page=first_page,
        last_page=last_page,
        dpi=dpi,
    )
    # Name files with 1-based page numbers
    page_nums = list(range(first_page, last_page + 1))
    stem = pdf_path.stem
    for img, pnum in zip(images, page_nums):
        out_path = out_dir / f"{stem}_page_{pnum}.jpg"
        img.save(out_path, "JPEG")
        created.append(out_path)
    return created


# ----------------------------
# Alignment & Template subtraction
# ----------------------------

def orb_align(
    img: np.ndarray,  # moving
    ref: np.ndarray,  # fixed
    n_features: int = 5000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Feature-based alignment using ORB + BF (Hamming) + RANSAC homography.
    Returns (aligned_img, H) where aligned_img matches ref's size.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(img, None)
    k2, d2 = orb.detectAndCompute(ref, None)

    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 4:
        return None, None

    matches = sorted(matches, key=lambda m: m.distance)
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    if H is None:
        return None, None

    h, w = ref.shape
    aligned = cv2.warpPerspective(img, H, (w, h), borderValue=255)
    return aligned, H


def build_template_mask(
    template_crop: np.ndarray,
    adaptive_block_size: int,
    adaptive_c: int,
    canny_low: int,
    canny_high: int,
    blur_sigma: float,
    thresh_value: float,
    dilate_kernel: int,
) -> np.ndarray:
    """
    Create a 'thickened' mask of the printed template lines to be removed.
    Returns a binary uint8 mask (0/255).
    """
    # Ensure odd block size for adaptive threshold
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1

    thr = cv2.adaptiveThreshold(
        template_crop,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )
    edges = cv2.Canny(template_crop, canny_low, canny_high)
    combined = cv2.bitwise_or(thr, edges)

    combined_f = combined.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(combined_f, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    _, thick = cv2.threshold(blurred, thresh_value, 1.0, cv2.THRESH_BINARY)

    thick_u8 = (thick * 255).astype(np.uint8)
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    mask = cv2.dilate(thick_u8, kernel, iterations=1)
    return mask


def process_one_side(
    img_gray: np.ndarray,
    template_gray: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    params: dict,
) -> Optional[np.ndarray]:
    """
    Resize -> crop -> align to template crop -> build template mask -> remove lines -> return result image.
    """
    th, tw = template_gray.shape
    resized = cv2.resize(img_gray, (tw, th))

    # Crop both to ROI
    l, t, r, b = crop_box
    img_crop = resized[t:b, l:r]
    tmpl_crop = template_gray[t:b, l:r]

    aligned, _ = orb_align(img_crop, tmpl_crop, n_features=params["orb_n_features"])
    if aligned is None:
        return None

    # Build mask from the template crop (lines to remove)
    mask = build_template_mask(
        tmpl_crop,
        adaptive_block_size=params["adaptive_block_size"],
        adaptive_c=params["adaptive_c"],
        canny_low=params["canny_low"],
        canny_high=params["canny_high"],
        blur_sigma=params["blur_sigma"],
        thresh_value=params["thresh_value"],
        dilate_kernel=params["dilate_kernel"],
    )

    # Create a "thickened" template image (black lines on white background)
    template_thick = np.full_like(tmpl_crop, 255, dtype=np.uint8)
    template_thick[mask == 255] = 0

    # Remove template lines from the aligned image by whitening those pixels
    result = aligned.copy()
    result[mask == 255] = 255

    # (Optionally) difference for QC — not saved by default
    # difference = cv2.absdiff(aligned, template_thick)

    return result


def process_image_file(
    img_path: Path,
    template_gray: np.ndarray,
    front_box: Tuple[int, int, int, int],
    back_box: Tuple[int, int, int, int],
    out_dir: Path,
    params: dict,
) -> None:
    img_gray = load_grayscale(img_path)

    base = img_path.stem
    # back
    back_result = process_one_side(img_gray, template_gray, back_box, params)
    if back_result is not None:
        cv2.imwrite(str(out_dir / f"{base}_back_result.jpg"), back_result)
    else:
        print(f"[WARN] Back alignment failed for {img_path.name}")

    # front
    front_result = process_one_side(img_gray, template_gray, front_box, params)
    if front_result is not None:
        cv2.imwrite(str(out_dir / f"{base}_front_result.jpg"), front_result)
    else:
        print(f"[WARN] Front alignment failed for {img_path.name}")


# ----------------------------
# Main entry
# ----------------------------

def main():
    load_dotenv()  # load .env if present

    parser = argparse.ArgumentParser(description="Process body-map PDFs/images into aligned, template-subtracted outputs.")
    parser.add_argument("--input-pdf-dir", default=os.getenv("INPUT_PDF_DIR", ""), help="Directory with PDFs to rasterize.")
    parser.add_argument("--output-image-dir", default=os.getenv("OUTPUT_IMAGE_DIR", ""), help="Where rasterized JPEGs go.")
    parser.add_argument("--output-result-dir", default=os.getenv("OUTPUT_RESULT_DIR", ""), help="Where processed outputs go.")
    parser.add_argument("--template-image-path", default=os.getenv("TEMPLATE_IMAGE_PATH", ""), help="Grayscale template image (JPEG/PNG).")

    parser.add_argument("--pages", default=os.getenv("PAGES_TO_CONVERT", "2-4"), help="Page range to convert, e.g., '2-4'.")
    parser.add_argument("--expected-pages", default=os.getenv("EXPECTED_PAGES", "5"), help="Skip PDFs that don't match this page count (blank to disable).")
    parser.add_argument("--dpi", type=int, default=int(os.getenv("DPI", "300")), help="DPI used when rasterizing PDFs.")

    parser.add_argument("--front-crop-box", default=os.getenv("FRONT_CROP_BOX", "296,260,511,650"))
    parser.add_argument("--back-crop-box", default=os.getenv("BACK_CROP_BOX", "85,260,300,650"))

    # Thresholding/align parameters
    parser.add_argument("--adaptive-block-size", type=int, default=int(os.getenv("ADAPTIVE_BLOCK_SIZE", "35")))
    parser.add_argument("--adaptive-c", type=int, default=int(os.getenv("ADAPTIVE_C", "7")))
    parser.add_argument("--canny-low", type=int, default=int(os.getenv("CANNY_LOW", "50")))
    parser.add_argument("--canny-high", type=int, default=int(os.getenv("CANNY_HIGH", "150")))
    parser.add_argument("--blur-sigma", type=float, default=float(os.getenv("BLUR_SIGMA", "2")))
    parser.add_argument("--thresh-value", type=float, default=float(os.getenv("THRESH_VALUE", "0.41")))
    parser.add_argument("--dilate-kernel", type=int, default=int(os.getenv("DILATION_KERNEL", "3")))
    parser.add_argument("--orb-n-features", type=int, default=int(os.getenv("ORB_N_FEATURES", "5000")))

    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF rasterization; process existing JPEGs only.")
    parser.add_argument("--images-glob", default=os.getenv("IMAGES_GLOB", "*.jpg"), help="Glob for images in OUTPUT_IMAGE_DIR.")

    args = parser.parse_args()

    # Resolve paths
    input_pdf_dir = Path(args.input_pdf_dir).expanduser() if args.input_pdf_dir else None
    output_image_dir = Path(args.output_image_dir).expanduser() if args.output_image_dir else None
    output_result_dir = Path(args.output_result_dir).expanduser() if args.output_result_dir else None
    template_image_path = Path(args.template_image_path).expanduser() if args.template_image_path else None

    if template_image_path is None or not template_image_path.exists():
        sys.exit("ERROR: TEMPLATE_IMAGE_PATH is required and must exist.")

    if output_result_dir is None:
        sys.exit("ERROR: OUTPUT_RESULT_DIR is required.")

    ensure_dir(output_result_dir)

    # Load template (grayscale)
    template_gray = load_grayscale(template_image_path)

    # Optional: PDF → JPEGs
    if not args.skip_pdf:
        if input_pdf_dir is None or not input_pdf_dir.exists():
            sys.exit("ERROR: INPUT_PDF_DIR must exist (or pass --skip-pdf).")
        if output_image_dir is None:
            sys.exit("ERROR: OUTPUT_IMAGE_DIR is required when rasterizing PDFs.")
        ensure_dir(output_image_dir)

        first_page, last_page = parse_pages(args.pages)
        expected = int(args.expected_pages) if str(args.expected_pages).strip() else None

        for pdf in sorted(input_pdf_dir.glob("*.pdf")):
            created = convert_pdf_pages_to_images(
                pdf, output_image_dir, first_page, last_page, dpi=args.dpi, expected_pages=expected
            )
            if not created:
                print(f"[INFO] Skipped (page count mismatch?): {pdf.name}")
            else:
                print(f"[INFO] Rasterized {pdf.name}: {len(created)} pages → {output_image_dir}")

    # Process JPEGs
    if output_image_dir is None or not output_image_dir.exists():
        sys.exit("ERROR: OUTPUT_IMAGE_DIR must exist with images to process (or point --images-glob to a valid folder).")

    front_box = parse_box(args.front_crop_box)
    back_box = parse_box(args.back_crop_box)

    params = {
        "adaptive_block_size": args.adaptive_block_size,
        "adaptive_c": args.adaptive_c,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "blur_sigma": args.blur_sigma,
        "thresh_value": args.thresh_value,
        "dilate_kernel": args.dilate_kernel,
        "orb_n_features": args.orb_n_features,
    }

    images = sorted(output_image_dir.glob(args.images_glob))
    if not images:
        sys.exit(f"ERROR: No images found in {output_image_dir} matching {args.images_glob}")

    for img_path in images:
        try:
            process_image_file(img_path, template_gray, front_box, back_box, output_result_dir, params)
            print(f"[OK] {img_path.name}")
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("[DONE] Processing complete.")


if __name__ == "__main__":
    main()
