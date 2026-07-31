#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect image edges using the OpenCV Canny algorithm."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Default: <input_name>_edges.png",
    )

    parser.add_argument(
        "--threshold-low",
        type=int,
        default=100,
        help="Lower Canny threshold. Default: 100",
    )

    parser.add_argument(
        "--threshold-high",
        type=int,
        default=200,
        help="Upper Canny threshold. Default: 200",
    )

    parser.add_argument(
        "--blur",
        type=int,
        default=5,
        help="Gaussian blur kernel size. Must be an odd number. Default: 5",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise ValueError(f"Input image does not exist: {args.image}")

    if not 0 <= args.threshold_low <= 255:
        raise ValueError("--threshold-low must be between 0 and 255")

    if not 0 <= args.threshold_high <= 255:
        raise ValueError("--threshold-high must be between 0 and 255")

    if args.threshold_low >= args.threshold_high:
        raise ValueError(
            "--threshold-low must be smaller than --threshold-high"
        )

    if args.blur < 1 or args.blur % 2 == 0:
        raise ValueError("--blur must be a positive odd number")


def detect_edges(
    input_path: Path,
    output_path: Path,
    threshold_low: int,
    threshold_high: int,
    blur_size: int,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"OpenCV could not decode the image: {input_path}"
        )

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        grayscale,
        (blur_size, blur_size),
        sigmaX=0,
    )

    edges = cv2.Canny(
        blurred,
        threshold1=threshold_low,
        threshold2=threshold_high,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), edges):
        raise RuntimeError(f"Could not save output image: {output_path}")

    edge_pixels = cv2.countNonZero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_pixels / total_pixels * 100

    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Resolution:  {edges.shape[1]}x{edges.shape[0]}")
    print(f"Edge pixels: {edge_pixels} ({edge_ratio:.2f}%)")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

        output_path = args.output or args.image.with_name(
            f"{args.image.stem}_edges.png"
        )

        detect_edges(
            input_path=args.image,
            output_path=output_path,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            blur_size=args.blur,
        )

        return 0

    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())