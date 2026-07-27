# Prunes the iNaturalist sound dataset
# (https://github.com/visipedia/inat_sounds/tree/main/2024)
# to images containing birds, then crops each image to the
# highest-confidence detected bird.

import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep images containing a bird and crop each image to the "
            "highest-confidence bird detection."
        )
    )

    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the original dataset.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("iNaturalistPrunedCropped"),
        help="Path to the output. Default: iNaturalistPrunedCropped",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Prune and crop images in place. Images without a bird are deleted, "
            "and retained images are replaced by their processed versions."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the actions that would be taken without modifying files.",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="yolo26m.pt",
        help="YOLO model to use. Default: yolo26m.pt",
    )

    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=0.5,
        help="Minimum bird-detection confidence. Default: 0.5",
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help=(
            "Fractional padding added around the detected bird before square "
            "padding. For example, 0.1 adds 10%% of the bounding-box width and "
            "height to each side. Default: 0.0"
        ),
    )

    parser.add_argument(
        "--square-pad",
        action="store_true",
        help=(
            "Center the cropped bird image on a white square canvas without "
            "changing its aspect ratio."
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="PIXELS",
        help=(
            "Resize retained images to PIXELS x PIXELS. Specifying this option "
            "also enables white square padding."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output without disabling the progress bar.",
    )

    return parser.parse_args()


def find_best_bird_box(
    prediction,
    confidence_threshold: float,
) -> Optional[tuple[list[float], float]]:
    """
    Return the bounding box and confidence for the highest-confidence bird.

    The bounding box is returned in xyxy format:
        [left, top, right, bottom]

    Returns None when no qualifying bird is found.
    """
    if prediction.boxes is None or len(prediction.boxes) == 0:
        return None

    best_box: Optional[list[float]] = None
    best_confidence = confidence_threshold

    for box in prediction.boxes:
        class_id = int(box.cls.item())
        class_name = prediction.names[class_id]
        confidence = float(box.conf.item())

        if class_name == "bird" and confidence >= best_confidence:
            best_confidence = confidence
            best_box = box.xyxy[0].tolist()

    if best_box is None:
        return None

    return best_box, best_confidence


def crop_to_box(
    image: np.ndarray,
    box: list[float],
    padding: float,
) -> Optional[np.ndarray]:
    """
    Crop an image to an xyxy bounding box.

    Fractional padding is added around each side of the box and constrained
    to the original image boundaries.
    """
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    horizontal_padding = box_width * padding
    vertical_padding = box_height * padding

    x1 = max(0, int(np.floor(x1 - horizontal_padding)))
    y1 = max(0, int(np.floor(y1 - vertical_padding)))
    x2 = min(image_width, int(np.ceil(x2 + horizontal_padding)))
    y2 = min(image_height, int(np.ceil(y2 + vertical_padding)))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()


def pad_to_square(
    image: np.ndarray,
    value: int = 255,
) -> np.ndarray:
    """
    Center an image on a square canvas.

    The canvas is white by default. Grayscale, BGR, and BGRA images are
    supported.
    """
    height, width = image.shape[:2]
    square_size = max(height, width)

    top = (square_size - height) // 2
    bottom = square_size - height - top
    left = (square_size - width) // 2
    right = square_size - width - left

    if image.ndim == 2:
        border_value = value
    else:
        border_value = tuple([value] * image.shape[2])

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=border_value,
    )


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize an already-square image to size x size.
    """
    current_height, current_width = image.shape[:2]

    if current_height == size and current_width == size:
        return image

    if size < max(current_height, current_width):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(
        image,
        (size, size),
        interpolation=interpolation,
    )


def process_crop(
    image: np.ndarray,
    box: list[float],
    crop_padding: float,
    square_pad: bool,
    output_size: Optional[int],
) -> Optional[np.ndarray]:
    """
    Crop to the bird, optionally pad to a white square, and optionally resize.
    """
    processed = crop_to_box(
        image=image,
        box=box,
        padding=crop_padding,
    )

    if processed is None or processed.size == 0:
        return None

    # A square output size requires square padding first so that the image
    # is not stretched or distorted.
    if square_pad or output_size is not None:
        processed = pad_to_square(processed, value=255)

    if output_size is not None:
        processed = resize_square(processed, output_size)

    return processed


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_path = args.dataset_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0.0 and 1.0")

    if args.padding < 0.0:
        raise ValueError("--padding must be greater than or equal to 0.0")

    if args.size is not None and args.size <= 0:
        raise ValueError("--size must be greater than zero")

    if not args.in_place and output_path == dataset_path:
        raise ValueError(
            "The output path cannot equal the dataset path unless "
            "--in-place is used."
        )

    # Prevent an output directory inside the input directory from being
    # accidentally included when the input is scanned.
    if not args.in_place and output_path.is_relative_to(dataset_path):
        raise ValueError(
            "The output directory cannot be inside the dataset directory. "
            "Choose an output path outside the input dataset."
        )

    return dataset_path, output_path


def main() -> None:
    args = parse_args()
    dataset_path, output_path = validate_args(args)

    if args.verbose:
        print(f"Arguments: {args}")

    print(f"Input: {dataset_path}")
    print(f"Output: {'in place' if args.in_place else output_path}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Crop padding: {args.padding}")
    print(f"Square padding: {args.square_pad or args.size is not None}")
    print(f"Output size: {args.size or 'unchanged'}")
    print(f"Dry run: {args.dry_run}")

    model = YOLO(args.model)

    if not args.in_place and output_path.exists():
        if args.verbose:
            tqdm.write(f"Removing existing output directory: {output_path}")

        if not args.dry_run:
            shutil.rmtree(output_path)

    if not args.in_place and not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    kept_count = 0
    removed_count = 0
    failed_count = 0

    progress = tqdm(
        files,
        desc="Processing images",
        unit="image",
        dynamic_ncols=True,
    )

    for file in progress:
        progress.set_postfix(
            kept=kept_count,
            rejected=removed_count,
            failed=failed_count,
        )

        image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

        if image is None:
            failed_count += 1
            tqdm.write(f"Warning: Could not read image: {file}")
            continue

        # YOLO generally expects grayscale or three-channel input rather than
        # an image with an alpha channel.
        if image.ndim == 2:
            inference_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            inference_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            inference_image = image

        try:
            prediction = model(inference_image, verbose=False)[0]
        except Exception as error:
            failed_count += 1
            tqdm.write(f"Warning: Inference failed for {file}: {error}")
            continue

        bird_detection = find_best_bird_box(
            prediction,
            confidence_threshold=args.conf,
        )

        if bird_detection is None:
            removed_count += 1

            if args.in_place:
                if args.verbose:
                    tqdm.write(f"Deleting image without bird: {file}")

                if not args.dry_run:
                    try:
                        file.unlink()
                    except OSError as error:
                        failed_count += 1
                        tqdm.write(f"Warning: Could not delete {file}: {error}")
            elif args.verbose:
                tqdm.write(f"Skipping image without bird: {file}")

            continue

        box, confidence = bird_detection

        processed_image = process_crop(
            image=image,
            box=box,
            crop_padding=args.padding,
            square_pad=args.square_pad,
            output_size=args.size,
        )

        if processed_image is None:
            failed_count += 1
            tqdm.write(f"Warning: Invalid crop generated for: {file}")
            continue

        if args.in_place:
            destination = file
            action = "Processing in place"
        else:
            relative_path = file.relative_to(dataset_path)
            destination = output_path / relative_path
            action = "Saving processed image"

        if args.verbose:
            height, width = processed_image.shape[:2]
            tqdm.write(
                f"{action}: {file} -> {destination} "
                f"(confidence={confidence:.3f}, "
                f"box={[round(value, 1) for value in box]}, "
                f"output={width}x{height})"
            )

        if not args.dry_run:
            try:
                save_image(destination, processed_image)
            except (OSError, RuntimeError) as error:
                failed_count += 1
                tqdm.write(f"Warning: Could not save {destination}: {error}")
                continue

        kept_count += 1

    progress.set_postfix(
        kept=kept_count,
        rejected=removed_count,
        failed=failed_count,
    )

    print()
    print(f"Images scanned: {len(files)}")
    print(f"Bird images retained: {kept_count}")
    print(f"Images without birds: {removed_count}")
    print(f"Images that could not be processed: {failed_count}")


if __name__ == "__main__":
    main()
