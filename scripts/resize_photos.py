import argparse
import os
from pathlib import Path
import shutil
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brief description of what this script does."
    )

    parser.add_argument(
        "photo_dataset_path",
        type=Path,
        help="Path to the photo dataset.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("iNaturalistPhotosResized"),
        help="Path to the output. Default: iNaturalistPhotosResized",
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=128,
        help="Size to resize the images to. Default: 128",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite images in the original dataset",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do a dry run, where files will not be moved or deleted",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"Arguments: {args}")

    if not args.in_place and not args.output:
        print("Please specify an output path")
        exit(1)

    photo_output_folder = args.output

    print(f"Input: {args.photo_dataset_path}")
    print(f"Output: {photo_output_folder}")
    print(f"In place: {args.in_place}")

    if not args.dry_run and not args.in_place and os.path.exists(photo_output_folder):
        if args.verbose:
            print(f"{photo_output_folder} exists, deleting...")
        if not args.dry_run:
            shutil.rmtree(photo_output_folder)

    files = (
        list(args.photo_dataset_path.glob("**/*.jpg"))
        + list(args.photo_dataset_path.glob("**/*.png"))
        + list(args.photo_dataset_path.glob("**/*.jpeg"))
    )
    for file in files:
        img = cv2.imread(str(file))
        if img is None:
            print(f"Warning: unable to read {file}")
            continue
        resized_img = cv2.resize(
            img, (args.size, args.size), interpolation=cv2.INTER_AREA
        )

        if args.in_place:
            if args.verbose:
                print(f"Resizing '{file.resolve().as_uri()}'")
            if not args.dry_run:
                cv2.imwrite(str(file), resized_img)
        else:
            relative_path = file.relative_to(args.photo_dataset_path)
            destination = args.output / relative_path
            if args.verbose:
                print(
                    f"Saving resized '{file.resolve().as_uri()}' to {destination.resolve().as_uri()}"
                )
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destination), resized_img)


if __name__ == "__main__":
    main()
