# Prunes the iNaturalist sound dataset (https://github.com/visipedia/inat_sounds/tree/main/2024) to just Avians

import argparse
import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brief description of what this script does."
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
        default=Path("iNaturalistPruned"),
        help="Path to the output. Default: iNatPruned",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Enable pruning in-place (deletes folders in original dataset)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do a dry run, where files will not be moved or deleted",
    )

    parser.add_argument("-m", "--model", default="yolo26m.pt", help="YOLO model to use")

    parser.add_argument(
        "-c", "--conf", type=float, default=0.5, help="Confidence threshold"
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

    print(f"Input: {args.dataset_path}")
    print(f"Output: {args.output}")
    print(f"In place: {args.in_place}")
    print(f"Model: {args.model}")

    model = YOLO(args.model)

    if not args.in_place and os.path.exists(args.output):
        if args.verbose:
            print(f"{args.output} exists, deleting...")
        if not args.dry_run:
            shutil.rmtree(args.output)

    if not args.in_place and not args.dry_run:
        os.makedirs(os.path.join(args.output))

    files = (
        list(args.dataset_path.glob("**/*.jpg"))
        + list(args.dataset_path.glob("**/*.png"))
        + list(args.dataset_path.glob("**/*.jpeg"))
    )
    for file in files:
        img = cv2.imread(file)
        predictions = model(img, verbose=False)[0]
        classes = [predictions.names[cls.item()] for cls in predictions.boxes.cls.int()]
        confs = predictions.boxes.conf
        if (
            max([confs[i] for i in range(len(confs)) if classes[i] == "bird"] + [0])
            > args.conf
        ):
            if args.in_place:
                if args.verbose:
                    print(f"Keeping '{file.resolve().as_uri()}'")
            else:
                relative_path = file.relative_to(args.dataset_path)
                destination = args.output / relative_path
                if args.verbose:
                    print(
                        f"Copying '{file.resolve().as_uri()}' to {destination.resolve().as_uri()}"
                    )
                if not args.dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, destination)
        else:
            if args.in_place:
                if args.verbose:
                    print(f"Deleting '{file.resolve().as_uri()}'")
                if not args.dry_run:
                    os.remove(file)
                else:
                    if args.verbose:
                        print(
                            f"--dry-run is set, skipping deleting '{file.resolve().as_uri()}'"
                        )
            else:
                if args.verbose:
                    print(f"Skipping '{file.resolve().as_uri()}'")


if __name__ == "__main__":
    main()
