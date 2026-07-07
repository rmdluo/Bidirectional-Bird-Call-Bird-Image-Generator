# Prunes the iNaturalist sound dataset (https://github.com/visipedia/inat_sounds/tree/main/2024) to just Avians

import argparse
import os
from pathlib import Path
import shutil


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

    if not args.in_place and os.path.exists(args.output):
        if args.verbose:
            print(f"{args.output} exists, deleting...")
        shutil.rmtree(args.output)

    for partition in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(args.dataset_path, partition)):
            if args.verbose:
                print(f"{partition} partition is missing; skipping")
            continue

        if not args.in_place:
            os.makedirs(os.path.join(args.output, partition))

        files = sorted(os.listdir(os.path.join(args.dataset_path, partition)))
        for file in files:
            if "Animalia_Chordata_Aves" in file:
                if args.in_place:
                    if args.verbose:
                        print(
                            f"Keeping {os.path.join(args.dataset_path, partition, file)}"
                        )
                else:
                    if args.verbose:
                        print(
                            f"Copying {os.path.join(args.dataset_path, partition, file)} to {os.path.join(args.output, partition, file)}"
                        )
                    shutil.copytree(
                        os.path.join(args.dataset_path, partition, file),
                        os.path.join(args.output, partition, file),
                    )
            else:
                if args.in_place:
                    if args.verbose:
                        print(
                            f"Deleting {os.path.join(args.dataset_path, partition, file)}"
                        )
                    shutil.rmtree(os.path.join(args.dataset_path, partition, file))
                else:
                    if args.verbose:
                        print(
                            f"Skipping {os.path.join(args.dataset_path, partition, file)}"
                        )


if __name__ == "__main__":
    main()
