# Renames the iNaturalist sound dataset (https://github.com/visipedia/inat_sounds/tree/main/2024) to common names

import argparse
import os
from pathlib import Path
import shutil
import json
import pandas as pd


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
        default=Path("iNaturalistNameConverted"),
        help="Path to the output. Default: iNaturalistNameConverted",
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

        if not os.path.exists(os.path.join(args.dataset_path, partition + ".json")):
            if args.verbose:
                print(f"{partition} json is missing; skipping")
            continue

        if not args.in_place:
            os.makedirs(os.path.join(args.output, partition))
            shutil.copy2(
                os.path.join(args.dataset_path, partition + ".json"),
                os.path.join(args.output, partition + ".json"),
            )

        with open(
            os.path.join(args.dataset_path, partition + ".json")
        ) as partition_json:
            original_data = json.loads(partition_json.read())
            data = [
                {
                    "common_name": datum["common_name"],
                    "image_dir_name": datum["image_dir_name"],
                }
                for datum in original_data["categories"]
            ]
            data = pd.DataFrame.from_records(data)

        files = sorted(os.listdir(os.path.join(args.dataset_path, partition)))
        for file in files:
            common_name = data[data["image_dir_name"] == file]["common_name"].item()
            if args.in_place:
                if args.verbose:
                    print(
                        f"Renaming {os.path.join(args.dataset_path, partition, file)} to {os.path.join(args.dataset_path, partition, common_name)}"
                    )
                shutil.move(
                    os.path.join(args.dataset_path, partition, file),
                    os.path.join(args.dataset_path, partition, common_name),
                )
            else:
                if args.verbose:
                    print(
                        f"Copying {os.path.join(args.dataset_path, partition, file)} to {os.path.join(args.output, partition, common_name)}"
                    )
                shutil.copytree(
                    os.path.join(args.dataset_path, partition, file),
                    os.path.join(args.output, partition, common_name),
                )


if __name__ == "__main__":
    main()
