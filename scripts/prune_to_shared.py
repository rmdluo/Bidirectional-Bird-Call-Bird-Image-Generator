import argparse
import os
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brief description of what this script does."
    )

    parser.add_argument(
        "audio_dataset_path",
        type=Path,
        help="Path to the audio dataset.",
    )

    parser.add_argument(
        "photo_dataset_path",
        type=Path,
        help="Path to the photo dataset.",
    )

    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=Path("iNaturalistPruned"),
        help="Path to the output. Default: iNaturalistPruned",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Enable pruning in-place (deletes folders in original datasets)",
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

    if not args.in_place and not args.output_prefix:
        print("Please specify an output path")
        exit(1)

    audio_output_folder = Path(f"{args.output_prefix}Audio")
    photo_output_folder = Path(f"{args.output_prefix}Photo")

    print(f"Input: {args.audio_dataset_path}")
    print(f"Input: {args.photo_dataset_path}")
    print(f"Output: {audio_output_folder}")
    print(f"Output: {photo_output_folder}")
    print(f"In place: {args.in_place}")

    if not args.dry_run and not args.in_place and os.path.exists(audio_output_folder):
        if args.verbose:
            print(f"{audio_output_folder} exists, deleting...")
        if not args.dry_run:
            shutil.rmtree(audio_output_folder)

    if not args.dry_run and not args.in_place and os.path.exists(photo_output_folder):
        if args.verbose:
            print(f"{photo_output_folder} exists, deleting...")
        if not args.dry_run:
            shutil.rmtree(photo_output_folder)

    shared_species = None
    for split in ["train", "val", "test"]:
        if not os.path.exists(args.photo_dataset_path / split) or not os.path.exists(
            args.audio_dataset_path / split
        ):
            if args.verbose:
                print(
                    f"Spilt {split} does not exist for photo dataset or audio dataset; skipping"
                )
            continue

        photo_species = set(os.listdir(args.photo_dataset_path / split))
        audio_species = set(os.listdir(args.audio_dataset_path / split))

        if shared_species is None:
            shared_species = photo_species.intersection(audio_species)
        else:
            shared_species = shared_species.intersection(photo_species).intersection(
                audio_species
            )

    for split in ["train", "val", "test"]:
        if not os.path.exists(args.photo_dataset_path / split) or not os.path.exists(
            args.audio_dataset_path / split
        ):
            if args.verbose:
                print(
                    f"Spilt {split} does not exist for photo dataset or audio dataset; skipping"
                )
            continue

        photo_species = set(os.listdir(args.photo_dataset_path / split))
        audio_species = set(os.listdir(args.audio_dataset_path / split))

        if not args.in_place and not args.dry_run:
            os.makedirs(os.path.join(photo_output_folder, split))
            os.makedirs(os.path.join(audio_output_folder, split))

        if args.in_place:
            for species in sorted(photo_species.difference(shared_species)):
                if args.verbose:
                    print(f"Removing {species} from {args.photo_dataset_path / split}")
                if not args.dry_run:
                    shutil.rmtree(os.path.join(args.photo_dataset_path, split, species))
        else:
            for species in sorted(shared_species):
                if args.verbose:
                    print(
                        f"Copying {species} from {args.photo_dataset_path / split} to {photo_output_folder / split}"
                    )
                if not args.dry_run:
                    shutil.copytree(
                        os.path.join(args.photo_dataset_path, split, species),
                        os.path.join(photo_output_folder, split, species),
                    )

        if args.in_place:
            for species in sorted(audio_species.difference(shared_species)):
                if args.verbose:
                    print(f"Removing {species} from {args.audio_dataset_path / split}")
                if not args.dry_run:
                    shutil.rmtree(os.path.join(args.audio_dataset_path, split, species))
        else:
            for species in sorted(shared_species):
                if args.verbose:
                    print(
                        f"Copying {species} from {args.audio_dataset_path / split} to {audio_output_folder / split}"
                    )
                if not args.dry_run:
                    shutil.copytree(
                        os.path.join(args.audio_dataset_path, split, species),
                        os.path.join(audio_output_folder, split, species),
                    )


if __name__ == "__main__":
    main()
