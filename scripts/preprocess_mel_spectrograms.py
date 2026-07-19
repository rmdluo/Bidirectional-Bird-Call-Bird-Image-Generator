import argparse
import os
from pathlib import Path
import shutil
import torch
import torchaudio
import torch.nn.functional as F

import sys

sys.path.append("../src")
from processing.audio import waveform_to_log_mel, log_mel_to_waveform


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
        "-o",
        "--output",
        type=Path,
        default=Path("iNaturalistAudioPreprocessed"),
        help="Path to the output. Default: iNaturalistAudioPreprocessed",
    )

    parser.add_argument(
        "-s",
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate to resample to. Default: 16000",
    )

    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=5,
        help="Number of seconds to use. Default: 5",
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

    audio_output_folder = args.output

    print(f"Input: {args.audio_dataset_path}")
    print(f"Output: {audio_output_folder}")
    print(f"In place: {args.in_place}")

    if not args.dry_run and not args.in_place and os.path.exists(audio_output_folder):
        if args.verbose:
            print(f"{audio_output_folder} exists, deleting...")
        if not args.dry_run:
            shutil.rmtree(audio_output_folder)

    files = sorted(list(args.audio_dataset_path.glob("**/*.wav")))
    for file in files:
        waveform, sample_rate = torchaudio.load(file, normalize=True)

        if waveform.numel() == 0 or waveform.shape[-1] == 0:
            if args.verbose:
                print(f"Skipping empty audio file: {file}")
            continue

        log_mel = waveform_to_log_mel(
            waveform, sample_rate, args.sample_rate, args.time
        )

        if args.in_place:
            if args.verbose:
                print(
                    f"Log mel'ing '{file.resolve().as_uri()}' to '{file.with_suffix('.pt').resolve().as_uri()}'"
                )
            if not args.dry_run:
                torch.save(log_mel, file.with_suffix(".pt"))
        else:
            relative_path = file.relative_to(args.audio_dataset_path)
            destination = (args.output / relative_path).with_suffix(".pt")
            if args.verbose:
                print(
                    f"Saving log mel'd '{file.resolve().as_uri()}' to {destination.resolve().as_uri()}"
                )
            if not args.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(log_mel, destination)


if __name__ == "__main__":
    main()
