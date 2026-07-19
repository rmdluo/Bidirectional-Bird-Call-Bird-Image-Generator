import os
import logging
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BirdAudio(Dataset):
    """Dataset for bird calls"""

    def __init__(
        self,
        audio_dir,
        split="train",
        from_spectrogram=True,
        audio_transform=None,
        mel_spectrogram_transform=None,
        target_transform=None,
        max_pairs=None,
        **kwargs,
    ):
        """PyTorch dataset for bird calls.

        Positional arguments:
        audio_dir -- the directory that holds the audio samples. Assumes the audio files are partitioned by species.
        """
        logger.info(
            f"Constructing BirdAudio dataset from audio directory `{audio_dir}`."
        )
        self.audio_dir = audio_dir
        self.audio_transform, self.mel_spectrogram_transform, self.target_transform = (
            audio_transform,
            mel_spectrogram_transform,
            target_transform,
        )
        self.from_spectrogram = from_spectrogram

        # Extract species
        audio_species = set(
            [
                name
                for name in os.listdir(os.path.join(audio_dir, split))
                if os.path.isdir(os.path.join(audio_dir, split, name))
            ]
        )
        logger.info(f"Found {len(audio_species)} species.")

        # Generate pairings
        if from_spectrogram:
            extension = ".pt"
        else:
            extension = ".wav"
        self.pairs = {
            "species": [],
            "audio_filepath": [],
        }
        for species in sorted(audio_species):
            species_audio_files = sorted(
                [
                    file
                    for file in os.listdir(os.path.join(audio_dir, split, species))
                    if file.endswith(extension)
                ]
            )
            for audio_file in species_audio_files:
                self.pairs["species"].append(species)
                self.pairs["audio_filepath"].append(
                    os.path.join(audio_dir, split, species, audio_file)
                )
                if (
                    max_pairs is not None
                    and len(self.pairs["audio_filepath"]) >= max_pairs
                ):
                    break
            if max_pairs is not None and len(self.pairs["audio_filepath"]) >= max_pairs:
                break

        self.pairs = pd.DataFrame(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]

        species = pair["species"]
        if self.target_transform is not None:
            species = self.target_transform(species)

        if not self.from_spectrogram:
            waveform, sample_rate = torchaudio.load(
                pair["audio_filepath"], normalize=True
            )
            if self.audio_transform is not None:
                waveform, sample_rate = self.audio_transform(waveform, sample_rate)
            return (waveform, sample_rate), species
        else:
            log_mel = torch.load(pair["audio_filepath"], weights_only=True)
            if self.mel_spectrogram_transform is not None:
                log_mel = self.mel_spectrogram_transform(log_mel)
            return log_mel, species
