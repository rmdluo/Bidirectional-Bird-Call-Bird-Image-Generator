import os
import logging
import random
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BirdCallPhotoMix(Dataset):
    """Dataset that pairs bird calls and bird photos"""

    def __init__(
        self,
        audio_dir,
        photo_dir,
        mixing_mode="one-to-one",
        num_pairs_per_species=50,
        random_seed=None,
        **kwargs,
    ):
        """PyTorch dataset that links audio to photos, matched on species.

        Positional arguments:
        audio_dir -- the directory that holds the audio samples. Assumes the audio files are partitioned by species.
        photo_dir -- the directory that holds the photo samples. Assumes the photo files are partitioned by species.

        Keyword arguments:
        mixing_mode -- whether to use one-to-one, one-to-many, many-to-one, or many-to-many pairings
            - one-to-one: each audio sample gets paired to a single image
            - one-to-many: each audio sample gets paired to multiple images, but each image is only paired to one audio sample
                - requires kwarg `num_matches`. It matches each audio sample to `num_matches` photo samples.
                  If there are no more unique photo samples, then audio files can be left out.
                  If there are are less photo samples left than `num_matches`, then the next audio
                  sample gets matched to the remaining photo samples.
            - many-to-one: each audio sample gets paired to one image, but each image can be paired to multiple audio samples
                - requires kwarg `num_matches`. It matches each photo sample to `num_matches` audio samples.
                  If there are no more unique audio samples, then photo files can be left out.
                  If there are are less audio samples left than `num_matches`, then the next photo
                  sample gets matched to the remaining audio samples.
            - many-to-many: no constraints, both audio and photo samples may be paired with multiple of the other modality
        num_pairs_per_species -- limits the number of pairs generated per species. Be careful to not specify too few pairs. Defaults to 50.
        random_seed -- random seed to use, used in sampling for one-to-many, many-to-one, and many-to-many. Defaults to None.
        """
        logger.info(
            f"Constructing BirdCallPhotoMix dataset from audio directory `{audio_dir}` and photo directory `{photo_dir}`."
        )
        self.audio_dir, self.photo_dir = audio_dir, photo_dir
        self.rng = random.Random(random_seed)

        # Extract species
        audio_species = set(os.listdir(audio_dir))
        photo_species = set(os.listdir(photo_dir))
        shared_species = audio_species.intersection(photo_species)
        logger.info(f"Found {len(shared_species)} shared species.")

        # Generate pairings
        self.pairs = {
            "species": [],
            "photo_filepath": [],
            "audio_filepath": [],
        }
        for species in sorted(shared_species):
            species_audio_files = sorted(os.listdir(os.path.join(audio_dir, species)))
            species_photo_files = sorted(os.listdir(os.path.join(photo_dir, species)))
            if mixing_mode == "one-to-one":
                for i in range(
                    min(
                        len(species_audio_files),
                        len(species_photo_files),
                        num_pairs_per_species,
                    )
                ):
                    self.pairs["species"].append(species)
                    self.pairs["photo_filepath"].append(
                        os.path.join(photo_dir, species, species_photo_files[i])
                    )
                    self.pairs["audio_filepath"].append(
                        os.path.join(audio_dir, species, species_audio_files[i])
                    )
            elif mixing_mode == "one-to-many":
                num_matches = kwargs["num_matches"]
                num_pairs = 0
                species_photo_files = set(species_photo_files)
                for audio_file in species_audio_files:
                    photo_files = self.rng.sample(
                        sorted(species_photo_files),
                        min(num_matches, len(species_photo_files)),
                    )
                    species_photo_files = species_photo_files.difference(photo_files)
                    for photo_file in photo_files:
                        self.pairs["species"].append(species)
                        self.pairs["photo_filepath"].append(
                            os.path.join(photo_dir, species, photo_file)
                        )
                        self.pairs["audio_filepath"].append(
                            os.path.join(audio_dir, species, audio_file)
                        )

                        num_pairs += 1
                        if num_pairs == num_pairs_per_species:
                            break
                    if (
                        len(species_photo_files) == 0
                        or num_pairs == num_pairs_per_species
                    ):
                        break
            elif mixing_mode == "many-to-one":
                num_matches = kwargs["num_matches"]
                num_pairs = 0
                species_audio_files = set(species_audio_files)
                for photo_file in species_photo_files:
                    audio_files = self.rng.sample(
                        sorted(species_audio_files),
                        min(num_matches, len(species_audio_files)),
                    )
                    species_audio_files = species_audio_files.difference(audio_files)
                    for audio_file in audio_files:
                        self.pairs["species"].append(species)
                        self.pairs["photo_filepath"].append(
                            os.path.join(photo_dir, species, photo_file)
                        )
                        self.pairs["audio_filepath"].append(
                            os.path.join(audio_dir, species, audio_file)
                        )

                        num_pairs += 1
                        if num_pairs == num_pairs_per_species:
                            break
                    if (
                        len(species_audio_files) == 0
                        or num_pairs == num_pairs_per_species
                    ):
                        break
            elif mixing_mode == "many-to-many":
                possible_pairs = [
                    (photo_file, audio_file)
                    for photo_file in species_photo_files
                    for audio_file in species_audio_files
                ]

                species_pairs = self.rng.sample(
                    possible_pairs,
                    min(num_pairs_per_species, len(possible_pairs)),
                )

                for photo_file, audio_file in species_pairs:
                    self.pairs["species"].append(species)
                    self.pairs["photo_filepath"].append(
                        os.path.join(photo_dir, species, photo_file)
                    )
                    self.pairs["audio_filepath"].append(
                        os.path.join(audio_dir, species, audio_file)
                    )
            else:
                raise ValueError(
                    f"Invalid mixing mode: {mixing_mode}. Must be one-to-one, one-to-many, many-to-one, or many-to-many"
                )

        self.pairs = pd.DataFrame(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
