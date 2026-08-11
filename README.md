# Bidirectional-Bird-Call-Bird-Image-Generator

## Environment Setup
```bash
conda env create -f environment.yaml
```

## Prepare Dataset

The full dataset (preprocessed and latents) is available at https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rluo15_jh_edu/IQCEm53x2HKqQb22S9wMaJGzAeE_AVNClyl6p2F1jb8AdLg?e=Nj2gi8. Create a folder called data at the base of this project (next to src). Download the file to data, then cd into data and do `tar -xf birdcallbirdphoto_dataset_final.tar`. It should create `iNaturalistLatentsAudio`, `iNaturalistLatentsContrastiveAudio`, `iNaturalistLatentsPhoto`, `iNaturalistPrunedCropped`, and `iNaturalistPrunedPreprocessedAudio`.

Otherwise, to prepare the dataset from scratch, download the iNaturalist 2021 Competition Dataset (https://github.com/visipedia/inat_comp/tree/master/2021) and the iNaturalist Sounds Dataset (https://github.com/visipedia/inat_sounds/tree/main/2024). Then, run the following from the scripts directory:
 - prune_inat.py (both audio and photo)
 - rename_inat_audio_to_common_name.py (audio only)
 - rename_inat_photo_to_common_name.py (photo only)
 - prune_to_shared.py (both audio and photo)
 - crop_with_yolo.py (photo only)
 - preprocess_mel_spectrograms.py (audio only)

After training the VAEs using
 - train_audio_vae.py
 - train_photo_vae.py
 - train_audio_vae_contrastive.py

use generate_audio_latents.py and generate_photo_latents.py to pregenerate the audio latents. Then train the diffusion models using
 - train_audio_diffusion.py
 - train_photo_diffusion.py
 - train_audio_diffusion_contrastive.py
 - train_photo_diffusion_contrastive.py

Note that you will likely need to adjust arguments and input paths for each of these scripts. See each script for usage instructions.

## Dataset Sample [DEPRECATED]

To look at the sample data, run `dataset_sample.ipynb`.

Note: this has been deprecated as the dataset classes have evolved. The original notebook is left as is if you want to look.

## Training

Updating the constants as appropriate at the top of each file, run the following from the scripts directory:
 - train_audio_vae.py
 - train_photo_vae.py
 - train_audio_vae_contrastive.py
 - train_audio_diffusion.py
 - train_photo_diffusion.py
 - train_audio_diffusion_contrastive.py
 - train_photo_diffusion_contrastive.py

You can then use each of the test_*.ipynb files to test the trained files. Again, make sure to update any relevant constants.

Pretrained weights can be found at https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/rluo15_jh_edu/IQDwvkteZSOKTq0P_UvSSMFLAZ71BXw18l_Q0jaELZyv3P4?e=7lb0n7. Download models.tar to the base of this project (next to src) and run `tar -xf models.tar`. It should create a models folder. Each of the test_*.ipynb files were originally run with these models.
