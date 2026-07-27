import os

import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets.bird_call_photo_latent_mix import BirdCallPhotoLatentMix
from models.diffusion import Diffusion, DiffusionParameters


@dataclass
class DiffusionTrainParameters:
    lr: float = 0.001
    batch_size: int = 64
    epochs: int = 1
    num_workers: int = 16
    model_save_path: str = "photo_diffusion/"
    model_save_name: str = "photo_diffusion"
    model_save_frequency: int = 5
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    seed: int = 42
    compile: bool = True


@dataclass
class DiffusionTrainOutputs:
    diffusion: Diffusion
    train_total_losses: list[float] = field(default_factory=list)
    val_total_losses: list[float] = field(default_factory=list)
    test_total_loss: float = field(default_factory=list)


def train(
    train_dataset: BirdCallPhotoLatentMix,
    val_dataset: BirdCallPhotoLatentMix,
    test_dataset: BirdCallPhotoLatentMix,
    diffusion_parameters: DiffusionParameters,
    train_parameters: DiffusionTrainParameters,
):
    train_generator = torch.Generator()
    train_generator.manual_seed(train_parameters.seed)
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_parameters.batch_size,
        shuffle=True,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=train_generator,
    )

    val_generator = torch.Generator()
    val_generator.manual_seed(train_parameters.seed)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_parameters.batch_size,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=val_generator,
    )

    if test_dataset is not None:
        test_generator = torch.Generator()
        test_generator.manual_seed(train_parameters.seed)
        test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_parameters.batch_size,
            num_workers=train_parameters.num_workers,
            pin_memory=True,
            generator=test_generator,
        )
    else:
        test_dl = None
        test_generator = None

    diffusion = Diffusion(diffusion_parameters).to(train_parameters.device)
    if train_parameters.compile:
        diffusion.compile()

    optimizer = torch.optim.AdamW(diffusion.parameters(), train_parameters.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_parameters.epochs, eta_min=train_parameters.lr / 100
    )

    return train_loops(
        train_parameters,
        diffusion_parameters,
        diffusion,
        optimizer,
        scheduler,
        train_generator,
        train_dl,
        val_generator,
        val_dl,
        test_generator,
        test_dl,
    )


def continue_training(
    checkpoint_path: str,
    train_dataset: BirdCallPhotoLatentMix,
    val_dataset: BirdCallPhotoLatentMix,
    test_dataset: BirdCallPhotoLatentMix,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    train_parameters: DiffusionTrainParameters = checkpoint["train_parameters"]
    diffusion_parameters: DiffusionParameters = checkpoint["model_parameters"]

    completed_epoch = checkpoint["epoch"]
    start_epoch = completed_epoch + 1

    # The original training run has already finished.
    if start_epoch >= train_parameters.epochs:
        diffusion = Diffusion(diffusion_parameters).to(train_parameters.device)
        diffusion.load_state_dict(checkpoint["model_state_dict"])

        if train_parameters.compile:
            diffusion.compile()

        print(
            f"Training already complete: "
            f"{start_epoch}/{train_parameters.epochs} epochs finished."
        )

        return DiffusionTrainOutputs(
            diffusion=diffusion,
            train_total_losses=checkpoint["train_losses"]["total_loss"],
            val_total_losses=checkpoint["val_losses"]["total_loss"],
            test_total_loss=None,
        )

    train_generator = torch.Generator()

    if "train_generator" in checkpoint:
        train_generator.set_state(checkpoint["train_generator"])
    else:
        train_generator.manual_seed(train_parameters.seed)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_parameters.batch_size,
        shuffle=True,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=train_generator,
    )

    val_generator = torch.Generator()
    val_generator.manual_seed(train_parameters.seed)

    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_parameters.batch_size,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=val_generator,
    )

    if test_dataset is not None:
        test_generator = torch.Generator()
        test_generator.manual_seed(train_parameters.seed)

        test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_parameters.batch_size,
            num_workers=train_parameters.num_workers,
            pin_memory=True,
            generator=test_generator,
        )
    else:
        test_generator = None
        test_dl = None

    diffusion = Diffusion(diffusion_parameters).to(train_parameters.device)

    # Load before compiling so state-dict keys match.
    diffusion.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        train_parameters.lr,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move restored optimizer state to the configured device.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(train_parameters.device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_parameters.epochs,
        eta_min=train_parameters.lr / 100,
    )
    scheduler.load_state_dict(checkpoint["lrscheduler_state_dict"])

    if train_parameters.compile:
        diffusion.compile()

    return train_loops(
        train_parameters=train_parameters,
        diffusion_parameters=diffusion_parameters,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_generator=train_generator,
        train_dl=train_dl,
        val_generator=val_generator,
        val_dl=val_dl,
        test_generator=test_generator,
        test_dl=test_dl,
        start_epoch=start_epoch,
        train_losses=checkpoint["train_losses"]["total_loss"],
        val_losses=checkpoint["val_losses"]["total_loss"],
    )


def train_loops(
    train_parameters,
    diffusion_parameters,
    diffusion,
    optimizer,
    scheduler,
    train_generator,
    train_dl,
    val_generator,
    val_dl,
    test_generator,
    test_dl,
    start_epoch=0,
    train_losses=[],
    val_losses=[],
):
    for epoch in range(start_epoch, train_parameters.epochs):
        # Train
        diffusion.train()
        running_total_loss = 0
        batches = 0
        for latent_1, latent_2, _ in tqdm(train_dl):
            optimizer.zero_grad()

            latent_1 = latent_1.to(train_parameters.device)
            latent_2 = latent_2.to(train_parameters.device)

            # forward diffusion
            noise = diffusion.generate_noise(
                latent_1.shape[0], generator=train_generator
            ).to(train_parameters.device)
            dif_times = diffusion.generate_random_timesteps(
                latent_1.shape[0], generator=train_generator
            ).to(train_parameters.device)
            noisy_images = diffusion.generate_noisy_images(
                latent_1, noise, dif_times
            ).to(train_parameters.device)

            if (
                train_parameters.device == torch.device("cpu")
                or train_parameters.device == "cpu"
            ):
                diffusion_outputs = diffusion(noise, noisy_images, dif_times, latent_2)
                diffusion_loss = diffusion.loss(diffusion_outputs)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    diffusion_outputs = diffusion(
                        noise, noisy_images, dif_times, latent_2
                    )
                    diffusion_loss = diffusion.loss(diffusion_outputs)

            diffusion_loss.total_loss.backward()
            optimizer.step()
            running_total_loss += diffusion_loss.total_loss.detach().cpu().item()
            batches += 1

        train_losses.append(running_total_loss / batches)

        # Val
        diffusion.eval()
        running_total_loss = 0
        batches = 0
        with torch.no_grad():
            for latent_1, latent_2, _ in tqdm(val_dl):
                latent_1 = latent_1.to(train_parameters.device)
                latent_2 = latent_2.to(train_parameters.device)

                # forward diffusion
                noise = diffusion.generate_noise(
                    latent_1.shape[0], generator=val_generator
                ).to(train_parameters.device)
                dif_times = diffusion.generate_random_timesteps(
                    latent_1.shape[0], generator=val_generator
                ).to(train_parameters.device)
                noisy_images = diffusion.generate_noisy_images(
                    latent_1, noise, dif_times
                ).to(train_parameters.device)

                if (
                    train_parameters.device == torch.device("cpu")
                    or train_parameters.device == "cpu"
                ):
                    diffusion_outputs = diffusion(
                        noise, noisy_images, dif_times, latent_2
                    )
                    diffusion_loss = diffusion.loss(diffusion_outputs)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        diffusion_outputs = diffusion(
                            noise, noisy_images, dif_times, latent_2
                        )
                        diffusion_loss = diffusion.loss(diffusion_outputs)

                running_total_loss += diffusion_loss.total_loss.cpu().item()
                batches += 1

        val_losses.append(running_total_loss / batches)

        scheduler.step()

        if (epoch + 1) % train_parameters.model_save_frequency == 0:
            print(
                f"Epoch {epoch+1}/{train_parameters.epochs} train results: total loss {train_losses[-1]}, lr {scheduler.get_last_lr()[0]}"
            )
            print(
                f"Epoch {epoch+1}/{train_parameters.epochs} val results: total loss {val_losses[-1]}"
            )

            if not os.path.exists(train_parameters.model_save_path):
                os.makedirs(train_parameters.model_save_path)

            torch.save(
                {
                    "train_parameters": train_parameters,
                    "model_parameters": diffusion_parameters,
                    "epoch": epoch,
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lrscheduler_state_dict": scheduler.state_dict(),
                    "train_losses": {
                        "total_loss": train_losses,
                    },
                    "val_losses": {
                        "total_loss": val_losses,
                    },
                    "train_generator": train_generator.get_state(),
                },
                os.path.join(
                    train_parameters.model_save_path,
                    f"{train_parameters.model_save_name}_epoch_{epoch}.tar",
                ),
            )

    if test_dl is not None:
        # Test
        diffusion.eval()
        running_total_loss = 0
        batches = 0
        with torch.no_grad():
            for latent_1, latent_2, _ in tqdm(test_dl):
                latent_1 = latent_1.to(train_parameters.device)
                latent_2 = latent_2.to(train_parameters.device)

                # forward diffusion
                noise = diffusion.generate_noise(
                    latent_1.shape[0], generator=val_generator
                ).to(train_parameters.device)
                dif_times = diffusion.generate_random_timesteps(
                    latent_1.shape[0], generator=val_generator
                ).to(train_parameters.device)
                noisy_images = diffusion.generate_noisy_images(
                    latent_1, noise, dif_times
                ).to(train_parameters.device)

                if (
                    train_parameters.device == torch.device("cpu")
                    or train_parameters.device == "cpu"
                ):
                    diffusion_outputs = diffusion(
                        noise, noisy_images, dif_times, latent_2
                    )
                    diffusion_loss = diffusion.loss(diffusion_outputs)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        diffusion_outputs = diffusion(
                            noise, noisy_images, dif_times, latent_2
                        )
                        diffusion_loss = diffusion.loss(diffusion_outputs)

                running_total_loss += diffusion_loss.total_loss.cpu().item()
                batches += 1

        test_loss = running_total_loss / batches

        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} test results: total loss {test_loss}"
        )

    if not os.path.exists(train_parameters.model_save_path):
        os.makedirs(train_parameters.model_save_path)

    torch.save(
        {
            "train_parameters": train_parameters,
            "model_parameters": diffusion_parameters,
            "epoch": epoch,
            "model_state_dict": diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lrscheduler_state_dict": scheduler.state_dict(),
            "train_losses": {
                "total_loss": train_losses,
            },
            "val_losses": {
                "total_loss": val_losses,
            },
            "train_generator": train_generator.get_state(),
        },
        os.path.join(
            train_parameters.model_save_path,
            f"{train_parameters.model_save_name}_final.tar",
        ),
    )

    return DiffusionTrainOutputs(
        diffusion,
        train_losses,
        val_losses,
        test_loss if test_dl is not None else None,
    )
