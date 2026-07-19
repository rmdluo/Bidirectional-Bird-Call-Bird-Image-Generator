import os

import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets.bird_photos import BirdPhotos
from models.vae import VAE, VAEParameters


@dataclass
class PhotoVAETrainParameters:
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    num_workers: int = 16
    model_save_path: str = "photo_vae/"
    model_save_name: str = "photo_vae"
    model_save_frequency: int = 5
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    seed: int = 42
    compile: bool = True


@dataclass
class PhotoVAETrainOutputs:
    vae: VAE
    train_total_losses: list[float] = field(default_factory=list)
    train_reconstruction_losses: list[float] = field(default_factory=list)
    train_kl_losses: list[float] = field(default_factory=list)
    val_total_losses: list[float] = field(default_factory=list)
    val_reconstruction_losses: list[float] = field(default_factory=list)
    val_kl_losses: list[float] = field(default_factory=list)
    test_total_loss: float = field(default_factory=list)
    test_reconstruction_loss: float = field(default_factory=list)
    test_kl_loss: float = field(default_factory=list)


def train(
    train_photo_dataset: BirdPhotos,
    val_photo_dataset: BirdPhotos,
    test_photo_dataset: BirdPhotos,
    vae_parameters: VAEParameters,
    train_parameters: PhotoVAETrainParameters,
):
    train_generator = torch.Generator()
    train_generator.manual_seed(train_parameters.seed)
    train_dl = torch.utils.data.DataLoader(
        train_photo_dataset,
        batch_size=train_parameters.batch_size,
        shuffle=True,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=train_generator,
    )

    val_generator = torch.Generator()
    val_generator.manual_seed(train_parameters.seed)
    val_dl = torch.utils.data.DataLoader(
        val_photo_dataset,
        batch_size=train_parameters.batch_size,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=val_generator,
    )

    if test_photo_dataset is not None:
        test_generator = torch.Generator()
        test_generator.manual_seed(train_parameters.seed)
        test_dl = torch.utils.data.DataLoader(
            test_photo_dataset,
            batch_size=train_parameters.batch_size,
            num_workers=train_parameters.num_workers,
            pin_memory=True,
            generator=test_generator,
        )

    vae = VAE(vae_parameters).to(train_parameters.device)
    if train_parameters.compile:
        vae.compile()

    optimizer = torch.optim.AdamW(vae.parameters(), train_parameters.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_parameters.epochs, eta_min=train_parameters.lr / 100
    )

    train_losses, train_reconstruction_losses, train_kl_losses = [], [], []
    val_losses, val_reconstruction_losses, val_kl_losses = [], [], []
    for epoch in range(train_parameters.epochs):
        # Train
        vae.train()
        running_total_loss = 0
        running_reconstruction_loss = 0
        running_kl_loss = 0
        batches = 0
        for image, _ in tqdm(train_dl):
            image = image.to(train_parameters.device)
            optimizer.zero_grad()
            vae_output = vae(image)
            vae_loss = vae.loss(vae_output)
            vae_loss.total_loss.backward()
            optimizer.step()
            running_total_loss += vae_loss.total_loss.detach().cpu().item()
            running_reconstruction_loss += (
                vae_loss.reconstruction_loss.detach().cpu().item()
            )
            running_kl_loss += vae_loss.kl_loss.detach().cpu().item()
            batches += 1

        train_losses.append(running_total_loss / batches)
        train_reconstruction_losses.append(running_reconstruction_loss / batches)
        train_kl_losses.append(running_kl_loss / batches)

        # Val
        vae.eval()
        running_total_loss = 0
        running_reconstruction_loss = 0
        running_kl_loss = 0
        batches = 0
        for image, _ in tqdm(val_dl):
            image = image.to(train_parameters.device)
            optimizer.zero_grad()
            vae_output = vae(image)
            vae_loss = vae.loss(vae_output)
            vae_loss.total_loss.backward()
            optimizer.step()
            running_total_loss += vae_loss.total_loss.detach().cpu().item()
            running_reconstruction_loss += (
                vae_loss.reconstruction_loss.detach().cpu().item()
            )
            running_kl_loss += vae_loss.kl_loss.detach().cpu().item()
            batches += 1

        val_losses.append(running_total_loss / batches)
        val_reconstruction_losses.append(running_reconstruction_loss / batches)
        val_kl_losses.append(running_kl_loss / batches)

        scheduler.step()

        if (epoch + 1) % train_parameters.model_save_frequency == 0:
            print(
                f"Epoch {epoch+1}/{train_parameters.epochs} train results: total loss {train_losses[-1]}, reconstruction loss {train_reconstruction_losses[-1]}, kl loss {train_kl_losses[-1]}, lr {scheduler.get_last_lr()[0]}"
            )
            print(
                f"Epoch {epoch+1}/{train_parameters.epochs} val results: total loss {val_losses[-1]}, reconstruction loss {val_reconstruction_losses[-1]}, kl loss {val_kl_losses[-1]}"
            )

            if not os.path.exists(train_parameters.model_save_path):
                os.makedirs(train_parameters.model_save_path)

            torch.save(
                {
                    "train_parameters": train_parameters,
                    "model_parameters": vae_parameters,
                    "epoch": epoch,
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lrscheduler_state_dict": scheduler.state_dict(),
                    "train_losses": {
                        "total_loss": train_losses,
                        "reconstruction_loss": train_reconstruction_losses,
                        "kl_loss": train_kl_losses,
                    },
                    "val_losses": {
                        "total_loss": val_losses,
                        "reconstruction_loss": val_reconstruction_losses,
                        "kl_loss": val_kl_losses,
                    },
                    "train_generator": train_generator.get_state(),
                },
                os.path.join(
                    train_parameters.model_save_path,
                    f"{train_parameters.model_save_name}_epoch_{epoch}.tar",
                ),
            )

    if test_photo_dataset is not None:
        # Test
        vae.eval()
        running_total_loss = 0
        running_reconstruction_loss = 0
        running_kl_loss = 0
        batches = 0
        for image, _ in tqdm(test_dl):
            image = image.to(train_parameters.device)
            optimizer.zero_grad()
            vae_output = vae(image)
            vae_loss = vae.loss(vae_output)
            vae_loss.total_loss.backward()
            optimizer.step()
            running_total_loss += vae_loss.total_loss.detach().cpu().item()
            running_reconstruction_loss += (
                vae_loss.reconstruction_loss.detach().cpu().item()
            )
            running_kl_loss += vae_loss.kl_loss.detach().cpu().item()
            batches += 1

        test_loss = running_total_loss / batches
        test_reconstruction_loss = running_reconstruction_loss / batches
        test_kl_loss = running_kl_loss / batches

        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} test results: total loss {test_loss}, reconstruction loss {test_reconstruction_loss}, kl loss {test_kl_loss}"
        )

    if not os.path.exists(train_parameters.model_save_path):
        os.makedirs(train_parameters.model_save_path)

    torch.save(
        {
            "train_parameters": train_parameters,
            "model_parameters": vae_parameters,
            "epoch": epoch,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lrscheduler_state_dict": scheduler.state_dict(),
            "train_losses": {
                "total_loss": train_losses,
                "reconstruction_loss": train_reconstruction_losses,
                "kl_loss": train_kl_losses,
            },
            "val_losses": {
                "total_loss": val_losses,
                "reconstruction_loss": val_reconstruction_losses,
                "kl_loss": val_kl_losses,
            },
            "train_generator": train_generator.get_state(),
        },
        os.path.join(
            train_parameters.model_save_path,
            f"{train_parameters.model_save_name}_final.tar",
        ),
    )

    return PhotoVAETrainOutputs(
        vae,
        train_losses,
        train_reconstruction_losses,
        train_kl_losses,
        val_losses,
        val_reconstruction_losses,
        val_kl_losses,
        test_loss if test_photo_dataset is not None else None,
        test_reconstruction_loss if test_photo_dataset is not None else None,
        test_kl_loss if test_photo_dataset is not None else None,
    )
