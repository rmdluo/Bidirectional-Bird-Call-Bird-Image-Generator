import os

import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from datasets.bird_call_photo_mix import BirdCallPhotoMix
from models.vae import VAE, VAEParameters


@dataclass
class ContrastiveVAETrainParameters:
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    num_workers: int = 16
    infonce_logvar_weight: float = 0.0
    infonce_temperature: float = 0.1
    latent_mean_mae_loss_weight: float = 0.01
    latent_logvar_mae_loss_weight: float = 0.01
    contrastive_loss_weight: float = 0.01
    model_save_path: str = "contrastive_audio_vae/"
    model_save_name: str = "contrastive_audio_vae"
    model_save_frequency: int = 5
    device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    seed: int = 42
    compile: bool = True


@dataclass
class ContrastiveVAETrainOutputs:
    vae: VAE
    train_total_losses: list[float] = field(default_factory=list)
    train_reconstruction_losses: list[float] = field(default_factory=list)
    train_kl_losses: list[float] = field(default_factory=list)
    train_latent_mean_mae_losses: list[float] = field(default_factory=list)
    train_latent_logvar_mae_losses: list[float] = field(default_factory=list)
    train_contrastive_losses: list[float] = field(default_factory=list)
    val_total_losses: list[float] = field(default_factory=list)
    val_reconstruction_losses: list[float] = field(default_factory=list)
    val_kl_losses: list[float] = field(default_factory=list)
    val_latent_mean_mae_losses: list[float] = field(default_factory=list)
    val_latent_logvar_mae_losses: list[float] = field(default_factory=list)
    val_contrastive_losses: list[float] = field(default_factory=list)
    test_total_loss: float = None
    test_reconstruction_loss: float = None
    test_kl_loss: float = None
    test_latent_mean_mae_loss: float = None
    test_latent_logvar_mae_loss: float = None
    test_contrastive_loss: float = None


def calculate_infonce(
    audio_means,
    audio_logvars,
    photo_means,
    photo_logvars,
    logvar_weight=0.0,
    temperature=0.1,
):
    assert len(audio_means.shape) == 4 and len(audio_logvars.shape) == 4
    assert len(photo_means.shape) == 4 and len(photo_logvars.shape) == 4

    flattened_normed_audio_means = torch.nn.functional.normalize(
        audio_means.flatten(1), dim=1
    )
    flattened_normed_audio_logvars = torch.nn.functional.normalize(
        audio_logvars.flatten(1), dim=1
    )

    flattened_normed_photo_means = torch.nn.functional.normalize(
        photo_means.flatten(1), dim=1
    )
    flattened_normed_photo_logvars = torch.nn.functional.normalize(
        photo_logvars.flatten(1), dim=1
    )

    mean_similarities = flattened_normed_audio_means @ flattened_normed_photo_means.T
    logvar_similarities = (
        flattened_normed_audio_logvars @ flattened_normed_photo_logvars.T
    )

    combined_similarities = mean_similarities + logvar_weight * logvar_similarities
    scores = torch.exp(combined_similarities / temperature)

    positive_scores = scores.diag()

    return -(torch.log(positive_scores / scores.sum(dim=1))).mean()


def train(
    train_dataset: BirdCallPhotoMix,
    val_dataset: BirdCallPhotoMix,
    test_dataset: BirdCallPhotoMix,
    vae_parameters: VAEParameters,
    base_vae_checkpoint_path: os.PathLike,
    train_parameters: ContrastiveVAETrainParameters,
):
    for ds in [train_dataset, val_dataset, test_dataset]:
        if ds is not None and not ds.from_spectrogram:
            raise Exception(
                "Non-spectrogram bird audio dataset passed to train_constrastive_audio_vae.py"
            )

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

    vae = VAE(vae_parameters).to(train_parameters.device)
    if train_parameters.compile:
        vae.compile()

    if not torch.cuda.is_available():
        base_vae_device = torch.device("cpu")
    elif (
        train_parameters.device == torch.device("cuda:0")
        or train_parameters.device == "cuda:0"
    ):
        base_vae_device = torch.device(
            "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        )
    else:
        base_vae_device = torch.device("cuda:0")

    print(f"Using device {base_vae_device} for base vae")

    checkpoint = torch.load(
        base_vae_checkpoint_path,
        map_location=base_vae_device,
    )
    base_vae_parameters = checkpoint["model_parameters"]

    base_vae = VAE(base_vae_parameters).to(base_vae_device)
    base_vae.load_state_dict(checkpoint["model_state_dict"])
    base_vae.eval()
    for parameter in base_vae.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(vae.parameters(), train_parameters.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_parameters.epochs, eta_min=train_parameters.lr / 100
    )

    train_losses, train_reconstruction_losses, train_kl_losses = [], [], []
    (
        train_latent_mean_mae_losses,
        train_latent_logvar_mae_losses,
        train_contrastive_losses,
    ) = ([], [], [])
    val_losses, val_reconstruction_losses, val_kl_losses = [], [], []
    (
        val_latent_mean_mae_losses,
        val_latent_logvar_mae_losses,
        val_contrastive_losses,
    ) = ([], [], [])
    for epoch in range(train_parameters.epochs):
        # Train
        vae.train()
        running_total_loss = 0
        running_reconstruction_loss = 0
        running_kl_loss = 0
        running_latent_mean_mae_loss = 0
        running_latent_logvar_mae_loss = 0
        running_contrastive_loss = 0
        batches = 0
        for logmel, image, species in tqdm(train_dl):
            optimizer.zero_grad()
            logmel = logmel.to(train_parameters.device, non_blocking=True)
            vae_output = vae(logmel)
            vae_loss = vae.loss(vae_output)

            with torch.no_grad():
                image = image.to(base_vae_device, non_blocking=True)
                base_vae_output = base_vae(image)
                base_mu = base_vae_output.mu.to(vae_output.mu.device, non_blocking=True)
                base_logvar = base_vae_output.logvar.to(
                    vae_output.logvar.device, non_blocking=True
                )

            latent_mean_mae_loss = torch.nn.functional.smooth_l1_loss(
                vae_output.mu, base_mu
            )
            latent_logvar_mae_loss = torch.nn.functional.smooth_l1_loss(
                vae_output.logvar, base_logvar
            )
            infonce_loss = calculate_infonce(
                vae_output.mu,
                vae_output.logvar,
                base_mu,
                base_logvar,
                logvar_weight=train_parameters.infonce_logvar_weight,
                temperature=train_parameters.infonce_temperature,
            )

            # contrastive_grad = torch.autograd.grad(
            #     infonce_loss,
            #     vae_output.mu,
            #     retain_graph=True,
            #     allow_unused=True,
            # )[0]

            # print("infonce loss:", infonce_loss.detach().cpu().item())

            # print(
            #     "contrastive gradient:",
            #     None if contrastive_grad is None else contrastive_grad.norm().item(),
            # )

            total_loss = (
                vae_loss.total_loss
                + train_parameters.latent_mean_mae_loss_weight * latent_mean_mae_loss
                + train_parameters.latent_logvar_mae_loss_weight
                * latent_logvar_mae_loss
                + train_parameters.contrastive_loss_weight * infonce_loss
            )

            # total_loss = infonce_loss
            total_loss.backward()

            # for name, parameter in vae.named_parameters():
            #     if parameter.grad is not None:
            #         print(
            #             name,
            #             "parameter norm:",
            #             parameter.norm().item(),
            #             "gradient norm:",
            #             parameter.grad.norm().item(),
            #         )
            #         break

            optimizer.step()
            running_total_loss += total_loss.detach().cpu().item()
            running_reconstruction_loss += (
                vae_loss.reconstruction_loss.detach().cpu().item()
            )
            running_kl_loss += vae_loss.kl_loss.detach().cpu().item()
            running_latent_mean_mae_loss += latent_mean_mae_loss.detach().cpu().item()
            running_latent_logvar_mae_loss += (
                latent_logvar_mae_loss.detach().cpu().item()
            )
            running_contrastive_loss += infonce_loss.detach().cpu().item()
            batches += 1

        train_losses.append(running_total_loss / batches)
        train_reconstruction_losses.append(running_reconstruction_loss / batches)
        train_kl_losses.append(running_kl_loss / batches)
        train_latent_mean_mae_losses.append(running_latent_mean_mae_loss / batches)
        train_latent_logvar_mae_losses.append(running_latent_logvar_mae_loss / batches)
        train_contrastive_losses.append(running_contrastive_loss / batches)

        # Val
        vae.eval()
        with torch.no_grad():
            running_total_loss = 0
            running_reconstruction_loss = 0
            running_kl_loss = 0
            running_latent_mean_mae_loss = 0
            running_latent_logvar_mae_loss = 0
            running_contrastive_loss = 0
            batches = 0
            for logmel, image, species in tqdm(val_dl):
                logmel = logmel.to(train_parameters.device, non_blocking=True)
                vae_output = vae(logmel)
                vae_loss = vae.loss(vae_output)

                image = image.to(base_vae_device, non_blocking=True)
                base_vae_output = base_vae(image)
                base_mu = base_vae_output.mu.to(vae_output.mu.device, non_blocking=True)
                base_logvar = base_vae_output.logvar.to(
                    vae_output.logvar.device, non_blocking=True
                )

                latent_mean_mae_loss = torch.nn.functional.smooth_l1_loss(
                    vae_output.mu, base_mu
                )
                latent_logvar_mae_loss = torch.nn.functional.smooth_l1_loss(
                    vae_output.logvar, base_logvar
                )
                infonce_loss = calculate_infonce(
                    vae_output.mu,
                    vae_output.logvar,
                    base_mu,
                    base_logvar,
                    logvar_weight=train_parameters.infonce_logvar_weight,
                    temperature=train_parameters.infonce_temperature,
                )

                total_loss = (
                    vae_loss.total_loss
                    + train_parameters.latent_mean_mae_loss_weight
                    * latent_mean_mae_loss
                    + train_parameters.latent_logvar_mae_loss_weight
                    * latent_logvar_mae_loss
                    + train_parameters.contrastive_loss_weight * infonce_loss
                )

                running_total_loss += total_loss.detach().cpu().item()
                running_reconstruction_loss += (
                    vae_loss.reconstruction_loss.detach().cpu().item()
                )
                running_kl_loss += vae_loss.kl_loss.detach().cpu().item()
                running_latent_mean_mae_loss += (
                    latent_mean_mae_loss.detach().cpu().item()
                )
                running_latent_logvar_mae_loss += (
                    latent_logvar_mae_loss.detach().cpu().item()
                )
                running_contrastive_loss += infonce_loss.detach().cpu().item()
                batches += 1

        val_losses.append(running_total_loss / batches)
        val_reconstruction_losses.append(running_reconstruction_loss / batches)
        val_kl_losses.append(running_kl_loss / batches)
        val_latent_mean_mae_losses.append(running_latent_mean_mae_loss / batches)
        val_latent_logvar_mae_losses.append(running_latent_logvar_mae_loss / batches)
        val_contrastive_losses.append(running_contrastive_loss / batches)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} train results: "
            f"total loss {train_losses[-1]}, "
            f"reconstruction loss {train_reconstruction_losses[-1]}, "
            f"kl loss {train_kl_losses[-1]}, "
            f"latent mean mae loss {train_latent_mean_mae_losses[-1]}, "
            f"latent logvar mae loss {train_latent_logvar_mae_losses[-1]}, "
            f"contrastive loss {train_contrastive_losses[-1]}, "
            f"lr {scheduler.get_last_lr()[0]}"
        )
        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} val results: "
            f"total loss {val_losses[-1]}, "
            f"reconstruction loss {val_reconstruction_losses[-1]}, "
            f"kl loss {val_kl_losses[-1]}, "
            f"latent mean mae loss {val_latent_mean_mae_losses[-1]}, "
            f"latent logvar mae loss {val_latent_logvar_mae_losses[-1]}, "
            f"contrastive loss {val_contrastive_losses[-1]}"
        )
        if (epoch + 1) % train_parameters.model_save_frequency == 0:
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
                        "latent_mean_mae_loss": train_latent_mean_mae_losses,
                        "latent_logvar_mae_loss": train_latent_logvar_mae_losses,
                        "contrastive_loss": train_contrastive_losses,
                    },
                    "val_losses": {
                        "total_loss": val_losses,
                        "reconstruction_loss": val_reconstruction_losses,
                        "kl_loss": val_kl_losses,
                        "latent_mean_mae_loss": val_latent_mean_mae_losses,
                        "latent_logvar_mae_loss": val_latent_logvar_mae_losses,
                        "contrastive_loss": val_contrastive_losses,
                    },
                    "train_generator": train_generator.get_state(),
                },
                os.path.join(
                    train_parameters.model_save_path,
                    f"{train_parameters.model_save_name}_epoch_{epoch}.tar",
                ),
            )

    if test_dataset is not None:
        # Test
        vae.eval()
        with torch.no_grad():
            running_total_loss = 0
            running_reconstruction_loss = 0
            running_kl_loss = 0
            running_latent_mean_mae_loss = 0
            running_latent_logvar_mae_loss = 0
            running_contrastive_loss = 0
            batches = 0
            for logmel, image, species in tqdm(test_dl):
                logmel = logmel.to(train_parameters.device, non_blocking=True)
                vae_output = vae(logmel)
                vae_loss = vae.loss(vae_output)

                image = image.to(base_vae_device, non_blocking=True)
                base_vae_output = base_vae(image)
                base_mu = base_vae_output.mu.to(vae_output.mu.device, non_blocking=True)
                base_logvar = base_vae_output.logvar.to(
                    vae_output.logvar.device, non_blocking=True
                )

                latent_mean_mae_loss = torch.nn.functional.smooth_l1_loss(
                    vae_output.mu, base_mu
                )
                latent_logvar_mae_loss = torch.nn.functional.smooth_l1_loss(
                    vae_output.logvar, base_logvar
                )
                infonce_loss = calculate_infonce(
                    vae_output.mu,
                    vae_output.logvar,
                    base_mu,
                    base_logvar,
                    logvar_weight=train_parameters.infonce_logvar_weight,
                    temperature=train_parameters.infonce_temperature,
                )

                total_loss = (
                    vae_loss.total_loss
                    + train_parameters.latent_mean_mae_loss_weight
                    * latent_mean_mae_loss
                    + train_parameters.latent_logvar_mae_loss_weight
                    * latent_logvar_mae_loss
                    + train_parameters.contrastive_loss_weight * infonce_loss
                )

                running_total_loss += total_loss.detach().cpu().item()
                running_reconstruction_loss += (
                    vae_loss.reconstruction_loss.detach().cpu().item()
                )
                running_kl_loss += vae_loss.kl_loss.detach().cpu().item()
                running_latent_mean_mae_loss += (
                    latent_mean_mae_loss.detach().cpu().item()
                )
                running_latent_logvar_mae_loss += (
                    latent_logvar_mae_loss.detach().cpu().item()
                )
                running_contrastive_loss += infonce_loss.detach().cpu().item()
                batches += 1

        test_loss = running_total_loss / batches
        test_reconstruction_loss = running_reconstruction_loss / batches
        test_kl_loss = running_kl_loss / batches
        test_latent_mean_mae_loss = running_latent_mean_mae_loss / batches
        test_latent_logvar_mae_loss = running_latent_logvar_mae_loss / batches
        test_contrastive_loss = running_contrastive_loss / batches

        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} test results: "
            f"total loss {test_loss}, "
            f"reconstruction loss {test_reconstruction_loss}, "
            f"kl loss {test_kl_loss}, "
            f"latent mean mae loss {test_latent_mean_mae_loss}, "
            f"latent logvar mae loss {test_latent_logvar_mae_loss}, "
            f"contrastive loss {test_contrastive_loss}"
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
                "latent_mean_mae_loss": train_latent_mean_mae_losses,
                "latent_logvar_mae_loss": train_latent_logvar_mae_losses,
                "contrastive_loss": train_contrastive_losses,
            },
            "val_losses": {
                "total_loss": val_losses,
                "reconstruction_loss": val_reconstruction_losses,
                "kl_loss": val_kl_losses,
                "latent_mean_mae_loss": val_latent_mean_mae_losses,
                "latent_logvar_mae_loss": val_latent_logvar_mae_losses,
                "contrastive_loss": val_contrastive_losses,
            },
            "train_generator": train_generator.get_state(),
        },
        os.path.join(
            train_parameters.model_save_path,
            f"{train_parameters.model_save_name}_final.tar",
        ),
    )

    return ContrastiveVAETrainOutputs(
        vae=vae,
        train_total_losses=train_losses,
        train_reconstruction_losses=train_reconstruction_losses,
        train_kl_losses=train_kl_losses,
        train_latent_mean_mae_losses=train_latent_mean_mae_losses,
        train_latent_logvar_mae_losses=train_latent_logvar_mae_losses,
        train_contrastive_losses=train_contrastive_losses,
        val_total_losses=val_losses,
        val_reconstruction_losses=val_reconstruction_losses,
        val_kl_losses=val_kl_losses,
        val_latent_mean_mae_losses=val_latent_mean_mae_losses,
        val_latent_logvar_mae_losses=val_latent_logvar_mae_losses,
        val_contrastive_losses=val_contrastive_losses,
        test_total_loss=test_loss if test_dataset is not None else None,
        test_reconstruction_loss=(
            test_reconstruction_loss if test_dataset is not None else None
        ),
        test_kl_loss=test_kl_loss if test_dataset is not None else None,
        test_latent_mean_mae_loss=(
            test_latent_mean_mae_loss if test_dataset is not None else None
        ),
        test_latent_logvar_mae_loss=(
            test_latent_logvar_mae_loss if test_dataset is not None else None
        ),
        test_contrastive_loss=(
            test_contrastive_loss if test_dataset is not None else None
        ),
    )
