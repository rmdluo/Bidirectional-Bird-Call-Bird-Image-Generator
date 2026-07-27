from dataclasses import dataclass
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler


@dataclass
class DiffusionParameters:
    input_shape: tuple = (4, 16, 16)
    conditioning_shape: tuple = (4, 16, 40)
    output_shape: tuple = (4, 16, 16)
    conditioning_dim: int = 256
    num_train_steps: int = 1000
    unet_block_out_channels: tuple[int] = (320, 640, 1280)
    seed: int = 42


@dataclass
class DiffusionOutput:
    noise: torch.Tensor
    noisy_images: torch.Tensor
    predicted_noise: torch.Tensor


@dataclass
class DiffusionLoss:
    total_loss: torch.Tensor


class Diffusion(torch.nn.Module):
    def __init__(self, params: DiffusionParameters):
        super().__init__()
        self.input_shape = params.input_shape
        self.train_steps = params.num_train_steps
        self.generator = torch.Generator()
        self.generator = self.generator.manual_seed(params.seed)

        self.embed_conditioning = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=3, padding=1),
            torch.nn.GroupNorm(8, 64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.GroupNorm(16, 128),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, params.conditioning_dim, kernel_size=3, padding=1),
        )
        self.conditioning_positional_embedding = torch.nn.Parameter(
            torch.randn(
                1,
                (params.conditioning_shape[1] // 2)
                * (params.conditioning_shape[2] // 2),
                params.conditioning_dim,
            )
            * 0.02
        )

        self.diffusion = UNet2DConditionModel(
            sample_size=(params.input_shape[1], params.input_shape[2]),
            in_channels=params.input_shape[0],
            out_channels=params.output_shape[0],
            cross_attention_dim=params.conditioning_dim,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            block_out_channels=params.unet_block_out_channels,
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        )

        self.scheduler = DDIMScheduler(
            num_train_timesteps=params.num_train_steps, clip_sample=False
        )

    def forward(self, noise, noisy_images, timesteps, conditioning_latents):
        conditioning = self.embed_conditioning(conditioning_latents)
        conditioning = torch.flatten(conditioning, start_dim=2)
        conditioning = torch.transpose(conditioning, 1, 2)
        conditioning = conditioning + self.conditioning_positional_embedding

        diffusion_output = self.diffusion(noisy_images, timesteps, conditioning)

        return DiffusionOutput(
            noise=noise,
            noisy_images=noisy_images,
            predicted_noise=diffusion_output.sample,
        )

    def generate_noise(self, batch_size, generator=None):
        noise = torch.randn((batch_size, *self.input_shape), generator=generator)
        return noise

    def generate_random_timesteps(self, batch_size, generator=None):
        timesteps = torch.randint(
            0,
            self.train_steps,
            (batch_size,),
            generator=generator,
        )
        return timesteps

    def generate_noisy_images(self, latent, noise, timesteps):
        return self.scheduler.add_noise(latent, noise, timesteps)

    def sample(
        self,
        conditioning_latent,
        init_latents=None,
        start_timestep=None,
        n_samples=1,
        dif_steps=100,
        generator=None,
        device="cpu",
    ):
        with torch.no_grad():
            self.scheduler.set_timesteps(dif_steps)
            if init_latents is None:
                x = self.generate_noise(n_samples, generator=generator).to(device)
            else:
                x = init_latents.to(device)
            xlist = [x]

            timesteps = self.scheduler.timesteps
            if start_timestep is not None:
                timesteps = timesteps[timesteps <= start_timestep]

            for t in timesteps:
                if device == torch.device("cpu") or device == "cpu":
                    pred_noise = self(None, x, t, conditioning_latent).predicted_noise
                else:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        pred_noise = self(
                            None, x, t, conditioning_latent
                        ).predicted_noise

                x = self.scheduler.step(
                    pred_noise, t, x, generator=generator
                ).prev_sample

                xlist += [x.cpu()]

            return xlist

    def loss(self, diffusion_output: DiffusionOutput):
        return DiffusionLoss(
            total_loss=torch.nn.functional.mse_loss(
                diffusion_output.predicted_noise, diffusion_output.noise
            )
        )
