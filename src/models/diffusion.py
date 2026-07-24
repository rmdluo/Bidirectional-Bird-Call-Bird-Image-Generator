from dataclasses import dataclass
from math import pi
import torch
from models.common import ResidualBlock, SinusoidalEmbedding


@dataclass
class DiffusionParameters:
    input_shape: tuple = (4, 16, 16)
    conditioning_shape: tuple = (4, 16, 40)
    output_shape: tuple = (4, 16, 16)
    seed: int = 42


@dataclass
class DiffusionOutput:
    noise: torch.Tensor
    noisy_images: torch.Tensor
    noise_var: torch.Tensor
    predicted_noise: torch.Tensor


@dataclass
class DiffusionLoss:
    total_loss: torch.Tensor


class DownBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, block_depth=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                ResidualBlock(in_ch if i == 0 else out_ch, out_ch)
                for i in range(block_depth)
            ]
        )
        self.pool = torch.nn.AvgPool2d(2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)  # push and store skips
        return self.pool(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, block_depth=2):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.blocks = torch.nn.ModuleList(
            [ResidualBlock(in_ch + out_ch, out_ch)]
            + [ResidualBlock(out_ch + out_ch, out_ch) for _ in range(block_depth - 1)]
        )

    def forward(self, x, skips):
        x = self.up(x)
        for block in self.blocks:
            skip = skips.pop()  # retrieve skips
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class Diffusion(torch.nn.Module):
    def __init__(self, params: DiffusionParameters):
        super().__init__()
        self.input_shape = params.input_shape

        self.initial_conv = torch.nn.Conv2d(params.input_shape[0], 32, kernel_size=1)
        self.conditioning_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(
                params.conditioning_shape[0], 32, kernel_size=3, padding="same"
            ),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d((params.input_shape[1], params.input_shape[2])),
        )
        self.noise_embedding = SinusoidalEmbedding(embed_d=32)  # 32 size
        self.upsample = torch.nn.Upsample(
            size=(params.input_shape[1], params.input_shape[2]), mode="nearest"
        )

        self.down1 = DownBlock(
            32 + 32 + 32, 32
        )  # initial conv + noise_embedding embed_d
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 96)

        self.res1 = ResidualBlock(96, 128)
        self.res2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(128, 96)
        self.up2 = UpBlock(96, 64)
        self.up3 = UpBlock(64, 32)

        self.final_conv = torch.nn.Conv2d(32, params.output_shape[0], kernel_size=1)
        generator = torch.Generator()
        generator.manual_seed(params.seed)
        torch.nn.init.kaiming_normal_(
            self.final_conv.weight, nonlinearity="linear", generator=generator
        )

    def forward(self, noise, noisy_images, noise_var, conditioning_latent):
        x = self.initial_conv(noisy_images)
        noise_embed = self.noise_embedding(noise_var)
        noise_embed = self.upsample(noise_embed)
        cond_embed = self.conditioning_embedding(conditioning_latent)
        cond_embed = self.upsample(cond_embed)

        x = torch.cat([x, noise_embed, cond_embed], dim=1)  # textbook pg.217
        skips = []  # stack data structure

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.res1(x)
        x = self.res2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return DiffusionOutput(
            noise=noise,
            noisy_images=noisy_images,
            noise_var=noise_var,
            predicted_noise=self.final_conv(x),
        )

    @staticmethod
    def generate_noise(shape, generator=None):
        noise = torch.randn(shape, generator=generator)
        return noise

    @staticmethod
    def generate_random_timesteps(shape, generator=None):
        dif_times = torch.rand(shape, generator=generator)
        return dif_times

    @staticmethod
    def generate_noisy_images(latent, noise, signal_rates, noise_rates):
        return (
            signal_rates.view(latent.shape[0], 1, 1, 1) * latent
            + noise_rates.view(latent.shape[0], 1, 1, 1) * noise
        )

    @staticmethod
    def get_schedule(t):
        return torch.cos(t * pi / 2), torch.sin(t * pi / 2)  # alpha_t, sigma_t

    @staticmethod
    def sample(
        diffusion,
        conditioning_latent,
        n_samples=1,
        dif_steps=100,
        generator=None,
        device="cpu",
    ):
        with torch.no_grad():
            x = diffusion.generate_noise(
                (n_samples, *diffusion.input_shape), generator=generator
            ).to(device)
            xlist = [x]
            timesteps = torch.linspace(1.0, 0.0, dif_steps + 1, device=device)

            for i in range(dif_steps):
                t_now = timesteps[i]
                t_next = timesteps[i + 1]

                signal_now = torch.cos(t_now * pi / 2)
                noise_now = torch.sin(t_now * pi / 2)
                signal_next = torch.cos(t_next * pi / 2)
                noise_next = torch.sin(t_next * pi / 2)

                noise_cond = noise_now.expand(n_samples)

                if device == torch.device("cpu") or device == "cpu":
                    pred_noise = diffusion(None, x, noise_cond**2, conditioning_latent)
                else:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        pred_noise = diffusion(
                            None, x, noise_cond**2, conditioning_latent
                        ).predicted_noise

                # estimate clean image
                pred_x0 = (x - noise_now * pred_noise) / signal_now.clamp(min=1e-5)

                # CRUCIAL -- clamp predicted x0 to data range for stability
                pred_x0 = pred_x0.clamp(-3, 3)

                # DDIM step: re-noise to next level
                x = signal_next * pred_x0 + noise_next * pred_noise

                xlist += [x.cpu()]

            return xlist

    def loss(self, diffusion_output: DiffusionOutput):
        return DiffusionLoss(
            total_loss=torch.nn.functional.mse_loss(
                diffusion_output.predicted_noise, diffusion_output.noise
            )
        )
