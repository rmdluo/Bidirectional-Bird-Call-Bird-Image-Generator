import torch


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        group_norm_num_groups=8,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="same"
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, padding="same"
        )
        self.act1 = torch.nn.SiLU()
        self.act2 = torch.nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(group_norm_num_groups, in_channels)
        self.norm2 = torch.nn.GroupNorm(group_norm_num_groups, out_channels)
        self.residual = (
            torch.nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual
