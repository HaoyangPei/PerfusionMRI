import torch
from torch import nn
from torch.nn import functional as F

class ST_Generator(nn.Module):
    """
    PyTorch implementation of a SpatioTemporal U-Net model.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 8,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(self.out_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, 0.5)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        self.outconv = nn.Conv2d(self.in_chans, 1, kernel_size=1, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        stack = []
        output = image
        
        output = output[:,None,:,:,:]

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1  # padding front
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        output = output.squeeze(1)
        maps = self.tanh(self.outconv(output))

        return maps

class ConvBlock(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.spatial_layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=(1,3,3), padding=(0,1,1), bias=True),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=(1,3,3), padding=(0,1,1), bias=True),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
        )

        self.temporal_layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=(3,1,1), padding=(1,0,0), bias=True),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3,1,1), padding=(1,0,0), bias=True),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        return self.temporal_layers(image)+self.spatial_layers(image)


class TransposeConvBlock(nn.Module):

    def __init__(self, in_chans: int, out_chans: int):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.spatial_layers = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='nearest'),
            nn.Conv3d(in_chans, out_chans, kernel_size=(1,3,3), padding=(0,1,1), bias=True),
        )

        self.temporal_layers = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1), mode='nearest'),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3,1,1), padding=(1,0,0), bias=True),
        )

        self.out = nn.Sequential(
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        return self.out(self.temporal_layers(self.spatial_layers(image)))
