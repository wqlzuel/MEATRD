import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False, separable=False):
        super().__init__()

        if separable:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, 
                          stride, padding, groups=in_channels, bias=bias),
                nn.Conv2d(in_channels, out_channels, 1, bias=bias)
            )
        else:
            self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                  stride, padding, bias=bias)

    def forward(self, x):
        return self.Conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, Mobile=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2d(in_channels, mid_channels, separable=Mobile),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, separable=Mobile),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, Mobile=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DoubleConv(in_channels, out_channels, Mobile=Mobile)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, Mobile=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels, Mobile=Mobile)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=1)
            self.conv = DoubleConv(in_channels, out_channels, Mobile=Mobile)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, Mobile=False):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 
                           kernel_size=1, padding=0, separable=Mobile)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=[32, 64, 128, 256, 512], 
                 bilinear=True, Mobile=True):
        super().__init__()

        factor = 2 if bilinear else 1
        self.emb_chan = (out_channels[-1] // 2)

        self.inc = DoubleConv(in_channels, out_channels[0], Mobile=Mobile)

        self.encoder = nn.ModuleList()
        chan_1 = out_channels[0]
        for num in range(1, len(out_channels)):
            f = 1 if num != len(out_channels)-1 else factor
            chan_2 = out_channels[num] // f
            self.encoder.append(Down(chan_1, chan_2, Mobile=Mobile))
            chan_1 = chan_2
        
        self.decoder = nn.ModuleList()
        out_channels = out_channels[::-1]
        chan_1 = out_channels[0]
        for num in range(1, len(out_channels)):
            f = factor if num != len(out_channels)-1 else 1
            chan_2 = out_channels[num] // f
            self.decoder.append(Up(chan_1, chan_2, bilinear, Mobile=Mobile))
            chan_1 = out_channels[num]
        
        self.outc = OutConv(out_channels[-1], in_channels, Mobile=Mobile)
    
    def encode(self, graph, x):
        node_ID = graph.num_dst_nodes()
        x = self.inc(x)
        x_skips = [x[:node_ID]]
        for down in self.encoder:
            x = down(x)
            x_skips.append(x[:node_ID])
        return x, x_skips
    
    def decode(self, graph, z, x_skips):
        node_ID = graph.num_dst_nodes()
        x = z[:node_ID]
        for up, x_down in zip(self.decoder, reversed(x_skips[:-1])):
            x = up(x, x_down)

        x = self.outc(x)
        return x
    
    def forward(self, graph, x):
        z, x_skips = self.encode(graph, x)
        x = self.decode(graph, z, x_skips)
        return x