import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv

from .fusion import GraphMBT, ConcatFusion
from .unet import UNet


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm: bool = True,
                 act: bool = True, dropout: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.InstanceNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.8) if dropout else nn.Identity(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class GeneEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256]):
        super().__init__()
        self.linear = nn.Sequential(
            LinearBlock(in_dim, out_dim[0]),
            LinearBlock(out_dim[0], out_dim[1], False, False, False)
        )

    def forward(self, feat):
        z = self.linear(feat)
        return z


class GeneDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256], nheads=4):
        super().__init__()
        self.GAT = GATv2Conv(out_dim[-1], in_dim, num_heads=nheads,
                             feat_drop=0.2, attn_drop=0.1)
        self.fc = nn.Sequential(
            nn.Linear(nheads*in_dim, in_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, graph, z):
        feat = self.GAT(graph, z).flatten(1)
        return self.fc(feat)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ImageEncoder(nn.Module):
    def __init__(self, patch_size, n_ResidualBlock=8, n_levels=2,
                 input_channels=3, z_dim=256, MultiResSkips=True):
        super().__init__()
        
        self.max_filters = 2**(n_levels+3)
        self.n_levels = n_levels
        self.MultiResSkips = MultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2**(i + 3)
            n_filters_2 = 2**(i + 4)
            ks = 2**(n_levels - i)

            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                              for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                nn.Sequential(
                    nn.Conv2d(n_filters_1, n_filters_2,
                              kernel_size=(2, 2), stride=(2, 2), padding=0),
                    nn.InstanceNorm2d(n_filters_2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if MultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                  kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        nn.InstanceNorm2d(self.max_filters),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.z_dim = z_dim
        self.img_latent_dim = patch_size // (2**n_levels)
        self.feat_dim = self.z_dim*self.img_latent_dim**2
        self.fc = nn.Linear(self.feat_dim, self.z_dim)

    def forward(self, feat):
        feat = self.input_conv(feat)
        skips = []

        for i in range(self.n_levels):
            feat = self.res_blk_list[i](feat)
            if self.MultiResSkips:
                skips.append(self.multi_res_skip_list[i](feat))
            feat = self.conv_list[i](feat)

        if self.MultiResSkips:
            feat = sum([feat] + skips)

        z = feat = self.output_conv(feat)
        z = self.fc(feat.flatten(1))
        return z


class ImageDecoder(nn.Module):
    def __init__(self, patch_size, n_ResidualBlock=8, n_levels=2,
                 z_dim=256, output_channels=3, MultiResSkips=True):
        super().__init__()
        self.max_filters = 2**(n_levels+3)
        self.n_levels = n_levels
        self.MultiResSkips = MultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.InstanceNorm2d(self.max_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2**(self.n_levels - i + 3)
            n_filters_1 = 2**(self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                              for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                       kernel_size=(2, 2), stride=(2, 2), padding=0),
                    nn.InstanceNorm2d(n_filters_1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if MultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                           kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        nn.InstanceNorm2d(n_filters_1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.z_dim = z_dim
        self.img_latent_dim = patch_size // (2**n_levels)
        self.feat_dim = self.z_dim*self.img_latent_dim**2
        self.fc = nn.Linear(z_dim, self.feat_dim)

    def forward(self, z):
        z = self.fc(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim)
        z = z_top = self.input_conv(z)
        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.MultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        x = self.output_conv(z)
        return x




class Generator(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim=[512, 256], Mobile=False, **kwargs):
        super().__init__()
        self.GeneEncoder = GeneEncoder(in_dim, out_dim)
        self.GeneDecoder = GeneDecoder(in_dim, out_dim)

        self.UNet = UNet(3, Mobile=Mobile, **kwargs)

        emb_chan = self.UNet.emb_chan
        self.Fusion = GraphMBT(out_dim[-1], emb_chan, patch_size // (2**4))

        self.z_g_dim = out_dim[-1]
        self.z_p_dim = emb_chan * patch_size**2

    def pretrain_unet(self, graph, feat_p):
        feat_p = self.UNet(graph, feat_p)
        return feat_p

    def forward(self, blocks, feat_g, feat_p):
        real_g = feat_g[:blocks[-1].num_dst_nodes()]
        real_p = feat_p[:blocks[-1].num_dst_nodes()]

        z_g = self.GeneEncoder(feat_g)
        z_p, p_skips = self.UNet.encode(blocks[-1], feat_p)

        # Fusion with GraphMBT
        z_g, z_p = self.Fusion(blocks[:-1], z_g, z_p)

        fake_g = self.GeneDecoder(blocks[-1], z_g)
        fake_p = self.UNet.decode(blocks[-1], z_p, p_skips)
        
        return real_g, real_p, fake_g, fake_p


class SVDDEncoder(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim=[512, 256]):
        super().__init__()
        self.gene_enc = nn.Linear(in_dim, out_dim[-1])
        self.image_enc = ImageEncoder(patch_size, z_dim=out_dim[-1])
        self.ConcatFusion = ConcatFusion(out_dim[-1], out_dim[-1])
    
    def forward(self, gene, patch):
        eg = self.gene_enc(gene)
        ep = self.image_enc(patch)
        ef = self.ConcatFusion(eg, 0.1*ep)
        return ef.abs()




