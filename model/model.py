import torch
import torch.nn as nn
from einops import rearrange
from monai.networks.nets.swin_unetr import SwinTransformer
from model.MedNext.MedNext_model import MedNeXt_encoder
from model.MedNext.mednext.MedNextV2 import *
import torch
from model.MedNext.MedNext_model import MedM3AE, MedNeXtBlock
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool


class Transformer_regression(nn.Module):
    def __init__(self,
                 feature=24,  # must be a mutiple of 12
                 in_channels=3,
                 out_channels=3,
                 patch=4):
        super(Transformer_regression, self).__init__()

        self.encoder = SwinTransformer(in_chans=in_channels,
                                       embed_dim=feature,
                                       window_size=(7, 7, 7),
                                       patch_size=(patch, patch, patch),
                                       depths=(2, 2, 2, 2),
                                       num_heads=(3, 6, 12, 24),
                                       )

        self.header = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=(2, 2, 1)),
            nn.Flatten(),
            nn.Linear(in_features=feature * 16 * 4, out_features=256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=out_channels, bias=True),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)[-1]
        return self.header(x)


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: str,
            norm: str,
            bias: bool,
            dropout: float = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: str,
            norm: str,
            bias: bool,
            dropout: float = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UNet_regression(nn.Module):
    def __init__(self, spatial_dims: int = 3,
                 in_channels: int = 1,
                 fea=(32, 64, 128, 256, 512),
                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm=("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: float = 0.0):
        super(UNet_regression, self).__init__()
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=fea[4], out_features=64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True)
        )

    def forward(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return self.head(x4)


class MedNeXt_regression(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 fea=32,
                 exp_r=(2, 3, 4, 4, 4),
                 block_counts=(3, 4, 4, 4, 4),
                 kernel_size=3,
                 num_embeddings=3):
        super(MedNeXt_regression, self).__init__()
        self.encoder = MedNeXt_encoder(in_channels=in_channels,
                                       n_channels=fea,
                                       exp_r=exp_r,
                                       kernel_size=kernel_size,  # Ofcourse can test kernel_size
                                       do_res=True,  # Can be used to individually test residual connection
                                       do_res_up_down=True,
                                       block_counts=block_counts,
                                       dim="3d",  # 2d or 3d
                                       grn=False)

        self.decoder = MedNeXt_linear(n_channels=fea,
                                      n_classes=out_channels,
                                      exp_r=[2, 2, 2, 2, 2],
                                      kernel_size=3,  # Ofcourse can test kernel_size
                                      do_res_up_down=True,  # Additional 'res' connection on up and down convs
                                      dim="3d",  # 2d or 3d
                                      grn=False,
                                      num_embeddings=num_embeddings)

        # self.decoder = MedNeXt_linear_2(n_channels=fea,
        #                                 n_classes=out_channels,
        #                                 exp_r=[2, 2, 2, 2, 2],
        #                                 kernel_size=3,  # Ofcourse can test kernel_size
        #                                 do_res_up_down=True,  # Additional 'res' connection on up and down convs
        #                                 dim="3d",  # 2d or 3d
        #                                 grn=False)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class M3AE_regression(nn.Module):
    def __init__(self, out_channels: int = 1):
        super(M3AE_regression, self).__init__()
        model = MedM3AE(in_channels=2,
                        n_channels=24,
                        n_classes=1,
                        exp_r=[2, 4, 8, 16, 16],
                        kernel_size=3,
                        deep_supervision=False,
                        do_res=True,
                        do_skip=(False, False, False, True, True),
                        do_res_up_down=True,
                        block_counts=[2, 4, 8, 16, 32])

        model.load_state_dict(torch.load("/data/Altolia/checkpoint/MedNext/010_MedNext_M3AE_24.pth"))
        self.encoder = model.image_encoder

        self.decoder = nn.Sequential(
            MedNeXtBlock(
                in_channels=16 * 24,
                out_channels=4 * 24,
                exp_r=2,
                kernel_size=3,
                do_res=False,
                dim="3d",  # 2d or 3d
                grn=False
            ),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )

        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool3d(()),
            nn.Flatten(),
            nn.Linear(in_features=4 * 24 * 8, out_features=512, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=out_channels, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.decoder(x)
        x = self.header(x)

        return x


class MedNeXt_regression_info(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 fea=32,
                 exp_r=(2, 3, 4, 4, 4),
                 block_counts=(3, 4, 4, 4, 4),
                 kernel_size=3,
                 embed_num=3,
                 info_dim=6):
        super(MedNeXt_regression_info, self).__init__()
        self.encoder = MedNeXt_encoder(in_channels=in_channels,
                                       n_channels=fea,
                                       exp_r=exp_r,
                                       kernel_size=kernel_size,  # Ofcourse can test kernel_size
                                       do_res=True,  # Can be used to individually test residual connection
                                       do_res_up_down=True,
                                       block_counts=block_counts,
                                       dim="3d",  # 2d or 3d
                                       grn=False)

        self.decoder = MedNeXt_linear_info(n_channels=fea,
                                           n_classes=out_channels,
                                           exp_r=[2, 2, 2, 2, 2],
                                           kernel_size=3,  # Ofcourse can test kernel_size
                                           do_res_up_down=True,  # Additional 'res' connection on up and down convs
                                           dim="3d",  # 2d or 3d
                                           info_dim=info_dim,
                                           num_embeddings=embed_num,
                                           grn=False)

    def forward(self, x, info):
        x = self.encoder(x)
        # print(x.shape)
        return self.decoder(x, info)


class MedNeXt_regression_contrast(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 fea=32,
                 exp_r=(2, 3, 4, 4, 4),
                 block_counts=(3, 4, 4, 4, 4),
                 kernel_size=3,
                 num_embeddings=3,
                 info_dim=6):
        super(MedNeXt_regression_contrast, self).__init__()
        self.encoder = MedNeXt_encoder(in_channels=in_channels,
                                       n_channels=fea,
                                       exp_r=exp_r,
                                       kernel_size=kernel_size,  # Ofcourse can test kernel_size
                                       do_res=True,  # Can be used to individually test residual connection
                                       do_res_up_down=True,
                                       block_counts=block_counts,
                                       dim="3d",  # 2d or 3d
                                       grn=False)

        self.decoder = MedNeXt_linear_contrast(n_channels=fea * 2,
                                               n_classes=out_channels,
                                               exp_r=[2, 2, 2, 2, 2],
                                               kernel_size=3,  # Ofcourse can test kernel_size
                                               do_res_up_down=True,  # Additional 'res' connection on up and down convs
                                               dim="3d",  # 2d or 3d
                                               grn=False,
                                               num_embeddings=1,
                                               info_dim=info_dim)

    def forward(self, x1, x2, info, score):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x = [torch.concat([x1[i], x2[i]], dim=1) for i in range(len(x1))]
        return self.decoder(x, info, score)


class MedNeXt_mixed(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 fea=24,
                 exp_r=(2, 3, 4, 4, 4),
                 block_counts=(3, 4, 4, 4, 4),
                 kernel_size=3,
                 info_dim=2,
                 num_embeddings=3):
        super(MedNeXt_mixed, self).__init__()
        self.encoder = MedNeXt_encoder(in_channels=in_channels,
                                       n_channels=fea,
                                       exp_r=exp_r,
                                       kernel_size=kernel_size,  # Ofcourse can test kernel_size
                                       do_res=True,  # Can be used to individually test residual connection
                                       do_res_up_down=True,
                                       block_counts=block_counts,
                                       dim="3d",  # 2d or 3d
                                       grn=False)

        self.linear_decoder = MedNeXt_linear_info(n_channels=fea,
                                                  n_classes=1,
                                                  exp_r=[2, 2, 2, 2, 2],
                                                  kernel_size=3,  # Ofcourse can test kernel_size
                                                  do_res_up_down=True,
                                                  info_dim=info_dim,
                                                  dim="3d",  # 2d or 3d
                                                  grn=False,
                                                  num_embeddings=num_embeddings)

        self.seg_decoder = MedNeXt_seg(out_channels=1,
                                       feature_size=fea)

    def forward(self, x, info):
        x = self.encoder(x)
        linear = self.linear_decoder(x, info)
        seg = self.seg_decoder(x)

        return linear, seg


class MedNeXt_mixed_info(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 fea=24,
                 exp_r=(2, 3, 4, 4, 4),
                 block_counts=(3, 4, 4, 4, 4),
                 kernel_size=3,
                 info_dim=1,
                 num_embeddings=3):
        super(MedNeXt_mixed_info, self).__init__()
        self.encoder = MedNeXt_encoder(in_channels=in_channels,
                                       n_channels=fea,
                                       exp_r=exp_r,
                                       kernel_size=kernel_size,  # Ofcourse can test kernel_size
                                       do_res=True,  # Can be used to individually test residual connection
                                       do_res_up_down=True,
                                       block_counts=block_counts,
                                       dim="3d",  # 2d or 3d
                                       grn=False)

        self.linear_decoder = MedNeXt_linear_info(n_channels=fea,
                                                  n_classes=1,
                                                  exp_r=[2, 2, 2, 2, 2],
                                                  kernel_size=3,  # Ofcourse can test kernel_size
                                                  do_res_up_down=True,
                                                  info_dim=info_dim,
                                                  # Additional 'res' connection on up and down convs
                                                  dim="3d",  # 2d or 3d
                                                  grn=False,
                                                  num_embeddings=num_embeddings)

        self.seg_decoder = MedNeXt_seg(out_channels=1,
                                       feature_size=fea)

    def forward(self, x, info):
        x = self.encoder(x)
        linear = self.linear_decoder(x, info)
        seg = self.seg_decoder(x)

        return linear, seg
