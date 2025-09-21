from torch import nn
import torch
from typing import Tuple, Union
from basic_model.unetr.neural_network import SegmentationNetwork
from basic_model.unetr.dynunet_block import UnetOutBlock, UnetResBlock
from basic_model.unetr.lung.model_components import UnetrPPEncoder, UnetrUpBlock, UnetrUpBlock_S

# class UNETR_PP(SegmentationNetwork):
#     """
#     UNETR++ based on: "Shaker et al.,
#     UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             feature_size: int = 16,
#             hidden_size: int = 256,
#             num_heads:int = 4,
#             pos_embed: str = "perceptron",
#             norm_name: Union[Tuple, str] = "instance",
#             dropout_rate: float = 0.0,
#             depths=(3, 3, 9, 3),
#             dims=(32, 64, 128, 256),
#             img_size=(256, 256, 192),
#             conv_op=nn.Conv3d,
#             do_ds=False,
#     ) -> None:
#         """
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             feature_size: dimension of network feature size.
#             hidden_size: dimensions of  the last encoder.
#             num_heads: number of attention heads.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             dropout_rate: faction of the input units to drop.
#             depths: number of blocks for each stage.
#             dims: number of channel maps for the stages.
#             conv_op: type of convolution operation.
#             do_ds: use deep supervision to compute the loss.
#         """
#
#         super().__init__()
#         if depths is None:
#             depths = [3, 3, 3, 3]
#         self.do_ds = do_ds
#         self.conv_op = conv_op
#         self.num_classes = out_channels
#         if not (0 <= dropout_rate <= 1):
#             raise AssertionError("dropout_rate should be between 0 and 1.")
#
#         if pos_embed not in ["conv", "perceptron"]:
#             raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
#
#         self.feat_size = (img_size[0] // 32, img_size[1] // 32, img_size[2] // 32)
#         self.hidden_size = hidden_size
#         self.img_dimension = img_size[0] * img_size[1] * img_size[2]
#         self.hier_size = (self.img_dimension, self.img_dimension // (2 ** 6),
#                           self.img_dimension // (2 ** 9), self.img_dimension // (2 ** 12),
#                           self.img_dimension // (2 ** 15))
#
#         self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads,
#                                                input_size=self.hier_size[1:], in_channels=in_channels)
#
#         self.encoder1 = UnetResBlock(
#             spatial_dims=3,
#             in_channels=in_channels,
#             out_channels=feature_size,
#             kernel_size=3,
#             stride=1,
#             norm_name=norm_name,
#         )
#         self.decoder5 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 16,
#             out_channels=feature_size * 8,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=self.hier_size[3],
#         )
#         self.decoder4 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 8,
#             out_channels=feature_size * 4,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=self.hier_size[2],
#         )
#         self.decoder3 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 4,
#             out_channels=feature_size * 2,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             out_size=self.hier_size[1],
#         )
#         self.decoder2 = UnetrUpBlock(
#             spatial_dims=3,
#             in_channels=feature_size * 2,
#             out_channels=feature_size,
#             kernel_size=3,
#             upsample_kernel_size=4,
#             norm_name=norm_name,
#             out_size=self.hier_size[0],
#             conv_decoder=True,
#         )
#         self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
#         if self.do_ds:
#             self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
#             self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
#
#     def proj_feat(self, x, hidden_size, feat_size):
#         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         return x
#
#     def forward(self, x_in):
#         #print("#####input_shape:", x_in.shape)
#         x_output, hidden_states = self.unetr_pp_encoder(x_in)
#
#         convBlock = self.encoder1(x_in)
#
#         # Four encoders
#         enc1 = hidden_states[0]
#         #print("ENC1:",enc1.shape)
#         enc2 = hidden_states[1]
#         #print("ENC2:",enc2.shape)
#         enc3 = hidden_states[2]
#         #print("ENC3:",enc3.shape)
#         enc4 = hidden_states[3]
#         #print("ENC4:",enc4.shape)
#
#         # Four decoders
#         dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
#         dec3 = self.decoder5(dec4, enc3)
#         dec2 = self.decoder4(dec3, enc2)
#         dec1 = self.decoder3(dec2, enc1)
#
#         out = self.decoder2(dec1, convBlock)
#         if self.do_ds:
#             logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
#         else:
#             logits = self.out1(out)
#
#         return logits
#
#
# class UNETR_S(SegmentationNetwork):
#     """
#     UNETR++ based on: "Shaker et al.,
#     UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             num_heads:int = 4,
#             pos_embed: str = "perceptron",
#             norm_name: Union[Tuple, str] = "instance",
#             dropout_rate: float = 0.0,
#             depths=(3, 3, 9, 3),
#             dims=(32, 64, 128, 256),
#             img_size=(256, 256, 192),
#             conv_op=nn.Conv3d,
#     ) -> None:
#         """
#         Args:
#             in_channels: dimension of input channels.
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             feature_size: dimension of network feature size.
#             hidden_size: dimensions of  the last encoder.
#             num_heads: number of attention heads.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             dropout_rate: faction of the input units to drop.
#             depths: number of blocks for each stage.
#             dims: number of channel maps for the stages.
#             conv_op: type of convolution operation.
#             do_ds: use deep supervision to compute the loss.
#         """
#
#         super().__init__()
#         if depths is None:
#             depths = [3, 3, 3, 3]
#         self.conv_op = conv_op
#         self.num_classes = out_channels
#         if not (0 <= dropout_rate <= 1):
#             raise AssertionError("dropout_rate should be between 0 and 1.")
#
#         if pos_embed not in ["conv", "perceptron"]:
#             raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
#
#         self.feat_size = (img_size[0] // 32, img_size[1] // 32, img_size[2] // 32)
#         self.feature_size = dims[0] // 2
#         self.hidden_size = dims[-1]
#         self.img_dimension = img_size[0] * img_size[1] * img_size[2]
#         self.hier_size = (self.img_dimension, self.img_dimension // (2 ** 6),
#                           self.img_dimension // (2 ** 9), self.img_dimension // (2 ** 12),
#                           self.img_dimension // (2 ** 15))
#
#         self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads,
#                                                input_size=self.hier_size[1:], in_channels=in_channels)
#         # self.encoder1 = UnetResBlock(
#         #     spatial_dims=3,
#         #     in_channels=in_channels,
#         #     out_channels=feature_size,
#         #     kernel_size=3,
#         #     stride=1,
#         #     norm_name=norm_name,
#         # )
#         self.decoder5 = UnetrUpBlock_S(
#             spatial_dims=3,
#             in_channels=self.feature_size * 16,
#             out_channels=self.feature_size * 8,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             depth=1,
#             out_size=self.hier_size[3],
#         )
#         self.decoder4 = UnetrUpBlock_S(
#             spatial_dims=3,
#             in_channels=self.feature_size * 8,
#             out_channels=self.feature_size * 4,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             depth=1,
#             out_size=self.hier_size[2],
#         )
#         self.decoder3 = UnetrUpBlock_S(
#             spatial_dims=3,
#             in_channels=self.feature_size * 4,
#             out_channels=self.feature_size * 2,
#             kernel_size=3,
#             upsample_kernel_size=2,
#             norm_name=norm_name,
#             depth=1,
#             out_size=self.hier_size[1],
#         )
#         self.decoder2 = UnetrUpBlock_S(
#             spatial_dims=3,
#             in_channels=self.feature_size * 2,
#             out_channels=self.feature_size,
#             kernel_size=3,
#             upsample_kernel_size=4,
#             norm_name=norm_name,
#             depth=1,
#             out_size=self.hier_size[0],
#             conv_decoder=True,
#         )
#         self.out1 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=out_channels)
#
#     def proj_feat(self, x, hidden_size, feat_size):
#         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         return x
#
#     def forward(self, x_in):
#         #print("#####input_shape:", x_in.shape)
#         x_output, hidden_states = self.unetr_pp_encoder(x_in)
#
#         # convBlock = self.encoder1(x_in)
#
#         # Four encoders
#         # enc1 = hidden_states[0]
#         #print("ENC1:",enc1.shape)
#         # enc2 = hidden_states[1]
#         #print("ENC2:",enc2.shape)
#         # enc3 = hidden_states[2]
#         #print("ENC3:",enc3.shape)
#         enc4 = hidden_states[3]
#         #print("ENC4:",enc4.shape)
#
#         # Four decoders
#         dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
#         dec3 = self.decoder5(dec4)
#         dec2 = self.decoder4(dec3)
#         dec1 = self.decoder3(dec2)
#
#         out = self.decoder2(dec1)
#         logits = self.out1(out)
#
#         return logits
#
#
# class UNETR_linear(SegmentationNetwork):
#     """
#     UNETR++ based on: "Shaker et al.,
#     UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
#     """
#     def __init__(
#             self,
#             out_channels: int,
#             norm_name: Union[Tuple, str] = "instance",
#             dropout_rate: float = 0.0,
#             dims=(32, 64, 128, 256),
#             img_size=(256, 256, 192),
#             conv_op=nn.Conv3d,
#     ) -> None:
#         """
#         Args:
#             out_channels: dimension of output channels.
#             img_size: dimension of input image.
#             pos_embed: position embedding layer type.
#             norm_name: feature normalization type and arguments.
#             dropout_rate: faction of the input units to drop.
#             dims: number of channel maps for the stages.
#             conv_op: type of convolution operation.
#         """
#
#         super().__init__()
#         self.conv_op = conv_op
#         self.num_classes = out_channels
#         if not (0 <= dropout_rate <= 1):
#             raise AssertionError("dropout_rate should be between 0 and 1.")
#
#         self.feat_size = (img_size[0] // 32, img_size[1] // 32, img_size[2] // 32)
#         self.feature_size = dims[0] // 2
#         self.hidden_size = dims[-1]
#         self.img_dimension = img_size[0] * img_size[1] * img_size[2]
#         self.hier_size = (self.img_dimension, self.img_dimension // (2 ** 6),
#                           self.img_dimension // (2 ** 9), self.img_dimension // (2 ** 12),
#                           self.img_dimension // (2 ** 15))
#
#         self.decoder4 = nn.Sequential(
#             UnetResBlock(
#                     spatial_dims=3,
#                     in_channels=self.feature_size * 16,
#                     out_channels=self.feature_size * 8,
#                     kernel_size=3,
#                     stride=1,
#                     norm_name=norm_name,
#                 ),
#             UnetResBlock(
#                 spatial_dims=3,
#                 in_channels=self.feature_size * 8,
#                 out_channels=self.feature_size * 4,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#             ),
#         )
#
#         self.decoder3 = nn.Sequential(
#             UnetResBlock(
#                     spatial_dims=3,
#                     in_channels=self.feature_size * 8,
#                     out_channels=self.feature_size * 4,
#                     kernel_size=3,
#                     stride=1,
#                     norm_name=norm_name,
#
#                 ),
#             nn.AdaptiveAvgPool3d((self.feat_size[0], self.feat_size[1], self.feat_size[2]))
#         )
#         self.decoder2 = nn.Sequential(
#             UnetResBlock(
#                     spatial_dims=3,
#                     in_channels=self.feature_size * 4,
#                     out_channels=self.feature_size * 4,
#                     kernel_size=3,
#                     stride=1,
#                     norm_name=norm_name,
#                 ),
#             nn.AdaptiveAvgPool3d((self.feat_size[0], self.feat_size[1], self.feat_size[2]))
#         )
#
#         self.downsample = nn.Sequential(
#             UnetResBlock(
#                 spatial_dims=3,
#                 in_channels=self.feature_size * 4 * 3,
#                 out_channels=self.feature_size * 4,
#                 kernel_size=3,
#                 stride=1,
#                 norm_name=norm_name,
#             )
#         )
#
#         self.header = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=self.feature_size * 4 * self.hier_size[-1], out_features=256, bias=True),
#             nn.LeakyReLU(),
#             nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
#         )
#
#     def proj_feat(self, x, hidden_size, feat_size):
#         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         return x
#
#     def forward(self, hidden_states):
#         # Four encoders
#         # enc1 = hidden_states[0]
#         #print("ENC1:",enc1.shape)
#         enc2 = hidden_states[1].float()
#         #print("ENC2:",enc2.shape)
#         enc3 = hidden_states[2].float()
#         #print("ENC3:",enc3.shape)
#         enc4 = hidden_states[3].float()
#         #print("ENC4:",enc4.shape)
#
#         dec4 = self.decoder4(self.proj_feat(enc4, self.hidden_size, self.feat_size))
#         dec3 = self.decoder3(enc3)
#         dec2 = self.decoder2(enc2)
#         # dec1 = self.decoder1(enc1)
#
#         out = self.downsample(torch.concat([dec4, dec3, dec2], dim=1))
#         logits = self.header(out)
#
#         return logits

class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            num_heads:int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=(3, 3, 9, 3),
            img_size=(224, 320, 224),
            conv_op=nn.Conv3d,
            conv_decoder=True,
            decoder_depth=(1, 1, 1, 1),
            do_ds=False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (img_size[0] // 32, img_size[1] // 32, img_size[2] // 32)
        self.hidden_size = feature_size * 16
        self.img_dimension = img_size[0] * img_size[1] * img_size[2]
        self.hier_size = (self.img_dimension, self.img_dimension // (2 ** 6),
                          self.img_dimension // (2 ** 9), self.img_dimension // (2 ** 12),
                          self.img_dimension // (2 ** 15))
        dims = (feature_size * 2, feature_size * 4, feature_size * 8, feature_size * 16)
        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads,
                                               input_size=self.hier_size[1:], in_channels=in_channels)

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.hier_size[3],
            conv_decoder=conv_decoder,
            depth=decoder_depth[0],
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.hier_size[2],
            conv_decoder=conv_decoder,
            depth=decoder_depth[1],
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.hier_size[1],
            conv_decoder=conv_decoder,
            depth=decoder_depth[2],
        )
        self.decoder2 = UnetrUpBlock_S(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            out_size=self.hier_size[0],
            conv_decoder=conv_decoder,
            depth=decoder_depth[3],
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        #print("#####input_shape:", x_in.shape)
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        # convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        #print("ENC1:",enc1.shape)
        enc2 = hidden_states[1]
        #print("ENC2:",enc2.shape)
        enc3 = hidden_states[2]
        #print("ENC3:",enc3.shape)
        enc4 = hidden_states[3]
        #print("ENC4:",enc4.shape)

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits


