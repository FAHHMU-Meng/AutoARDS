import torch
from transformers import BertTokenizer, BertForPreTraining, BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
from einops import rearrange
from model.MedNext.mednext.nnunet_mednext.network_architecture.mednextv1.blocks import *
from model.MedNext.unetr.lung.model_components import UnetrPPEncoder, UnetrUpBlock, UnetrUpBlock_S
from model.MedNext.unetr.dynunet_block import UnetOutBlock, UnetResBlock
from model.MedNext.losses.losses import ReconstructionLoss


# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# print(model)
# exit()
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.text_to_image_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.image_to_text_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_text = nn.LayerNorm(embed_dim)
        self.layer_norm_image = nn.LayerNorm(embed_dim)

    def forward(self, text_embeds, image_embeds):
        # Transpose for multihead attention (batch first)
        text_embeds = text_embeds.transpose(0, 1)
        image_embeds = image_embeds.transpose(0, 1)

        # Cross attention: text as query, image as key and value
        text_attention_output, _ = self.text_to_image_attention(text_embeds, image_embeds, image_embeds)
        fused_text_embeds = self.layer_norm_text(text_embeds + text_attention_output)

        # Cross attention: image as query, text as key and value
        image_attention_output, _ = self.image_to_text_attention(image_embeds, text_embeds, text_embeds)
        fused_image_embeds = self.layer_norm_image(image_embeds + image_attention_output)

        # Transpose back to original shape
        fused_text_embeds = fused_text_embeds.transpose(0, 1)
        fused_image_embeds = fused_image_embeds.transpose(0, 1)

        return fused_text_embeds, fused_image_embeds


class MedNeXt_encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 block_counts: list = (2, 2, 2, 2, 2),  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='2d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=2, stride=2)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[0])])

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[1])])

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[2])])

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[3])])

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])])

    def forward(self, x):
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)

        return [x, x_res_3, x_res_2, x_res_1, x_res_0]


class MedNeXt_decoder(nn.Module):
    def __init__(
            self,
            out_channels: int,
            feature_size: int = 16,
            pos_embed: str = "perceptron",
            norm_name="instance",
            dropout_rate: float = 0.0,
            img_size=(224, 320, 224),
            do_skip: list = (False, True, True, True),
            conv_op=nn.Conv3d,
            do_ds=False,
    ) -> None:
        super().__init__()
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
        self.do_skip = do_skip

        if do_skip[4]:
            self.decoder4 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 16,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[3],
                conv_decoder=True
            )
        else:
            self.decoder4 = UnetrUpBlock_S(
                spatial_dims=3,
                in_channels=feature_size * 16,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[3],
                conv_decoder=True
            )

        if do_skip[3]:
            self.decoder3 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[2],
                conv_decoder=True
            )
        else:
            self.decoder3 = UnetrUpBlock_S(
                spatial_dims=3,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[2],
                conv_decoder=True
            )

        if do_skip[2]:
            self.decoder2 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[1],
                conv_decoder=True
            )
        else:
            self.decoder2 = UnetrUpBlock_S(
                spatial_dims=3,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[1],
                conv_decoder=True
            )

        if do_skip[1]:
            self.decoder1 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[0],
                conv_decoder=True,
            )
        else:
            self.decoder1 = UnetrUpBlock_S(
                spatial_dims=3,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                out_size=self.hier_size[0],
                conv_decoder=True,
            )

        self.decoder0 = UnetrUpBlock_S(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.hier_size[0],
            conv_decoder=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

    def forward(self, hidden_states, inference=False):
        # Four encoders
        if not inference:
            enc4 = hidden_states[0].float()
            enc3 = hidden_states[1].float()
            enc2 = hidden_states[2].float()
            enc1 = hidden_states[3].float()
            enc0 = hidden_states[4].float()
        else:
            enc4 = hidden_states[0].half()
            enc3 = hidden_states[1].half()
            enc2 = hidden_states[2].half()
            enc1 = hidden_states[3].half()
            enc0 = hidden_states[4].half()

        dec4 = enc4
        if self.do_skip[4]:
            dec3 = self.decoder4(dec4, enc3)
        else:
            dec3 = self.decoder4(dec4)

        if self.do_skip[3]:
            dec2 = self.decoder3(dec3, enc2)
        else:
            dec2 = self.decoder3(dec3)

        if self.do_skip[2]:
            dec1 = self.decoder2(dec2, enc1)
        else:
            dec1 = self.decoder2(dec2)

        if self.do_skip[1]:
            dec0 = self.decoder1(dec1, enc0)
        else:
            dec0 = self.decoder1(dec1)

        # out = dec0
        out = self.decoder0(dec0)

        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits


class MedM3AE(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 n_channels: int = 32,
                 n_classes: int = 1,
                 exp_r: list = (2, 2, 2, 2, 2),  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 3,  # Ofcourse can test kernel_size
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 do_skip: list = (False, False, True, True, True),
                 # Whether do skip connection between encoder and decoder
                 block_counts: list = (2, 2, 2, 2, 2),  # Can be used to test staging ratio:
                 norm_type='group',
                 dim_latent=512,
                 dim_text=512,
                 dim_image=512,
                 vision_checkpoint=None,
                 logit_scale_init_value=0.07,
                 ) -> None:
        super().__init__()
        self.image_encoder = MedNeXt_encoder(in_channels=in_channels,
                                             n_channels=n_channels,
                                             exp_r=exp_r,
                                             kernel_size=kernel_size,  # Ofcourse can test kernel_size
                                             do_res=do_res,  # Can be used to individually test residual connection
                                             do_res_up_down=do_res_up_down,
                                             block_counts=block_counts,  # Can be used to test staging ratio:
                                             norm_type=norm_type,
                                             dim="3d",  # 2d or 3d
                                             grn=False)

        # self.vision_pooler = nn.Conv3d(in_channels=n_channels * 16, out_channels=n_channels * 8,
        #                                kernel_size=3, stride=1)

        self.image_decoder = MedNeXt_decoder(out_channels=n_classes,
                                             feature_size=n_channels,
                                             img_size=(224, 320, 224),
                                             do_skip=do_skip)

        text_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.text_encoder = text_model.bert
        self.text_decoder = text_model.cls

        # self.text_upsample = nn.Linear(256, 512, bias = False)
        # self.text_downsample = nn.Linear(512, 256, bias = False)

        self.fusion = CrossAttentionFusion(embed_dim=n_channels * 16, num_heads=8)

    def forward(self,
                input_ids, text_label,
                masked_image, gt_image,
                return_loss=None,
                ):

        vision_embeds = self.image_encoder(masked_image)

        vision_feature = vision_embeds[0]
        # text_feature = self.text_encoder(input_ids).last_hidden_state
        # print(text_feature.shape)

        vision_feature = rearrange(vision_feature, 'b c w d h -> b (w d h) c')
        # text_feature = rearrange(text_feature, 'b c f -> b f c')
        # text_feature = self.text_upsample(text_feature)

        # vision_feature, text_feature = self.fusion(vision_feature, text_feature)
        # text_feature = self.text_downsample(text_feature)

        vision_feature = rearrange(vision_feature, 'b (w d h) c -> b c w d h', w=7, d=10, h=7)
        # text_feature = rearrange(text_feature, 'b f c -> b c f')

        vision_embeds[0] = vision_feature

        re_img = self.image_decoder(vision_embeds)
        # re_text = self.text_decoder(text_feature)
        # print(re_text.shape)

        if return_loss:
            image_loss = ReconstructionLoss(re_img, gt_image, dim="3d", lamb=0.5)
            # clip_loss = self.mlm(re_text, text_label)
            # print(image_loss, clip_loss)
            # loss = clip_loss * 0.01 + image_loss
            loss = image_loss

        else:
            loss = None

        # return {"re_img": re_img, "re_text": re_text, 'loss_value': loss}
        return {"re_img": re_img, 'loss_value': loss}

    def mlm(self, logits, labels):
        shift_logits = logits.view(-1, logits.size(-1)).float()
        shift_labels = labels.view(-1)

        # Compute the loss manually
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        return loss


class Bert(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        text_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.text_encoder = text_model.bert
        self.text_decoder = text_model.cls

    def forward(self,
                input_ids, text_label,
                return_loss=None,
                ):

        text_feature = self.text_encoder(input_ids).last_hidden_state
        re_text = self.text_decoder(text_feature)
        # print(re_text.shape)

        if return_loss:
            # image_loss = ReconstructionLoss(re_img, gt_image, dim="3d", lamb=0.5)
            clip_loss = self.mlm(re_text, text_label)
            # print(image_loss, clip_loss)
            loss = clip_loss

        else:
            loss = None

        return {"re_text": re_text, 'loss_value': loss}

    def mlm(self, logits, labels):
        shift_logits = logits.view(-1, logits.size(-1)).float()
        shift_labels = labels.view(-1)

        # Compute the loss manually
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        return loss


class MedNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 norm_type: str = 'group',
                 n_groups: int or None = None,
                 dim: str = '3d',
                 grn: bool = False,
                 dropout_rate: float = 0.0  # New parameter for dropout rate
                 ):

        super().__init__()

        self.do_res = do_res
        self.dropout_rate = dropout_rate  # Save the dropout rate
        assert dim in ['2d', '3d']
        self.dim = dim

        if self.dim == '2d':
            conv = nn.Conv2d
            dropout = nn.Dropout2d  # Use 2D dropout for 2D inputs
        elif self.dim == '3d':
            conv = nn.Conv3d
            dropout = nn.Dropout3d  # Use 3D dropout for 3D inputs

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(
                normalized_shape=in_channels,
                elementwise_affine=True
            )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Dropout layer (after activation of expanded features)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GRN (Gated Residual Networks) parameters
        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):
        x1 = x

        # First convolution layer
        x1 = self.conv1(x1)

        # Normalization + Second convolution + Activation
        x1 = self.norm(x1)
        x1 = self.act(self.conv2(x1))

        # Optional GRN
        if self.grn:
            # GRN normalization
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1

        # Apply dropout after activation
        x1 = self.dropout(x1)

        # Third convolution layer
        x1 = self.conv3(x1)

        # Residual connection (if enabled)
        if self.do_res:
            x1 = x + x1

        return x1
