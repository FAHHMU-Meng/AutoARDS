import torch
import torch.nn as nn
from CTCLIP.ctvit import CTViT, CTViTEncoder
from einops import rearrange


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1 + self.conv2(x1)
        return x2


class CT_model(nn.Module):
    def __init__(self,
                 image_size=(480, 480, 240),
                 patch_size=2,
                 num_class=2,
                 pretrain_checkpoint=None):
        super().__init__()
        image_encoder = CTViTEncoder(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=30,
            temporal_patch_size=15,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )

        if pretrain_checkpoint is not None:
            checkpoint = torch.load(pretrain_checkpoint)

            filtered_state_dict = {
                k: v for k, v in checkpoint.items() if k in image_encoder.state_dict()
            }
            image_encoder.load_state_dict(filtered_state_dict, strict=False)

        self.patch = patch_size
        self.encoder = image_encoder

        self.decoder_1 = conv_block(16 * 512, 8 * 512)
        self.decoder_2 = conv_block(8 * 512, 4 * 512)
        self.decoder_3 = nn.Sequential(
            conv_block(4 * 512, 512),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2)),
        )

        self.header = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512 * 8 * 8, out_features=512, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=num_class, bias=True),
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # print(x.shape)
        for i in range(16):
            if i == 0:
                embedding = self.encoder(x[:, :, :15])
            else:
                sub_embedding = self.encoder(x[:, :, i * 15:(i + 1) * 15])
                embedding = torch.concat([embedding, sub_embedding], dim=1)

        de = rearrange(embedding, 'b f h w c -> b (f c) h w')
        de = self.decoder_1(de)
        de = self.decoder_2(de)
        de = self.decoder_3(de)
        return self.header(de)
