import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from model.MedNext.MedNext_model import MedNeXt_encoder


class ImageProjection(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SegmentationHead(nn.Module):
    """
    Lightweight segmentation head for Chan-Vese soft-label distillation.
    Uses progressive 3D transposed convolutions to upsample from the
    bottleneck (feature_size*16 channels, 1/32 spatial) back to full resolution.
    Self-contained: no dependency on the UNETR decoder chain.
    """
    def __init__(self, feature_size):
        super().__init__()
        c = feature_size * 16  # bottleneck channels (e.g. 384 for feature_size=24)
        self.ups = nn.Sequential(
            # 5 x stride-2 upsample steps to recover 32x spatial downsampling
            nn.ConvTranspose3d(c,      c // 2,  kernel_size=2, stride=2),
            nn.InstanceNorm3d(c // 2), nn.GELU(),
            nn.ConvTranspose3d(c // 2, c // 4,  kernel_size=2, stride=2),
            nn.InstanceNorm3d(c // 4), nn.GELU(),
            nn.ConvTranspose3d(c // 4, c // 8,  kernel_size=2, stride=2),
            nn.InstanceNorm3d(c // 8), nn.GELU(),
            nn.ConvTranspose3d(c // 8, c // 16, kernel_size=2, stride=2),
            nn.InstanceNorm3d(c // 16), nn.GELU(),
            nn.ConvTranspose3d(c // 16, 1,      kernel_size=2, stride=2),
        )

    def forward(self, encoder_features):
        # encoder_features[0] is the bottleneck tensor
        return self.ups(encoder_features[0])


class MetadataHead(nn.Module):
    """Predict age (regression) and sex (binary classification) from image features."""
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )
        self.age_head = nn.Linear(256, 1)
        self.sex_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.age_head(h), self.sex_head(h)


class AutoARDS_Pretrain(nn.Module):
    def __init__(self,
                 in_channels=1,
                 feature_size=24,
                 exp_r=(2, 4, 8, 16, 16),
                 block_counts=(2, 4, 8, 16, 32),
                 proj_dim=512,
                 img_size=(224, 320, 224),
                 logit_scale_init=0.07):
        super().__init__()

        self.image_encoder = MedNeXt_encoder(
            in_channels=in_channels,
            n_channels=feature_size,
            exp_r=list(exp_r),
            kernel_size=3,
            do_res=True,
            do_res_up_down=True,
            block_counts=list(block_counts),
            dim="3d",
            grn=False,
        )

        bottleneck_dim = feature_size * 16
        self.image_proj = ImageProjection(bottleneck_dim, proj_dim)

        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = TextProjection(768, proj_dim)

        self.seg_head = SegmentationHead(feature_size)

        self.meta_head = MetadataHead(bottleneck_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def encode_image(self, x):
        features = self.image_encoder(x)
        emb = self.image_proj(features[0])
        return features, emb

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] token
        return self.text_proj(cls)

    def forward(self, image, input_ids, attention_mask):
        features, img_emb = self.encode_image(image)
        txt_emb = self.encode_text(input_ids, attention_mask)

        seg_logit = self.seg_head(features)

        bottleneck_flat = F.adaptive_avg_pool3d(features[0], 1).flatten(1)
        age_pred, sex_pred = self.meta_head(bottleneck_flat)

        return img_emb, txt_emb, seg_logit, age_pred, sex_pred, self.logit_scale.exp()


def clip_infonce_loss(img_emb, txt_emb, scale):
    """Symmetric InfoNCE (CLIP-style) contrastive loss."""
    logits = scale * img_emb @ txt_emb.T  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def pretrain_loss(img_emb, txt_emb, scale,
                  seg_logit, soft_label,
                  age_pred, age_gt,
                  sex_pred, sex_gt,
                  w_clip=1.0, w_seg=1.0, w_meta=0.5):
    """
    Combined pretraining loss:
      - CLIP InfoNCE (image-text alignment)
      - Segmentation BCE with soft labels (distillation from Chan-Vese)
      - Metadata prediction: age MSE + sex BCE
    """
    loss_clip = clip_infonce_loss(img_emb, txt_emb, scale)

    loss_seg = F.binary_cross_entropy_with_logits(
        seg_logit.squeeze(1), soft_label.float()
    )

    age_gt = age_gt.float().unsqueeze(1)
    loss_age = F.mse_loss(age_pred, age_gt)

    sex_gt = sex_gt.float().unsqueeze(1)
    loss_sex = F.binary_cross_entropy_with_logits(sex_pred, sex_gt)

    loss_meta = loss_age + loss_sex

    total = w_clip * loss_clip + w_seg * loss_seg + w_meta * loss_meta
    return total, loss_clip, loss_seg, loss_meta
