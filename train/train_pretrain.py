import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("taskset -cp 0-7 %d" % os.getpid())

import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.pretrain_model import AutoARDS_Pretrain, pretrain_loss
from train.utils import TrainSetLoader_pretrain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="AutoARDS Pretraining")
parser.add_argument("--batchSize",      type=int,   default=4)
parser.add_argument("--nEpochs",        type=int,   default=100)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--step",           type=int,   default=20)
parser.add_argument("--gamma",          type=float, default=0.9)
parser.add_argument("--start-epoch",    type=int,   default=1)
parser.add_argument("--threads",        type=int,   default=8)
parser.add_argument("--folder",         type=int,   default=0,
                    help="Validation fold index")
parser.add_argument("--w_clip",         type=float, default=1.0)
parser.add_argument("--w_seg",          type=float, default=1.0)
parser.add_argument("--w_meta",         type=float, default=0.5)
parser.add_argument("--info_path",      type=str,
                    default="/data/Train_test/ARDS/ARDS_pretrain.xlsx")
parser.add_argument("--checkpoint_dir", type=str,
                    default="/data/Train_test/ARDS/checkpoint/ARDS/pretrain")


def train():
    opt = parser.parse_args()

    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = AutoARDS_Pretrain(
        in_channels=1,
        feature_size=24,
        exp_r=(2, 4, 8, 16, 16),
        block_counts=(2, 4, 8, 16, 32),
        proj_dim=512,
        img_size=(224, 320, 224),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.step, gamma=opt.gamma
    )
    scaler = GradScaler()

    train_set = TrainSetLoader_pretrain(
        info_path=opt.info_path,
        train=True,
        folder=opt.folder,
    )
    valid_set = TrainSetLoader_pretrain(
        info_path=opt.info_path,
        train=False,
        folder=opt.folder,
    )

    train_loader = DataLoader(
        train_set, batch_size=opt.batchSize,
        num_workers=opt.threads, shuffle=True, persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=opt.batchSize,
        num_workers=opt.threads, shuffle=False,
    )

    print("===> Starting AutoARDS Pretraining")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print(f"\n===> Epoch {epoch}/{opt.nEpochs}")
        train_one_epoch(model, train_loader, optimizer, scaler, epoch, opt)

        if epoch % 5 == 0:
            save_checkpoint(model, opt.checkpoint_dir, epoch)
            validate(model, valid_loader)

        scheduler.step()


def train_one_epoch(model, loader, optimizer, scaler, epoch, opt):
    model.train()
    total_loss = total_clip = total_seg = total_meta = 0.0
    lr = optimizer.param_groups[0]['lr']
    print(f"  lr={lr:.6f}")

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        image, input_ids, attn_mask, soft_label, age_gt, sex_gt = batch
        image      = image.to(device, non_blocking=True)
        input_ids  = input_ids.to(device, non_blocking=True)
        attn_mask  = attn_mask.to(device, non_blocking=True)
        soft_label = soft_label.to(device, non_blocking=True)
        age_gt     = age_gt.to(device, non_blocking=True)
        sex_gt     = sex_gt.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            img_emb, txt_emb, seg_logit, age_pred, sex_pred, scale = model(
                image, input_ids, attn_mask
            )
            loss, l_clip, l_seg, l_meta = pretrain_loss(
                img_emb, txt_emb, scale,
                seg_logit, soft_label,
                age_pred, age_gt,
                sex_pred, sex_gt,
                w_clip=opt.w_clip,
                w_seg=opt.w_seg,
                w_meta=opt.w_meta,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_clip += l_clip.item()
        total_seg  += l_seg.item()
        total_meta += l_meta.item()

    n = len(loader)
    print(
        f"  Loss={total_loss/n:.4f}  "
        f"CLIP={total_clip/n:.4f}  "
        f"Seg={total_seg/n:.4f}  "
        f"Meta={total_meta/n:.4f}"
    )


def validate(model, loader):
    model.eval()
    total_loss = total_clip = total_seg = total_meta = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            image, input_ids, attn_mask, soft_label, age_gt, sex_gt = batch
            image      = image.to(device)
            input_ids  = input_ids.to(device)
            attn_mask  = attn_mask.to(device)
            soft_label = soft_label.to(device)
            age_gt     = age_gt.to(device)
            sex_gt     = sex_gt.to(device)

            img_emb, txt_emb, seg_logit, age_pred, sex_pred, scale = model(
                image, input_ids, attn_mask
            )
            loss, l_clip, l_seg, l_meta = pretrain_loss(
                img_emb, txt_emb, scale,
                seg_logit, soft_label,
                age_pred, age_gt,
                sex_pred, sex_gt,
            )
            total_loss += loss.item()
            total_clip += l_clip.item()
            total_seg  += l_seg.item()
            total_meta += l_meta.item()

    n = len(loader)
    print(
        f"[Val] Loss={total_loss/n:.4f}  "
        f"CLIP={total_clip/n:.4f}  "
        f"Seg={total_seg/n:.4f}  "
        f"Meta={total_meta/n:.4f}"
    )


def save_checkpoint(model, path, epoch):
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, f"{epoch:03d}_pretrain.pth")
    # Save only the image encoder for downstream fine-tuning
    torch.save(model.image_encoder.state_dict(), out)
    print(f"  Checkpoint saved: {out}")


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    train()
