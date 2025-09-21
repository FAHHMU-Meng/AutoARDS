import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("taskset -cp 0-7 %d" % os.getpid())
import random
import math
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import torch.optim as optim
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import *
from tqdm import tqdm
from model.model import *
from train.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="ARDS 28-day Survival Prediction")
parser.add_argument("--batchSize", type=int, default=4)
parser.add_argument("--nEpochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--start-epoch", type=int, default=1)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--gamma", type=float, default=0.8)
parser.add_argument("--checkpoint_dir", type=str, default="/data/Train_test/ARDS/checkpoint/ARDS/")


def train():
    opt = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # cudnn.benchmark = True

    model = MedNeXt_regression_info(in_channels=1,
                                    out_channels=29,
                                    fea=24,
                                    exp_r=[2, 4, 8, 16, 16],
                                    kernel_size=3,
                                    info_dim=10,
                                    block_counts=[2, 4, 8, 16, 32]).to(device)
    model.encoder.load_state_dict(torch.load("./pretrained.pth"))
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    train_set = TrainSetLoader_survival(info_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                        lesion_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                        train=True)
    valid_set = TrainSetLoader_survival(info_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                        lesion_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                        train=False)

    train_loader = DataLoader(train_set, batch_size=opt.batchSize, num_workers=opt.threads, shuffle=True,
                              persistent_workers=True)
    valid_loader = DataLoader(valid_set, batch_size=opt.batchSize, num_workers=opt.threads, shuffle=False)

    # validate(model, valid_loader)
    # exit()

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print(f"\n===> Epoch {epoch}/{opt.nEpochs}")
        train_one_epoch(model, train_loader, optimizer, epoch)

        if epoch % 5 == 0:
            save_checkpoint(model, opt.checkpoint_dir, epoch)
            validate(model, valid_loader)

        scheduler.step()


def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc=f"Training Epoch {epoch}"):
        image, death_status, mask, info = batch  # durations: int day; events: 0 or 1
        image = image.to(device)
        death_status = death_status.to(device).float()  # shape: (B, 28)
        mask = mask.to(device).bool()
        info = info.to(device)

        optimizer.zero_grad()

        # Predict death logits
        pred_logits = model(image, info)  # shape: (B, 28), raw logits
        # pred_prob = torch.sigmoid(pred_logits)

        # Focal loss (only on masked positions)
        loss = sigmoid_focal_loss(
            inputs=pred_logits[mask],  # predicted logits at masked positions
            targets=death_status[mask],  # ground truth (0 or 1)
            alpha=0.6, gamma=2.0, reduction="mean"  # tune alpha/gamma as needed
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"===> Epoch {epoch} Average Loss: {avg_loss:.5f}")


def validate(model, loader):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            image, death_status, mask, info = batch  # durations: int day; events: 0 or 1
            image = image.to(device)
            death_status = death_status.to(device).float()  # shape: (B, 28)
            mask = mask.to(device).bool()
            info = info.to(device)

            target = death_status[mask]

            # Predict death logits
            pred_logits = model(image, info)[mask]  # shape: (B, 28), raw logits
            pred_prob = torch.sigmoid(pred_logits)  # shape: (N,)
            predicted = (pred_prob >= 0.5).long()  # shape: (N,)

            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(pred_prob.cpu().numpy())

        y_true = all_labels
        y_pred = all_preds
        y_probs = all_probs

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        sen = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spe = tn / (tn + fp)

        print("Acc: {:.3f}, F1 Score: {:.3f}, AUC: {:.3f}, Sensitivity: {:.3f}, Specificity: {:.3f}".
              format(accuracy, f1, auc, sen, spe))


def save_checkpoint(model, path, epoch):
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"{epoch:03d}_survival_4.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == '__main__':
    train()
