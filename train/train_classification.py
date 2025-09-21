import argparse
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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import *
from tqdm import tqdm
from model.model import *
from train.utils import *

# Pin specific CPU cores for performance
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use CUDA?")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Default=1e-4")
parser.add_argument("--step", type=int, default=10, help="Step size for LR scheduler")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use")
parser.add_argument("--gpus", default="0", type=str, help="GPU IDs (default: 0)")
parser.add_argument("--gamma", type=float, default=0.8, help="Learning rate decay factor")
parser.add_argument("--checkpoint_dir", type=str, default="/data/Train_test/ARDS/checkpoint/ARDS/base")


def train():
    # Parse command-line arguments
    opt = parser.parse_args()
    cuda = opt.cuda and torch.cuda.is_available()
    print(f"=> Use GPU: {opt.gpus}, CUDA Available: {cuda}")

    # Seed for reproducibility
    opt.seed = random.randint(1, 10000)
    print("Random Seed:", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.benchmark = True

    model = MedNeXt_regression_info(in_channels=2,
                                    out_channels=3,
                                    info_dim=10,
                                    fea=24,
                                    exp_r=[2, 4, 8, 16, 16],
                                    kernel_size=3,
                                    block_counts=[2, 4, 8, 16, 32]).cuda()
    model.encoder.load_state_dict(torch.load("./pretrained.pth"))
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Optimizer and scheduler
    print("===> Setting Optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    train_set = TrainSetLoader_classification(info_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                              train=True, folder=0)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True, persistent_workers=True)

    valid_set = TrainSetLoader_classification(info_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                              train=False, folder=0)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads,
                              batch_size=opt.batchSize, shuffle=False)
    scaler = GradScaler()

    # Training loop
    print("===> Starting Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print(f"Epoch {epoch}/{opt.nEpochs}")

        train_one_epoch(train_loader, optimizer, model, epoch, scaler)

        if epoch % 5 == 0:
            save_checkpoint(model, opt.checkpoint_dir, epoch)
            try:
                valid_epoch(valid_loader, model)
            except:
                continue

    scheduler.step()


def train_one_epoch(data_loader, optimizer, model, epoch, scaler):
    """Train for one epoch with mixed precision and log metrics."""
    print(f"Epoch {epoch}: Training...")
    model.train()

    total_loss = 0

    f_loss = nn.CrossEntropyLoss()

    for iteration, (data, info, target) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Move data to GPU if available
        data = data.to(device, non_blocking=True)
        info = info.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # view = data.cpu().detach().numpy()
        # plot_parallel(
        #     a=view[0, 0, :, :, 112],
        #     b=view[0, 1, :, :, 112],
        #     v_high=1, v_low=0
        # )

        with autocast():
            p_class = model(data, info)
            loss = f_loss(p_class, target)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (iteration + 1) % 100 == 0:
            avg_loss = total_loss / (iteration + 1)
            print(f"===> Epoch[{epoch}] Iteration[{iteration + 1}] Average Loss: {avg_loss:.5f}")

    avg_loss = total_loss / (iteration + 1)
    print(f"===> Epoch[{epoch}]  Average Loss: {avg_loss:.5f}")


def valid_epoch(data_loader, model):
    """Validate the model and compute classification metrics."""
    model.eval()  # Use eval mode for validation

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():  # No gradient computation during validation
        for iteration, (data, info, target) in enumerate(tqdm(data_loader, desc="Validation")):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            info = info.to(device, non_blocking=True)

            p_class = model(data, info)

            _, predicted = torch.max(p_class, dim=1)
            probabilities = torch.softmax(p_class, dim=1)

            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        # print(y_true.shape, y_pred.shape, y_probs.shape)
        for i in range(y_true.shape[0]):
            print(y_true[i], round(y_probs[i, 0], 3), round(y_probs[i, 1], 3), round(y_probs[i, 2], 3))

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        sen = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spe = tn / (tn + fp)

        print("Acc: {:.3f}, F1 Score: {:.3f}, AUC: {:.3f}, Sensitivity: {:.3f}, Specificity: {:.3f}".
              format(accuracy, f1, auc, sen, spe))


def save_checkpoint(model, path, epoch):
    """Save the model checkpoint."""
    os.makedirs(path, exist_ok=True)
    model_out_path = os.path.join(path, f"{epoch:03d}_classification_2.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


if __name__ == '__main__':
    from torch.multiprocessing import set_start_method

    set_start_method("spawn", force=True)

    train()
