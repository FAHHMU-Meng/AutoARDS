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
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20, help="Step size for LR scheduler")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use")
parser.add_argument("--gamma", type=float, default=0.9, help="Learning rate decay factor")
parser.add_argument("--checkpoint_dir", type=str, default="/data/Train_test/ARDS/checkpoint/ARDS/base")


def calculate_r_l2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_max = np.max(y_true)
    y_min = np.min(y_true)

    denominator = y_max - y_min
    if denominator == 0:
        raise ValueError("y_max and y_min are equal, denominator cannot be zero.")

    relative_l2 = np.power((y_true - y_pred) / denominator, 2)
    r_l2 = np.mean(relative_l2)
    return r_l2


def train():
    # Parse command-line arguments
    opt = parser.parse_args()
    cuda = torch.cuda.is_available()
    # print(f"=> Use GPU: {opt.gpus}, CUDA Available: {cuda}")

    # Seed for reproducibility
    opt.seed = random.randint(1, 10000)
    print("Random Seed:", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.benchmark = True

    model = MedNeXt_regression_info(in_channels=1,
                                    out_channels=1,
                                    fea=24,
                                    exp_r=[2, 4, 8, 16, 16],
                                    kernel_size=3,
                                    info_dim=10,
                                    block_counts=[2, 4, 8, 16, 32]).cuda()
    model.encoder.load_state_dict(torch.load("./pretrained.pth"))
    for param in model.encoder.parameters():
        param.requires_grad = False

    print("===> Setting Optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    train_set = TrainSetLoader_OI(info_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                  lesion_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                  train=True, folder=6)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True, persistent_workers=True)

    valid_set = TrainSetLoader_OI(info_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                  lesion_path="/data/Train_test/ARDS/ARDS_train_3.xlsx",
                                  train=False, folder=0)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads,
                              batch_size=opt.batchSize, shuffle=False)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print("===> Starting Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print(f"Epoch {epoch}/{opt.nEpochs}")

        train_one_epoch(train_loader, optimizer, model, epoch, scaler, cuda)

        if epoch % 5 == 0:
            save_checkpoint(model, opt.checkpoint_dir, epoch)
            valid_epoch(valid_loader, model, channel=1)

    scheduler.step()


def train_one_epoch(data_loader, optimizer, model, epoch, scaler, cuda):
    """Train for one epoch with mixed precision and log metrics."""
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: Training... Current Learning Rate = {current_lr:.5f}")
    model.train()

    # Loss functions
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    total_loss = 0
    all_predictions = []
    all_ground_truths = []

    for iteration, (data, gt, info) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Move data to GPU if available
        data = data.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        info = info.to(device, non_blocking=True)

        # view = data.cpu().detach().numpy()
        # plot_parallel(
        #     a=view[0, 0, :, :, 112],
        #     b=view[1, 0, :, :, 112],
        #     v_high=1, v_low=0
        # )

        with autocast():
            # optimizer.zero_grad()
            predictions = model(data, info)
            loss = (l1_loss(predictions, gt) +
                    torch.sqrt(l2_loss(predictions, gt)))
        # print(weight, predictions, gt, info)

        # Backward pass and optimizer step with mixed precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        total_loss += loss.item()

        # Collect predictions and ground truths for metrics
        all_predictions.extend(predictions.detach().cpu().numpy().flatten())
        all_ground_truths.extend(gt.detach().cpu().numpy().flatten())

        # Log every 100 iterations
        if (iteration + 1) % 100 == 0:
            avg_loss = total_loss / (iteration + 1)
            print(f"===> Epoch[{epoch}] Iteration[{iteration + 1}] Average Loss: {avg_loss:.5f}")

    # End of epoch logging
    avg_epoch_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} Completed. Average Loss: {avg_epoch_loss:.5f}")

    # Convert predictions and ground truths to numpy arrays for metric computation
    all_predictions = np.array(all_predictions)
    all_ground_truths = np.array(all_ground_truths)

    print(all_predictions.shape, all_ground_truths.shape)

    # Calculate Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(all_predictions, all_ground_truths)
    spearman_corr, _ = spearmanr(all_predictions, all_ground_truths)

    # Calculate L1 loss (mean absolute error)
    l1_metric = np.mean(np.abs(all_predictions - all_ground_truths))

    # Log metrics
    print(f"Epoch {epoch} Metrics:")
    print(f"  Pearson Correlation Coefficient: {pearson_corr:.4f}")
    print(f"  Spearman Correlation Coefficient: {spearman_corr:.4f}")
    print(f"  Average L1 Loss: {l1_metric:.5f}")


def valid_epoch(data_loader, model, channel):
    model.eval()  # Use eval mode for validation
    model.to(torch.float32)

    results = [[] for _ in range(channel)]  # Separate lists for each channel
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    loss_epoch = 0

    for iteration, (data, gt, info) in tqdm(enumerate(data_loader), total=len(data_loader)):
        data = data.to(device, non_blocking=True).to(torch.float32)
        gt = gt.to(device, non_blocking=True).to(torch.float32)
        info = info.to(device, non_blocking=True).to(torch.float32)

        with autocast():
            predictions = model(data, info)
            loss = (l1_loss(predictions, gt) +
                    torch.sqrt(l2_loss(predictions, gt)))

        # Log results for this batch per channel
        for i in range(data.shape[0]):
            for ch in range(channel):  # Assuming 4 channels in the output
                results[ch].append([predictions[i, ch].item(), gt[i, ch].item()])
        loss_epoch += loss.item()

    # Convert results to numpy arrays for metric computation
    results = [np.array(channel_results) for channel_results in results]

    # coefficient = [1, ]
    # Compute and log metrics for each channel
    for ch in range(channel):
        print(f"Channel {ch + 1}:")
        channel_results = results[ch]
        correlation_s, _ = spearmanr(channel_results[:, 0], channel_results[:, 1])
        correlation_p, _ = pearsonr(channel_results[:, 0], channel_results[:, 1])
        r_l2 = calculate_r_l2(channel_results[:, 0], channel_results[:, 1])
        l1 = np.mean(np.abs((channel_results[:, 0] - channel_results[:, 1])))

        print(f"  SCorrelation: {correlation_s:.4f}, PCorrelation: {correlation_p:.4f}")
        print(f"  Relative L2 Loss: {r_l2:.4f}, L1 Loss: {l1:.4f}")

    loss_epoch = loss_epoch / len(data_loader)
    print(f"Average loss: {loss_epoch:.4f}")


def save_checkpoint(model, path, epoch):
    """Save the model checkpoint."""
    os.makedirs(path, exist_ok=True)
    model_out_path = os.path.join(path, f"{epoch:03d}_OI_2.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


if __name__ == '__main__':
    from torch.multiprocessing import set_start_method

    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    train()
