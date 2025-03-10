import os
import sys
from typing import List

import click
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import ImageFile
from torch import nn, optim
from torch.utils import data, tensorboard
from torch.utils.data.distributed import DistributedSampler  # Multi-GPU

from src import configs
from src import data as lc_data
from src import network
from src.l3c import timer


def setup_device():
    """ Thiết lập GPU, chỉ dùng DataParallel thay vì DistributedSampler """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    print(f"✅ Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    return device, num_gpus


def train_loop(
        x: torch.Tensor, compressor: nn.Module,
        optimizer: optim.Optimizer,
        train_iter: int, plotter: tensorboard.SummaryWriter,
        plot_iters: int, clip: float, batch_idx: int, save_probs_path: str
) -> None:
    """ Training loop cho 1 batch và lưu `probs` nếu cần """
    compressor.train()
    optimizer.zero_grad()
    inp_size = np.prod(x.size())

    # 🔥 Chắc chắn dữ liệu chạy trên GPU
    x = x.to(next(compressor.parameters()).device)  

    # Chạy model trên GPU
    bits = compressor(x)

    # Lưu probs cho từng ảnh trong batch
    for img_idx, (img, prob_data) in enumerate(bits.probs):
        img_filename = os.path.join(save_probs_path, f"train_iter_{train_iter}_batch_{batch_idx}_img_{img_idx}.pt")
        torch.save(prob_data, img_filename)

    # Tính loss và cập nhật weights
    total_loss = bits.get_total_bpsp(inp_size)
    total_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(compressor.parameters(), clip)
    optimizer.step()

    # Chỉ log mỗi `plot_iters` lần để tránh log quá nhiều
    if train_iter % plot_iters == 0:
        plotter.add_scalar("train/bpsp", total_loss.item(), train_iter)
        plotter.add_scalar("train/grad_norm", grad_norm, train_iter)


@click.command()
@click.option("--train-path", type=click.Path(exists=True),
              help="path to directory of training images.")
@click.option("--eval-path", type=click.Path(exists=True),
              help="path to directory of eval images.")
@click.option("--train-file", type=click.File("r"),
              help="file for training image names.")
@click.option("--eval-file", type=click.File("r"),
              help="file for eval image names.")
@click.option("--batch", type=int, help="Batch size for training.")
@click.option("--workers", type=int, default=2,
              help="Number of worker threads to use in dataloader.")  # 🚀 Tăng workers nếu cần
@click.option("--plot", type=str,
              help="path to store tensorboard run data/plots.")
@click.option("--epochs", type=int, default=50, show_default=True,
              help="Number of epochs to run.")
@click.option("--resblocks", type=int, default=5, show_default=True,
              help="Number of resblocks to use.")
@click.option("--n-feats", type=int, default=64, show_default=True,
              help="Size of feature vector/channel width.")
@click.option("--scale", type=int, default=3, show_default=True,
              help="Scale of downsampling")
@click.option("--load", type=click.Path(exists=True), default="/dev/null",
              help="Path to load model")
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option("--eval-iters", type=int, default=0,
              help="Number of train iterations per evaluation.")
@click.option("--lr-epochs", type=int, default=1,
              help="Number of epochs before multiplying learning rate by 0.75")
@click.option("--plot-iters", type=int, default=1000,
              help="Number of train iterations before plotting data")
@click.option("--K", type=int, default=10,
              help="Number of clusters in logistic mixture model.")
@click.option("--clip", type=float, default=0.5,
              help="Norm to clip by for gradient clipping.")
@click.option("--crop", type=int, default=128,
              help="Size of image crops in training.")
@click.option("--gd", type=click.Choice(["sgd", "adam", "rmsprop"]), default="adam",
              help="Type of gd to use.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print detailed logs if enabled.")

def main(
        train_path: str, eval_path: str, train_file, eval_file,
        batch: int, workers: int, plot: str, epochs: int,
        resblocks: int, n_feats: int, scale: int, load: str,
        lr: float, eval_iters: int, lr_epochs: int,
        plot_iters: int, k: int, clip: float,
        crop: int, gd: str, verbose: bool
) -> None:
    """ Main training function """
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Setup GPU
    device, num_gpus = setup_device()

    # Cấu hình model
    compressor = network.Compressor()
    
    # 🔥 Nếu có nhiều GPU, sử dụng DataParallel
    if num_gpus > 1:
        compressor = nn.parallel.DataParallel(compressor)

    compressor = compressor.to(device)  # Đưa model lên GPU

    # Setup optimizer
    optimizer: optim.Optimizer
    if gd == "adam":
        optimizer = optim.Adam(compressor.parameters(), lr=lr, weight_decay=0)
    elif gd == "sgd":
        optimizer = optim.SGD(compressor.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif gd == "rmsprop":
        optimizer = optim.RMSprop(compressor.parameters(), lr=lr)
    else:
        raise NotImplementedError(gd)

    train_dataset = lc_data.ImageFolder(
        train_path,
        [filename.strip() for filename in train_file],
        scale,
        T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(crop)])
    )

    train_loader = data.DataLoader(
    train_dataset, batch_size=batch, shuffle=True,  # ❌ Bỏ DistributedSampler
    num_workers=workers, drop_last=True, pin_memory=True
    )


    # Training loop
    train_iter = 0
    for epoch in range(epochs):
        if num_gpus > 1:
            train_sampler.set_epoch(epoch)  # 🔥 Reset sampler mỗi epoch nếu có nhiều GPU

        with tensorboard.SummaryWriter(plot) as plotter:
            for batch_idx, (_, inputs) in enumerate(train_loader):
                train_iter += 1
                train_loop(inputs, compressor, optimizer, train_iter, plotter, plot_iters, clip, batch_idx, plot)

    print("✅ Training complete")


if __name__ == "__main__":
    main()
