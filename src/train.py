import os
import sys
import json
import time
import logging
from typing import List

import click
import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageFile
from torch import nn, optim
from torch.utils import data, tensorboard

from src import configs
from src import data as lc_data
from src import network
from src.l3c import timer

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_bpsp(plotter, bits, inp_size, train_iter):
    """ Vẽ BPSP trên TensorBoard """
    for key in bits.get_keys():
        plotter.add_scalar(f"{key}_self_bpsp", bits.get_self_bpsp(key).item(), train_iter)
        plotter.add_scalar(f"{key}_scaled_bpsp", bits.get_scaled_bpsp(key, inp_size).item(), train_iter)


def train_epoch(train_loader, compressor, optimizer, plotter, plot_iters, clip, epoch):
    """ Huấn luyện 1 epoch """
    compressor.train()
    total_loss = 0
    total_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (filenames, x) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.cuda()
        inp_size = np.prod(x.size())

        bits = compressor(x, filenames)  # Lưu xác suất khi cần
        loss = bits.get_total_bpsp(inp_size)
        loss.backward()
        nn.utils.clip_grad_norm_(compressor.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

        # Hiển thị log mỗi `plot_iters` batch
        if batch_idx % plot_iters == 0:
            plotter.add_scalar("train/bpsp", loss.item(), epoch * total_batches + batch_idx)
            logging.info(f"Epoch {epoch} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_batches
    end_time = time.time()
    logging.info(f"✅ Epoch {epoch} completed in {end_time - start_time:.2f}s - Avg Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(eval_loader, compressor, plotter, epoch):
    """ Chạy đánh giá trên tập validation """
    compressor.eval()
    total_loss = 0
    inp_size = 0
    total_batches = len(eval_loader)
    start_time = time.time()

    with torch.no_grad():
        bits_keeper = network.Bits()
        for filenames, x in eval_loader:
            x = x.cuda()
            inp_size += np.prod(x.size())
            bits = compressor(x, filenames)
            bits_keeper.add_bits(bits)

        total_bpsp = bits_keeper.get_total_bpsp(inp_size).item()
        plotter.add_scalar("eval/bpsp", total_bpsp, epoch)
        logging.info(f"🔍 Validation Epoch {epoch} - BPSP: {total_bpsp:.4f}")

    end_time = time.time()
    logging.info(f"✅ Evaluation completed in {end_time - start_time:.2f}s")
    return total_bpsp


@click.command()
@click.option("--train-path", type=click.Path(exists=True), help="Path to training images.")
@click.option("--eval-path", type=click.Path(exists=True), help="Path to eval images.")
@click.option("--train-file", type=click.File("r"), help="File with training image names.")
@click.option("--eval-file", type=click.File("r"), help="File with eval image names.")
@click.option("--batch", type=int, default=16, help="Batch size.")
@click.option("--epochs", type=int, default=50, help="Number of epochs.")
@click.option("--lr", type=float, default=1e-4, help="Learning rate.")
@click.option("--clip", type=float, default=0.5, help="Gradient clipping norm.")
@click.option("--plot-iters", type=int, default=100, help="Plot every N iterations.")
def main(train_path, eval_path, train_file, eval_file, batch, epochs, lr, clip, plot_iters):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Cấu hình logging và TensorBoard
    logging.info("🚀 Starting training...")
    os.makedirs(configs.plot, exist_ok=True)

    # Load dataset
    train_dataset = lc_data.ImageFolder(
        train_path, [filename.strip() for filename in train_file],
        configs.scale,
        T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(128)]),
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=2, drop_last=True)

    eval_dataset = lc_data.ImageFolder(
        eval_path, [filename.strip() for filename in eval_file],
        configs.scale,
        T.Lambda(lambda x: x),
    )
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    # Khởi tạo mô hình và optimizer
    compressor = network.Compressor().cuda()
    optimizer = optim.Adam(compressor.parameters(), lr=lr)
    plotter = tensorboard.SummaryWriter(configs.plot)

    # Lặp qua các epoch
    for epoch in range(epochs):
        avg_train_loss = train_epoch(train_loader, compressor, optimizer, plotter, plot_iters, clip, epoch)
        eval_bpsp = evaluate(eval_loader, compressor, plotter, epoch)

        # Lưu mô hình nếu tốt nhất
        if configs.best_bpsp > eval_bpsp:
            configs.best_bpsp = eval_bpsp
            torch.save({
                "nets": compressor.nets.state_dict(),
                "best_bpsp": configs.best_bpsp,
                "epoch": epoch,
            }, os.path.join(configs.plot, "best.pth"))
            logging.info(f"🎯 New best model saved at epoch {epoch} with BPSP {eval_bpsp:.4f}")

    logging.info("🏁 Training complete!")


if __name__ == "__main__":
    main()
