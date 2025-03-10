import os
import json
from collections import defaultdict
from typing import (DefaultDict, Generator, KeysView, List, NamedTuple,
                    Optional, Tuple)

import numpy as np
import torch
from torch import nn

from src import configs, data, util
from src.l3c import edsr
from src.l3c import logistic_mixture as lm
from src.l3c import prob_clf, quantizer


class LogisticMixtureProbability(NamedTuple):
    name: str
    pixel_index: int
    probs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


Probs = Tuple[torch.Tensor, Optional[LogisticMixtureProbability], int]


class Bits:
    """
    Tracks bpsps from different parts of the pipeline for one forward pass.
    """

    def __init__(self) -> None:
        assert configs.collect_probs or configs.log_likelihood, (
            configs.collect_probs, configs.log_likelihood)
        self.key_to_bits: DefaultDict[
            str, torch.Tensor] = defaultdict(float)  # type: ignore
        self.key_to_sizes: DefaultDict[str, int] = defaultdict(int)
        self.probs: List[Probs] = []  # Danh sách lưu xác suất nếu cần

    def add_with_size(
            self, key: str, nll_sum: torch.Tensor, size: int,
    ) -> None:
        if configs.log_likelihood:
            assert key not in self.key_to_bits, f"{key} already exists"
            self.key_to_bits[key] = nll_sum / np.log(2)
            self.key_to_sizes[key] = size

    def add(self, key: str, nll: torch.Tensor) -> None:
        self.add_with_size(
            key, nll.sum(), np.prod(nll.size()))

    def add_lm(
            self, y_i: torch.Tensor,
            lm_probs: LogisticMixtureProbability,
            loss_fn: lm.DiscretizedMixLogisticLoss) -> None:
        assert lm_probs.probs.shape[-2:] == y_i.shape[-2:], (
            lm_probs.probs.shape, y_i.shape)
        if configs.log_likelihood:
            nll = loss_fn(y_i, lm_probs.probs)
            self.add(lm_probs.name, nll)
        if configs.collect_probs:
            self.probs.append((y_i, lm_probs, -1))

    def save_probs_to_json(self, filenames: List[str], json_filename=configs.prob_save_path):
        """Lưu tên ảnh và xác suất vào file JSON"""
        data = []
        if os.path.exists(json_filename):
            with open(json_filename, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []

        for filename, (y_i, lm_probs, _) in zip(filenames, self.probs):
            if lm_probs is not None:
                probs = lm_probs.probs.cpu().detach().numpy().tolist()
                data.append({"filename": filename, "probs": probs})

        with open(json_filename, "w") as file:
            json.dump(data, file, indent=4)

        print(f"[DEBUG] Saved probabilities for {len(filenames)} images.")

    def get_total_bpsp(self, inp_size: int) -> torch.Tensor:
        return sum(self.key_to_bits.values()) / inp_size  # type: ignore


class PixDecoder(nn.Module):
    """ Super-resolution based decoder for pixel-based factorization. """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.scale = scale

    def forward_probs(
            self,
            x: torch.Tensor,
            ctx: torch.Tensor
    ) -> Generator[LogisticMixtureProbability, torch.Tensor,
                   Tuple[torch.Tensor, torch.Tensor]]:        
        raise NotImplementedError

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            ctx: torch.Tensor,
    ) -> Tuple[Bits, torch.Tensor]:
        bits = Bits()

        if __debug__:
            not_int = y.long().float() != y
            assert not torch.any(not_int), y[not_int]

        mode = "train" if self.training else "eval"
        deltas = x - util.tensor_round(x)
        bits.add_uniform(
            f"{mode}/{self.scale}_rounding",
            quantizer.to_sym(deltas, x_min=-0.25, x_max=0.5, L=4),
            levels=4)

        y_slices = group_2x2(y)
        gen = self.forward_probs(x, ctx)

        try:
            for i, y_slice in enumerate(y_slices):
                if i == 0:
                    lm_probs = next(gen)
                else:
                    lm_probs = gen.send(y_slices[i-1])
                bits.add_lm(y_slice, lm_probs, self.loss_fn)

        except StopIteration as e:
            last_pixels, ctx = e.value
            last_slice = y_slices[-1]
            _, _, last_h, last_w = last_slice.size()
            last_pixels = last_pixels[..., : last_h, : last_w]
            assert torch.all(last_pixels == last_slice), (
                last_pixels[last_pixels != last_slice],
                last_slice[last_pixels != last_slice])

        return bits, ctx


class Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert configs.scale >= 0, configs.scale

        self.loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.ctx_upsamplers = nn.ModuleList([
            nn.Identity(),  # type: ignore
            *[edsr.Upsampler(scale=2, n_feats=configs.n_feats)
              for _ in range(configs.scale-1)]
        ] if configs.scale > 0 else [])
        self.decs = nn.ModuleList([
            StrongPixDecoder(i) for i in range(configs.scale)
        ])
        self.nets = nn.ModuleList([
            self.ctx_upsamplers, self.decs,
        ])

    def forward(
            self,
            x: torch.Tensor,
            filenames: List[str] = None
    ) -> Bits:
        downsampled = data.average_downsamples(x)
        bits = Bits()
        ctx = 0.

        for dec, ctx_upsampler, x, y, in zip(
                self.decs, self.ctx_upsamplers,
                downsampled[::-1], downsampled[-2::-1]):
            ctx = ctx_upsampler(ctx)
            dec_bits, ctx = dec(x, util.tensor_round(y), ctx)
            bits.update(dec_bits)

        if configs.collect_probs and filenames:
            bits.save_probs_to_json(filenames)

        return bits
