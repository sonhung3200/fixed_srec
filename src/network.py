import torch
import torch.nn as nn
from collections import defaultdict
from typing import DefaultDict, Generator, KeysView, List, NamedTuple, Optional, Tuple
import numpy as np

from src import configs, data, util
from src.l3c import edsr, logistic_mixture as lm, prob_clf, quantizer


class LogisticMixtureProbability(NamedTuple):
    name: str
    pixel_index: int
    probs: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


Probs = Tuple[torch.Tensor, Optional[LogisticMixtureProbability], int]


class Bits:
    """ Tracks bpsps from different parts of the pipeline for one forward pass. """

    def __init__(self) -> None:
        assert configs.collect_probs or configs.log_likelihood, (
            configs.collect_probs, configs.log_likelihood)
        self.key_to_bits: DefaultDict[str, torch.Tensor] = defaultdict(float)
        self.key_to_sizes: DefaultDict[str, int] = defaultdict(int)
        self.probs: List[Probs] = []

    def add_with_size(self, key: str, nll_sum: torch.Tensor, size: int) -> None:
        if configs.log_likelihood:
            assert key not in self.key_to_bits, f"{key} already exists"
            self.key_to_bits[key] = nll_sum / np.log(2)
            self.key_to_sizes[key] = size

    def add(self, key: str, nll: torch.Tensor) -> None:
        self.add_with_size(key, nll.sum(), np.prod(nll.size()))

    def add_lm(self, y_i: torch.Tensor, lm_probs: LogisticMixtureProbability, loss_fn: lm.DiscretizedMixLogisticLoss) -> None:
        assert lm_probs.probs.shape[-2:] == y_i.shape[-2:], (
            lm_probs.probs.shape, y_i.shape)
        if configs.log_likelihood:
            nll = loss_fn(y_i, lm_probs.probs)
            self.add(lm_probs.name, nll)
        if configs.collect_probs:
            self.probs.append((y_i, lm_probs, -1))

    def get_total_bpsp(self, inp_size: int) -> torch.Tensor:
        return sum(self.key_to_bits.values()) / inp_size

    def update(self, other: "Bits") -> "Bits":
        """ Hợp nhất dữ liệu từ một object Bits khác """
        assert len(self.get_keys() & other.get_keys()) == 0, \
            f"{self.get_keys()} và {other.get_keys()} bị trùng."
        self.key_to_bits.update(other.key_to_bits)
        self.key_to_sizes.update(other.key_to_sizes)
        self.probs += other.probs
        return self

    def get_keys(self):
        """Trả về danh sách các keys trong `key_to_bits`"""
        return self.key_to_bits.keys()




class PixDecoder(nn.Module):
    """ Super-resolution based decoder for pixel-based factorization. """

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor) -> Tuple[Bits, torch.Tensor]:
        bits = Bits()
        mode = "train" if self.training else "eval"
        deltas = x - util.tensor_round(x)
        bits.add_with_size(f"{mode}/{self.scale}_rounding",
                   np.log(4) * np.prod(deltas.size()), 
                   np.prod(deltas.size()))

        _, _, x_h, x_w = x.size()
        if not isinstance(ctx, float):
            ctx = ctx[..., :x_h, :x_w]

        y_slices = group_2x2(y)
        gen = self.forward_probs(x, ctx)

        try:
            for i, y_slice in enumerate(y_slices):
                lm_probs = next(gen) if i == 0 else gen.send(y_slices[i-1])
                bits.add_lm(y_slice, lm_probs, self.loss_fn)
        except StopIteration as e:
            last_pixels, ctx = e.value
            assert torch.all(last_pixels == y_slices[-1])

        return bits, ctx


class StrongPixDecoder(PixDecoder):
    def __init__(self, scale: int) -> None:
        super().__init__(scale)
        self.rgb_decs = nn.ModuleList([
            edsr.EDSRDec(3 * i, configs.n_feats, resblocks=configs.resblocks, tail="conv")
            for i in range(1, 4)
        ])
        self.mix_logits_prob_clf = nn.ModuleList([
            prob_clf.AtrousProbabilityClassifier(
                configs.n_feats, C=3, K=configs.K, num_params=self.loss_fn._num_params)
            for _ in range(1, 4)
        ])
        self.feat_convs = nn.ModuleList([
            util.conv(configs.n_feats, configs.n_feats, 3) for _ in range(1, 4)
        ])

    def forward_probs(self, x: torch.Tensor, ctx: torch.Tensor) -> Generator[LogisticMixtureProbability, torch.Tensor,
                                                                             Tuple[torch.Tensor, torch.Tensor]]:
        mode = "train" if self.training else "eval"
        pix_sum = x * 4
        xy_normalized = x / 127.5 - 1
        y_i = torch.tensor([], device=x.device)
        z: torch.Tensor = 0.

        for i, (rgb_dec, clf, feat_conv) in enumerate(zip(self.rgb_decs, self.mix_logits_prob_clf, self.feat_convs)):
            xy_normalized = torch.cat((xy_normalized, y_i / 127.5 - 1), dim=1)
            z = rgb_dec(xy_normalized, ctx)
            ctx = feat_conv(z)

            probs = clf(z)
            lower = torch.max(pix_sum - (3 - i) * 255, torch.tensor(0., device=x.device))
            upper = torch.min(pix_sum, torch.tensor(255., device=x.device))

            y_i = yield LogisticMixtureProbability(f"{mode}/{self.scale}_{i}", i, probs, lower, upper)
            y_i = data.pad(y_i, x.shape[-2], x.shape[-1])
            pix_sum -= y_i

        return pix_sum, ctx


def group_2x2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, h, w = x.size()
    x_even_height = x[:, :, 0:h:2, :]
    x_odd_height = x[:, :, 1:h:2, :]
    return (
        x_even_height[:, :, :, 0:w:2],
        x_even_height[:, :, :, 1:w:2],
        x_odd_height[:, :, :, 0:w:2],
        x_odd_height[:, :, :, 1:w:2]
    )


class Compressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert configs.scale >= 0, configs.scale

        self.loss_fn = lm.DiscretizedMixLogisticLoss(rgb_scale=True)
        self.ctx_upsamplers = nn.ModuleList([
            nn.Identity(),
            *[edsr.Upsampler(scale=2, n_feats=configs.n_feats) for _ in range(configs.scale-1)]
        ] if configs.scale > 0 else [])
        self.decs = nn.ModuleList([StrongPixDecoder(i) for i in range(configs.scale)])

        assert len(self.ctx_upsamplers) == len(self.decs), \
            f"{len(self.ctx_upsamplers)}, {len(self.decs)}"

        self.nets = nn.ModuleList([self.ctx_upsamplers, self.decs])

        # 🔥 Dùng DataParallel nếu có nhiều GPU
        if torch.cuda.device_count() > 1:
            print(f"✅ Using {torch.cuda.device_count()} GPUs!")
            self.nets = nn.DataParallel(self.nets)

    def forward(self, x: torch.Tensor) -> Bits:
        x = x.cuda()
        downsampled = data.average_downsamples(x)
        mode = "train" if self.training else "eval"
        bits = Bits()
        bits.add_with_size(f"{mode}/codes_0", util.tensor_round(downsampled[-1]), downsampled[-1].numel())

        ctx = 0.
        for dec, ctx_upsampler, x, y in zip(self.decs, self.ctx_upsamplers, downsampled[::-1], downsampled[-2::-1]):
            ctx = ctx_upsampler(ctx)
            dec_bits, ctx = dec(x, util.tensor_round(y), ctx)
            bits.update(dec_bits)
        return bits
