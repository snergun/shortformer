import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Plif(nn.Module):

    # base_interval: defines the logits range on which the monotonic layer
    # will be applied
    def __init__(self, K, T, w_variance):
        super(Plif, self).__init__()
        self.T = T
        self.K = K
        self.plif_w = nn.Parameter(
            torch.randn(self.K) * w_variance + math.log(math.exp(1) - 1)
        )

    # logits : size = num_ctxts * bs * num_vocab_words ,
    # i.e. <h,w> dot products
    def forward(self, logits):
        size = logits.size()
        logits = logits.view(-1)
        delta = 2. * self.T / self.K
        indices = torch.clamp(
            ((logits + self.T) / delta).detach().long(),
            max=self.K - 1, min=0
        )
        all_pos_w = nn.Softplus()(self.plif_w)
        all_pos_cumsum = torch.cumsum(all_pos_w, dim=-1) - all_pos_w
        pos_w = torch.gather(all_pos_w, -1, indices)
        # use gather, not take
        pos_w_cumsum = torch.gather(all_pos_cumsum, -1, indices)
        knots = (-self.T + delta * indices.float())
        knots = torch.tensor(
            knots, dtype=knots.dtype, device=logits.device
        )
        result = (logits - knots) * pos_w + delta * pos_w_cumsum
        return result.view(size)
