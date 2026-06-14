import torch
from torch import nn
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm_to_01(img_tensor_normed: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized BCHW image tensor back to [0, 1]."""
    mean = IMAGENET_MEAN.to(img_tensor_normed.device, img_tensor_normed.dtype)
    std = IMAGENET_STD.to(img_tensor_normed.device, img_tensor_normed.dtype)
    return (img_tensor_normed * std + mean).clamp(0, 1)


class SideWindowMeanFilter(nn.Module):
    def __init__(
        self,
        radius: int = 16,
        eps: float = 1e-3,
        tau: float = 0.001,
        prob_guided: bool = True,
        beta_center: float = 100,
        smooth_w: bool = True,
        w_smooth_radius: int = None,
    ):
        super().__init__()
        self.r = int(radius)
        self.eps = float(eps)
        self.tau = float(tau)
        self.prob_guided = bool(prob_guided)
        self.beta_center = float(beta_center)
        self.smooth_w = bool(smooth_w)
        self.w_smooth_radius = w_smooth_radius

        r = self.r
        self.windows = [
            (r, 0, r, r), (0, r, r, r), (r, r, r, 0), (r, r, 0, r),
            (r, 0, r, 0), (0, r, r, 0), (r, 0, 0, r), (0, r, 0, r)
        ]

    @staticmethod
    def _box_mean_sym(x: torch.Tensor, r: int) -> torch.Tensor:
        if r <= 0:
            return x
        x_pad = F.pad(x, (r, r, r, r), mode="replicate")
        return F.avg_pool2d(x_pad, kernel_size=2 * r + 1, stride=1)

    @staticmethod
    def _box_sum_asym(x: torch.Tensor, left, right, top, bottom) -> torch.Tensor:
        B, C, H, W = x.shape
        l, r, t, b = int(left), int(right), int(top), int(bottom)

        x_pad = F.pad(x, (l, r, t, b), mode="replicate")
        ii = x_pad.cumsum(dim=2).cumsum(dim=3)
        ii = F.pad(ii, (1, 0, 1, 0), mode="constant", value=0.0)

        hh, ww = t + b + 1, l + r + 1

        return (
            ii[:, :, hh:hh + H, ww:ww + W]
            - ii[:, :, 0:H, ww:ww + W]
            - ii[:, :, hh:hh + H, 0:W]
            + ii[:, :, 0:H, 0:W]
        )

    def forward(self, I: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # I: RGB guidance [B,3,H,W], p: CAM/probability [B,1,H,W]
        Ir, Ig, Ib = I[:, 0:1], I[:, 1:2], I[:, 2:3]

        costs = []
        mu_p_list = []

        for (l, rr, t, b) in self.windows:
            N = float((l + rr + 1) * (t + b + 1))

            mean_Ir = self._box_sum_asym(Ir, l, rr, t, b) / N
            mean_Ig = self._box_sum_asym(Ig, l, rr, t, b) / N
            mean_Ib = self._box_sum_asym(Ib, l, rr, t, b) / N
            mean_p = self._box_sum_asym(p, l, rr, t, b) / N

            mean_Irr = self._box_sum_asym(Ir ** 2, l, rr, t, b) / N
            mean_Igg = self._box_sum_asym(Ig ** 2, l, rr, t, b) / N
            mean_Ibb = self._box_sum_asym(Ib ** 2, l, rr, t, b) / N

            var_rgb = (
                (mean_Irr - mean_Ir ** 2)
                + (mean_Igg - mean_Ig ** 2)
                + (mean_Ibb - mean_Ib ** 2)
            )

            cost = var_rgb

            mu_I = torch.cat([mean_Ir, mean_Ig, mean_Ib], dim=1)
            cost_center = ((mu_I - I) ** 2).sum(dim=1, keepdim=True)
            cost = cost + self.beta_center * cost_center

            if self.prob_guided:
                mean_pp = self._box_sum_asym(p ** 2, l, rr, t, b) / N
                var_p = mean_pp - mean_p ** 2
                cost = cost + 10.0 * var_p

            costs.append(cost)
            mu_p_list.append(mean_p)

        cost_stack = torch.cat(costs, dim=1)       # [B,8,H,W]
        mu_p_stack = torch.stack(mu_p_list, dim=1) # [B,8,1,H,W]

        w = torch.softmax(-cost_stack / max(self.tau, 1e-6), dim=1)

        if self.smooth_w:
            rs = self.r if self.w_smooth_radius is None else int(self.w_smooth_radius)
            w = self._box_mean_sym(w, rs)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        q = (w.unsqueeze(2) * mu_p_stack).sum(dim=1)
        return q.clamp(0, 1)