import torch
from torch import nn
from torch.nn import functional as F


def get_entropy(pred_probs):
    entropy = - (pred_probs * torch.log(pred_probs) + (1 - pred_probs) * torch.log(1 - pred_probs))
    return entropy


class ProSelfLC(nn.Module):
    """
    The implementation for progressive self label correction (CVPR 2021 paper).

    exp_base search: [4, 6, 8, 10, 12, 14,]
    transit_time_ratio: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    Note: Currently using the Conference version (Using conf_all, conf_top can be better in some case).
    """
    def __init__(
        self,
        total_steps: int, 
        exp_base: int=6,
        transit_time_ratio: float=0.2,
        relative=False,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.transit_time_ratio = transit_time_ratio
        self.exp_base = exp_base
        # self.temperature = 0.5
        self.temperature = 1
        self.conf_method = "all"
        self.relative = relative


    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            time_ratio_minus_half = torch.tensor(
                cur_time / self.total_steps - self.transit_time_ratio
            )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            # class_num = pred_probs.shape[1]
            if self.conf_method == "all":
                class_num = 2
                H_pred_probs = get_entropy(pred_probs)
                H_uniform = -torch.log(torch.tensor(1.0 / class_num))
                example_trust = 1 - H_pred_probs / H_uniform

                # the trade-off
            elif self.conf_method == "top":
                example_trust = torch.abs(2 * pred_probs - 1)

            self.epsilon = global_trust * example_trust


    def forward(
        self,
        preds,
        targets,
        cur_time: int,
        reduction="mean",
        pos_weight=1,
        **kwargs,
    ):
        if self.temperature != 1:
            with torch.no_grad():
                logit = torch.sigmoid(preds / (1 - preds))
                logit = logit / self.temperature
                scaled_preds = torch.sigmoid(logit)
        else:
            scaled_preds = preds
        self.update_epsilon_progressive_adaptive(scaled_preds, cur_time)

        new_target_probs = (1 - self.epsilon) * targets + self.epsilon * scaled_preds
        if pos_weight == 1:
            return F.binary_cross_entropy(preds, new_target_probs.detach(), reduction=reduction, **kwargs)


        if not self.relative:
            new_target_probs = new_target_probs.detach()
        loss = - pos_weight * new_target_probs * torch.log(preds) - (1 - new_target_probs) * torch.log(1 - preds)

        if reduction == "mean":
            return loss.mean()
        return loss

