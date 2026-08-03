
import torch


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    assert logits.ndim >= 2
    assert targets.shape == logits.shape[:-1]
    assert targets.dtype == torch.long

    max_logits = logits.max(dim=-1, keepdim=True).values

    stabilized_logits = logits - max_logits

    exp_logits = torch.exp(stabilized_logits)

    sum_exp = exp_logits.sum(dim=-1)

    log_sum_exp = torch.log(sum_exp)

    target_logits = stabilized_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    loss = log_sum_exp - target_logits

    return loss.mean()