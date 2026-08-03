
import torch


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    assert logits.ndim >= 2, f"logits.ndim = {logits.ndim} is not greater than or equal to 2"
    assert targets.shape == logits.shape[:-1] 
    assert targets.dtype == torch.long or targets.dtype == torch.int64, f"targets.dtype = {targets.dtype} is not long or int64"

    # Say, logits have a shape (batch_size, seq_len, vocab_size) = (2, 3, 5) 
    # and, targets have a shape (batch_size, seq_len) = (2, 3). 
    # Step 1 : We get the maximum logits over the last dimension. 
    # So, logits.max(dim=-1, keepdim=True).values return max_logits with shape (2, 3, 1). If we had made keepdim=False, it would have been (2, 3) and broadcasting would have failed.
    max_logits = logits.max(dim=-1, keepdim=True).values

    # Step 2 : Via broadcasting, stabilize logits as : logits (2, 3, 5) - max_logits (2, 3, 1) gives a tensor with shape (2, 3, 5).
    # We do this step for numerical stability.
    stabilized_logits = logits - max_logits

    # Step 3 : Exp on each item of stablized_logits. No change in shape of tensor.
    # So, exp_logits still have shape (2, 3, 5)
    exp_logits = torch.exp(stabilized_logits)

    # Step 4 : Now, we do a sum-reduction, so, shape of sum-exp is (2, 3). We go from (2, 3, 5) to (2, 3)
    sum_exp = exp_logits.sum(dim=-1)

    # Step 5 : Apply log on sum_exp. No change in shape. So, shape is still (2, 3), which is (B, S)
    log_sum_exp = torch.log(sum_exp)

    # Step 6:
    # tensor.gather(dim=a, index=b) means:
    # "For every location, select one or more elements along dimension 'a' using the indices provided in 'index'."
    # Example:
    # x =
    # [[10, 20, 30],
    #  [40, 50, 60]]
    # Shape: (2, 3)
    # index =
    # [[2],
    #  [1]]
    # Shape: (2, 1)
    # x.gather(dim=1, index=index) returns
    #
    # [[30],
    #  [50]]
    # because along dimension 1 (the last dimension here),
    # row 0 selects column 2 -> 30
    # row 1 selects column 1 -> 50.
    #
    # In our language model:
    # stabilized_logits has shape (batch_size, seq_len, vocab_size)
    # targets has shape (batch_size, seq_len)
    #
    # targets.unsqueeze(-1) changes the shape to
    # (batch_size, seq_len, 1), which is required by gather.
    #
    # gather(dim=-1, ...) selects the logit corresponding to the
    # correct target token from each vocabulary vector.
    #
    # The result has shape (batch_size, seq_len, 1).
    # squeeze(-1) removes the last singleton dimension, producing a tensor of shape (batch_size, seq_len).
    target_logits = stabilized_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # Step 7 : via broadcasting, (B, S) - (B, S)
    loss = log_sum_exp - target_logits

    # Finally, return the average cross entropy across all examples. So, we have to take mean of loss.
    return loss.mean()

# Math behind cross-entropy : 

# Cross-entropy loss for the correct class y:
#     l = -log(exp(x_y) / Σ_i exp(x_i))
#
# Expanding the logarithm:
#     l = -(x_y - log(Σ_i exp(x_i)))
#
# For numerical stability, let
#     m = max_i x_i
#
# Since subtracting the same constant from every logit does not change
# the softmax probabilities, we can rewrite the loss as
#     l = -((x_y - m) - log(Σ_i exp(x_i - m)))