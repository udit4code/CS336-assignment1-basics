import torch


# For N tensor elements, this implementation is O(N) work. It materializes
# shifted values and exponentials, so auxiliary memory is also O(N), although
# optimized fused kernels can use less intermediate storage.
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to apply softmax.

    Returns:
        Tensor of the same shape as x.
    """

    # Step 1 : Find the maximum value along the specified dimension.
    # keepdim=True preserves the dimension so broadcasting works.
    # Example for dim=1: x = [[2, 5, 1], [4, 8, 3]], shape (2,3).
    # torch.max(x, dim=1) gives [5, 8] with rank-1 shape (2,), not a 1x2
    # matrix. It cannot broadcast against (2,3) along the intended axis.
    # keepdim=True instead yields [[5], [8]], shape (2,1), which broadcasts.
    # Thus, [[2, 5, 1], [4, 8, 3]] with shape (2, 3) - [[5], [8]] with shape (2, 1)
    # = [[2, 5, 1], [4, 8, 3]] with shape (2, 3) - [[5, 5, 5], [8, 8, 8]] with shape (2, 3)
    # = [[-3, 0, -4], [-4, 0, -5]]
    # Shape: (..., 1, ...)
    max_vals = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    # Step 2 : Subtract the maximum for numerical stability. The largest element becomes 0.
    shifted = x - max_vals

    # Exponentiate every shifted element.
    exp = torch.exp(shifted)

    # Sum along the softmax axis while retaining a singleton dimension.
    exp_sum = torch.sum(
        exp,
        dim=dim,
        # Why keepdim=True ? So that we can broadcast when we opt for exp / exp_sum.
        # Think of keepdim=True as saying: "I'm reducing this dimension to a single value, but don't remove the dimension. Leave it there with size 1."
        # That singleton dimension is what allows broadcasting to work naturally.
        keepdim=True,
    )

    # Step 5 : Normalize
    return exp / exp_sum
