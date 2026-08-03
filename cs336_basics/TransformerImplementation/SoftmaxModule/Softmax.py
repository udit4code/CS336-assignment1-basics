

import torch


# Time Complexity : If tensor has N elements, then, Softmax performs 1 max reduction, 1 substraction, 1 exponentiation, 1 sum reduction, and 1 division. 
# So, overall time complexity is O(N) with no extra memory beyond a few intermediate tensors
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
    # Eg : Say, x.shape = (2, 3), like [[2, 5, 1], [4, 8, 3]]
    # Now, torch.max(x, dim=1) = [max([2, 5, 1]), max([4, 8, 3])] = [5, 8], whose shape is (2, ), meaning that it is a 1 x 2 row vector
    # We cannot use it for broadcasting with a (2, 3) tensor, because the dimensions do not align the way we want. 
    # So, we do torch.max(x, dim=1, keepdim=True) so that we get [[5], [8]], whose shape is (2, 1) . Now, we can do broadcasting with (2, 3)
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

    # Step 3 :Exponentiate every element.
    exp = torch.exp(shifted)

    # Step 4 : Sum exponentials along the softmax dimension. with final shape: (..., 1, ...)
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