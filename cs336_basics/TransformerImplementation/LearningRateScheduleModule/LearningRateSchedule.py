import math


# Learning rate schedules improve training stability.
# We first linearly warm up the learning rate to avoid unstable updates
# at the beginning of training, then gradually decay it using a cosine
# schedule so the optimizer takes smaller, more refined steps as training
# progresses.

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:

    assert T_w >= 0
    assert T_c >= T_w

    if t < T_w:
        return (t / T_w) * alpha_max

    if t <= T_c:
        cosine_term = math.cos(
            math.pi * (t - T_w) / (T_c - T_w)
        )

        return (
            alpha_min
            + 0.5
            * (1 + cosine_term)
            * (alpha_max - alpha_min)
        )

    return alpha_min