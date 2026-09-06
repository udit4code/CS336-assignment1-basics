import math


# Linear warmup reaches alpha_max at T_w, then cosine decay reaches alpha_min
# at T_c and stays there. Warmup can reduce instability from large early
# updates; cosine decay gradually lowers the step size later in training.


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
        cosine_term = math.cos(math.pi * (t - T_w) / (T_c - T_w))

        return alpha_min + 0.5 * (1 + cosine_term) * (alpha_max - alpha_min)

    return alpha_min
