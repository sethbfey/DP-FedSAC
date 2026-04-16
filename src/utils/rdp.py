# utils/rdp.py

import math

# Per-round RDP cost of the Poisson-subsample Gaussian mechanism
#   Wang et al. (2019) "Subsampled Rényi Differential Privacy and Analytical Moments Accountant."
#   https://arxiv.org/abs/1908.10530
def rdp_per_round(alpha, sigma, q):
    if alpha < 2 or not isinstance(alpha, int):
        raise ValueError("alpha must be an INT >= 2")

    log_terms = []

    for k in range(alpha + 1):
        log_binom     = (math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1))
        log_q_power   = k * math.log(q) if k > 0 else 0.0
        log_1mq_power = (alpha - k) * math.log(1.0 - q) if alpha - k > 0 else 0.0
        log_gauss     = k * (k - 1) / (2.0 * sigma ** 2)
        log_terms.append(log_binom + log_q_power + log_1mq_power + log_gauss)

    max_log = max(log_terms)
    log_sum = max_log + math.log(sum(math.exp(t - max_log) for t in log_terms))
    return log_sum / (alpha - 1)

# (epsilon, alpha)-RDP -> (epsilon, delta)-DP
def rdp_to_dp(epsilon_rdp, alpha, delta):
    return epsilon_rdp + math.log(1.0 / delta) / (alpha - 1)
