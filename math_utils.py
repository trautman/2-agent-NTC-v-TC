# math_utils.py

import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.special import logsumexp as scipy_logsumexp
from scipy.stats import entropy as scipy_entropy


def logsumexp(arr):
    # LIBRARY CALL: scipy.special.logsumexp
    return scipy_logsumexp(arr)


def softmax_from_logweights(logw):
    # LIBRARY CALL: scipy.special.softmax
    return scipy_softmax(logw)


def kl_divergence(p, q):
    # LIBRARY CALL: scipy.stats.entropy(pk, qk)
    # Computes KL(p || q) = sum p * log(p / q)
    return float(scipy_entropy(p.ravel(), q.ravel()))


def normalize_matrix(mat, eps=1e-12):
    # LIBRARY CALLS: np.min, np.ptp
    # np.ptp(mat) = np.max(mat) - np.min(mat)
    return (mat - np.min(mat)) / (np.ptp(mat) + eps)