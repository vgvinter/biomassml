import time
import numpy as np


def get_timestring():
    return time.strftime("%Y%m%d-%H%M%S")


def additive_errorprop(errors):
    """Asuming that we deal with standard deviations"""
    return np.sqrt(np.sum(np.square(errors), axis=0))
