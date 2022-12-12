import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np

COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

# in centimeters
ONE_COL_WIDTH = 8.3
TWO_COL_WIDTH = 17.1


__all__ = ["add_identity", "COLORS", "make_parity_plot", "set_plotstyle"]


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


GOLDEN_RATIO_FIG_ONE_COL = cm2inch(ONE_COL_WIDTH, ONE_COL_WIDTH / 1.618)
GOLDEN_RATIO_FIG_TWO_COL = cm2inch(TWO_COL_WIDTH, TWO_COL_WIDTH / 1.618)


def add_identity(axes, *line_args, **line_kwargs):
    (identity,) = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect("xlim_changed", callback)
    axes.callbacks.connect("ylim_changed", callback)
    return axes


def make_parity_plot(y_true, y_pred, ax, y_err=None, x_label="true", y_label="predicted", **kwargs):
    """
    X: (N, D)
    y: (N,)
    """
    if y_err is None:
        ax.scatter(y_true, y_pred, **kwargs)
    else:
        ax.errorbar(y_true, y_pred, yerr=y_err, **kwargs)
    add_identity(ax, "--k")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax


def set_plotstyle():
    plt.style.reload_library()
    plt.style.use("science")
    rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Helvetica"
