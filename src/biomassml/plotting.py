import matplotlib.pyplot as plt
from matplotlib import rcParams

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

__all__ = ["add_identity", "COLORS", "make_parity_plot", "set_plotstyle"]


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


def make_parity_plot(X, y, ax, **kwargs):
    """
    X: (N, D)
    y: (N,)
    """
    ax.scatter(X[:, 0], y, **kwargs)
    add_identity(ax)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$y$")
    ax.set_title("Parity Plot")
    return ax


def set_plotstyle():
    plt.style.reload_library()
    plt.style.use("science")
    rcParams["font.family"] = "sans-serif"
