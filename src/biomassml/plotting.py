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


def radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """Draw. If frame is polygon, make gridlines polygon-shaped"""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self, spine_type="circle", path=Path.unit_regular_polygon(num_vars)
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)

                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


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


def make_parity_plot(y_true, y_pred, ax, x_label="true", y_label="predicted", **kwargs):
    """
    X: (N, D)
    y: (N,)
    """
    ax.scatter(y_true, y_pred, **kwargs)
    add_identity(ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax


def set_plotstyle():
    plt.style.reload_library()
    plt.style.use("science")
    rcParams["font.family"] = "sans-serif"


def spider(data, labels, outname):
    """
        Example for data:
        data = [
        ["geometry", "metal", "ligand", "charge"],
        (
            "",
            [
                [
                    filtered_geometry_shap,
                    filtered_metal_shap,
                    filtered_ligand_shap,
                    filtered_charge_shap,
                ],
                [
                    unfiltered_geometry_shap,
                    unfiltered_metal_shap,
                    unfiltered_ligand_shap,
                    unfiltered_charge_shap,
                ],
            ],
        ),
    ]
    """
    N = len(data[0])
    theta = radar_factory(N, frame="polygon")

    spoke_labels = data.pop(0)
    title, case_data = data[0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, position=(0.5, 1.1), ha="center")
    labels = ["filtered", "unfiltered"]
    for i, d in enumerate(case_data):
        line = ax.plot(theta, d, c=COLORS[i])
        ax.fill(theta, d, alpha=0.25, c=COLORS[i], edgecolor=COLORS[i], label=labels[i])
    ax.set_varlabels(spoke_labels)

    fig.legend()
    if outname is not None:
        fig.savefig(outname, bbox_inches="tight")

    return fig, ax
