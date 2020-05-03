import numpy as np


def plot_lines(x, y, vis, opts=None, env='main', win=None):
    if isinstance(y, dict):
        series = np.column_stack(list(y.values()))
        legends = list(y.keys())
    else:
        series = y
        legends = None

    if opts is None:
        opts = dict()
    opts['legend'] = legends
    plot_vals = dict(
        X=x,
        Y=series,
        opts=opts,
        env=env
    )

    if win is not None:
        plot_vals['win'] = win
    return vis.line(**plot_vals)
