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


def hsv2bgr(hsv):
    h, s, v = hsv
    c = s * v
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        b, g, r = 0, x, c
    elif 60 <= h < 120:
        b, g, r = 0, c, x
    elif 120 <= h < 180:
        b, g, r = x, c, 0
    elif 180 <= h < 240:
        b, g, r = c, x, 0
    elif 240 <= h < 270:
        b, g, r = c, 0, x
    else:
        b, g, r = x, 0, c
    return int((b + m) * 255), int((g + m) * 255), int((r + m) * 255)
