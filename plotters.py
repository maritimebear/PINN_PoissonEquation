# import matplotlib.pyplot as plt
from typing import TypeVar

Axes = TypeVar("Axes")


def LogYPlot(ax: Axes,
             y,
             x=None,
             ylabel=None,
             xlabel=None,
             title=None) -> Axes:
    if x is None:
        ax.semilogy(y)
    else:
        ax.semilogy(x, y)
    [f(arg) for f, arg in zip([ax.set_xlabel, ax.set_ylabel, ax.set_title],
                              [xlabel, ylabel, title]) if arg is not None]
    return ax
