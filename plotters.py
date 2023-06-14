# import matplotlib.pyplot as plt
from typing import TypeVar

Axes = TypeVar("Axes")


def semilogy_plot(ax: Axes,
             y,
             x=None,
             label=None,
             ylabel=None,
             xlabel=None,
             title=None) -> Axes:
    args = [arg for arg in (x, y) if arg is not None]
    if label is None:
        ax.semilogy(*args)
    else:
        ax.semilogy(*args, label=label)
        ax.legend()
    # Set axes labels and title
    [f(arg) for f, arg in zip([ax.set_xlabel, ax.set_ylabel, ax.set_title],
                              [xlabel, ylabel, title]) if arg is not None]
    return ax
