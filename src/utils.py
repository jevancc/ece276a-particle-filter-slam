import os
import pylab
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc


def plot_map(mp,
             pos=None,
             npos=None,
             figsize=20,
             save_fig_name=None,
             navigation_heading=None,
             show_navigation=False):
    PATH_COLOR = '#ff4733'
    NAVIGATION_COLOR = '#3888ff'

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=80)
    mp = np.flip(mp, 1)
    if mp.ndim == 2:
        ax.imshow(mp, cmap='bone')
    else:
        ax.imshow(mp)

    if pos is not None:
        posx, posy = pos
        assert len(posx) == len(posy)

        for i, (px, py) in enumerate(zip(posx, posy)):
            ax.plot(mp.shape[1] - px, py, marker='o', color=PATH_COLOR, ms=1)

    if show_navigation:
        npos = len(posx)
        px, py = posx[-1], posy[-1]
        dx = -np.cos(navigation_heading)
        dy = np.sin(navigation_heading)

        ax.arrow(mp.shape[1] - px,
                 py,
                 dx,
                 dy,
                 length_includes_head=True,
                 head_width=15,
                 head_length=20,
                 head_starts_at_zero=True,
                 overhang=0.2,
                 zorder=999,
                 facecolor=NAVIGATION_COLOR,
                 edgecolor='black')

    if save_fig_name:
        Path(os.path.dirname(save_fig_name)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig_name, bbox_inches='tight')
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        del fig
        del ax
        gc.collect()
        return None, None
    else:
        plt.show(block=True)
        return fig, ax
