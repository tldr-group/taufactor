import numpy as np
import matplotlib.pyplot as plt


def flux_direction(im, outpath=None):
    """
    Plots the flux direction of the image and provides code for transposing the image to change the flux direction
    :param im: segmented input image with n phases
    :return: None
    """
    im = np.asarray(im)
    if im.ndim not in (2, 3):
        raise ValueError("im must be 2D or 3D")

    if im.ndim == 3:
        orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        titles = [
            "img",
            "np.transpose(img, (1,2,0))",
            "np.transpose(img, (2,0,1))",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, order, title in zip(axes, orders, titles):
            vol = np.transpose(im, order)
            sl = vol[vol.shape[0] // 2]  # middle slice after permutation
            ax.imshow(sl, cmap="gray")
            ax.set_title(title, fontfamily="monospace")
            ax.axis("off")
        fig.suptitle(
            "Direction of flux is into the page.\n"
            "Use the shown np.transpose to change the flux direction.",
            fontsize=12,
        )
    else:  # 2D
        orders = [(0, 1), (1, 0)]
        titles = ["img", "np.transpose(img, (1,0))"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        for ax, order, title in zip(axes, orders, titles):
            arr = np.transpose(im, order)
            ax.imshow(arr, cmap="gray")
            ax.set_title(title, fontfamily="monospace")
            ax.axis("off")
            # optional: annotate 'Flux' at the side
            h, w = arr.shape
            ax.annotate(
                "Flux", xy=(w * 1.02, h / 2), xytext=(w * 1.02, h / 2),
                textcoords="data", rotation=90, va="center", ha="left",
                bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=1),
                fontsize=12,
            )
        fig.suptitle(
            "Direction of flux is down.\n"
            "The transformations required to change direction of flux are shown.",
            fontsize=12,
        )
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    else:
        plt.show()
    return fig, axes
