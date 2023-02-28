import torch
import matplotlib.pyplot as plt

def flux_direction(im, outpath=None):
    """
    Plots the flux direction of the image and provides code for transposing the image to change the flux direction
    :param im: segmented input image with n phases
    :return: None
    """

    fig, ax = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
    for i, dir in enumerate([(0,1,2),(1,2,0),(2,0,1)]):
        im_temp = torch.permute(torch.tensor(im), dir).cpu()[0]
        x,y = im_temp.shape
        ax[i].imshow(im_temp, cmap='gray')
        ax[i].axis('off')
        if i==0:
            ax[i].set_title("img", family='monospace')
        else:
            ax[i].set_title(f"torch.permute(img,{dir})", family='monospace')
        plt.suptitle('Direction of flux is into the page.\nThe transformations required to change direction of flux are shown.', fontsize=12)
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()