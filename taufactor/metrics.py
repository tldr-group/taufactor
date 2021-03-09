import numpy as np
import cupy as cp

def volume_fraction(img, phases={}):
    """
    Calculates volume fractions of phases in an image
    :param img: segmented input image with n phases
    :param phases: a dictionary of phases to be calculated with keys as labels and phase values as values, default empty
    :return: list of volume fractions if no labels, dictionary of label: volume fraction pairs if labelled
    """

    if type(img) is not type(cp.array(1)):
        img = cp.asarray(img)

    if phases=={}:
        phases = cp.unique(img)
        vf_out = []
        for p in phases:
            vf_out.append((img==p).mean().item())
        if len(vf_out)==1:
            vf_out=vf_out[0]


    else:
        vf_out={}
        for p in phases:
            vf_out[p]=(img==phases[p]).mean().item()
    
    return vf_out