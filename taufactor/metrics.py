import numpy as np
import cupy as cp

def volume_fraction(img, phases=[]):
    """
    Calculates volume fractions of phases in an image
    :param img: segmented input image with n phases
    :param phases: a dictionary of phases to be calculated with keys as labels and phase values as values, default empty
    :return: list of volume fractions if no labels, dictionary of label: volume fraction pairs if labelled
    """

    if type(img) is not type(cp.array(1)):
        img = cp.asarray(img)

    if list(phases)==[]:
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

def surface_area(img, phases, periodic=False):
    """
    Calculate interfacial surface area between two phases or the total surface area of one phase 
    :param img:
    :param phases: list of phases to calculate SA, if lenght 1 calculate total SA, if length 2 calculate inerfacial SA
    :return: the surface area in faces per unit volume
    """
    shape = img.shape
    dim = len(shape)
    if periodic:
        pad = [(int(not x),int(not x)) for x in periodic]
        img = cp.pad(cp.asarray(img), pad, 'constant', constant_values=-1)
    else:
        img = cp.pad(cp.asarray(img), 1, 'constant', constant_values=-1)
        periodic=[0]*dim

    SA_map = cp.zeros_like(img)
    if not isinstance(phases, list):
        phases = [phases]
    for i in range(dim):
        for j in [1, -1]:
            i_rolled = cp.roll(img, j, i)
            if len(phases)==2:
                SA_map[(i_rolled == phases[0]) & (img == phases[1])] += 1
            else:
                SA_map[(i_rolled == phases[0]) & (img != phases[0])] += 1
    # remove padding
    if not periodic[0]:
        SA_map = SA_map[1:-1, :]
    if not periodic[1]:
        SA_map = SA_map[:, 1:-1]
    x, y = shape[0], shape[1]
    # scale factor calculated by taking into account edges
    periodic_mask=[not x for x in periodic]
    if dim == 3:
        z = shape[2]
        if not periodic[2]:
            SA_map = SA_map[:, :, 1:-1]
        sf = cp.sum(cp.array([x,y,z])[periodic_mask]*cp.roll(cp.array([x,y,z])[periodic_mask],1))
        total_faces = 3*(x*y*z)-sf
    elif dim == 2:
        sf = cp.sum(cp.array([x,y])[periodic_mask])
        total_faces = 2*(x+1)*(y+1)-(x+1)-(y+1)-2*sf
    else:
        total_faces=SA_map.size
    sa = cp.sum(SA_map)/total_faces
    return sa
