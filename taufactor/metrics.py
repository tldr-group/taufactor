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

    
def triple_phase_boundary(img):
    phases = cp.unique(cp.asarray(img))
    if len(phases)!=3:
        return None
    shape = img.shape
    dim = len(shape)
    ph_maps = []
    img = cp.pad(cp.asarray(img), 1, 'constant', constant_values=-1)
    if dim==2:
        x, y = shape
        total_edges = (x-1)*(y-1)
        for ph in phases:
            ph_map = cp.zeros_like(img)
            ph_map_temp = cp.zeros_like(img)
            ph_map_temp[img==ph] = 1
            for i in [0, 1]:
                for j in [0, 1]:
                    ph_map += cp.roll(cp.roll(ph_map_temp, i, 0), j, 1)
            ph_maps.append(ph_map)
        tpb_map = cp.ones_like(img)
        for ph_map in ph_maps:
            tp_map *= ph_map
        tpb_map[tpb_map>1] = 1
        tpb_map = tpb_map[1:-1, 1:-1]
        tpb = np.sum(tpb_map)
    else:
        tpb = 0
        x, y, z = shape
        total_edges = z*(x-1)*(y-1) + x*(y-1)*(z-1) + y*(x-1)*(z-1)
        print(total_edges)
        for d in range(dim):
            ph_maps = []
            for ph in phases:
                ph_map = cp.zeros_like(img)
                ph_map_temp = cp.zeros_like(img)
                ph_map_temp[img==ph] = 1
                for i in [0, 1]:
                    for j in [0, 1]:
                        d1 =( d + 1) % 3
                        d2 = (d + 2) % 3
                        ph_map += cp.roll(cp.roll(ph_map_temp, i, d1), j, d2)
                ph_maps.append(ph_map)
            tpb_map = cp.ones_like(img)
            for ph_map in ph_maps:
                tpb_map *= ph_map
            tpb_map[tpb_map>1] = 1
            tpb_map = tpb_map[1:-1, 1:-1, 1:-1]
            tpb += np.sum(tpb_map)

    return tpb/total_edges