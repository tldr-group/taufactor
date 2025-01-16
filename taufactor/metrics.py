import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import label, generate_binary_structure

def volume_fraction(img, phases={}):
    """
    Calculates volume fractions of phases in an image
    :param img: segmented input image with n phases
    :param phases: a dictionary of phases to be calculated with keys as labels and phase values as values, default empty
    :return: list of volume fractions if no labels, dictionary of label: volume fraction pairs if labelled
    """

    if type(img) is not type(torch.tensor(1)):
        img = torch.tensor(img)

    if phases=={}:
        volume = torch.numel(img)
        labels, counts = torch.unique(img, return_counts=True)
        labels = labels.int()
        counts = counts.float()
        counts /= volume
        vf_out = {}
        for i, label in enumerate(labels):
            vf_out[str(label.item())] = counts[i].item()
    else:
        vf_out={}
        for p in phases:
            vf_out[p]=(img==phases[p]).to(torch.float).mean().item()

    return vf_out

def surface_area(img, phases, periodic=False):
    """
    Calculate interfacial surface area between two phases or the total surface area of one phase
    :param img:
    :param phases: list of phases to calculate SA, if lenght 1 calculate total SA, if length 2 calculate inerfacial SA
    :param periodic: list of bools indicating if the image is periodic in each dimension
    :return: the surface area in faces per unit volume
    """
    shape = img.shape
    int_not_in_img = int(np.unique(img).max()+1)

    dim = len(shape)
    img = torch.tensor(img)
    # finding an int that is not in the img for padding:
    
    if periodic:
        periodic.reverse()
        pad = ()
        for x in periodic:
            pad += tuple((int(not x),)*2)
        img = F.pad(img, pad, 'constant', value=int_not_in_img)
        periodic.reverse()
    else:
        img = F.pad(img, (1,)*dim*2, 'constant', value=int_not_in_img)
        periodic=[0]*dim

    SA_map = torch.zeros_like(img)
    if not isinstance(phases, list):
        raise TypeError('phases should be a list')
    for i in range(dim):
        for j in [1, -1]:
            i_rolled = torch.roll(img, j, i)
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
        sf = torch.sum(torch.tensor([x,y,z])[periodic_mask]*torch.roll(torch.tensor([x,y,z])[periodic_mask],1))
        total_faces = 3*(x*y*z)-sf
    elif dim == 2:
        sf = torch.sum(torch.tensor([x,y])[periodic_mask])
        total_faces = 2*(x+1)*(y+1)-(x+1)-(y+1)-2*sf
    else:
        total_faces=SA_map.size
    sa = torch.sum(SA_map)/total_faces
    return sa

def triple_phase_boundary(img):
    """Calculate triple phase boundary density i.e. fraction of voxel verticies that touch at least 3 phases

    Args:
        img (numpy array): image to calculate metric on     
    Returns:
        float: triple phase boundary density 
    """
    phases = torch.unique(torch.tensor(img))
    if len(phases)!=3:
        raise ValueError('Image must have exactly 3 phases')
    shape = img.shape
    dim = len(shape)
    ph_maps = []
    img = F.pad(torch.tensor(img), (1,)*dim*2, 'constant', value=-1)
    if dim==2:
        x, y = shape
        total_edges = (x-1)*(y-1)
        for ph in phases:
            ph_map = torch.zeros_like(img)
            ph_map_temp = torch.zeros_like(img)
            ph_map_temp[img==ph] = 1
            for i in [0, 1]:
                for j in [0, 1]:
                    ph_map += torch.roll(torch.roll(ph_map_temp, i, 0), j, 1)
            ph_maps.append(ph_map)
        tpb_map = torch.ones_like(img)
        for ph_map in ph_maps:
            tpb_map *= ph_map
        tpb_map[tpb_map>1] = 1
        tpb_map = tpb_map[1:-1, 1:-1]
        tpb = torch.sum(tpb_map)
    else:
        tpb = 0
        x, y, z = shape
        total_edges = z*(x-1)*(y-1) + x*(y-1)*(z-1) + y*(x-1)*(z-1)
        print(total_edges)
        for d in range(dim):
            ph_maps = []
            for ph in phases:
                ph_map = torch.zeros_like(img)
                ph_map_temp = torch.zeros_like(img)
                ph_map_temp[img==ph] = 1
                for i in [0, 1]:
                    for j in [0, 1]:
                        d1 =( d + 1) % 3
                        d2 = (d + 2) % 3
                        ph_map += torch.roll(torch.roll(ph_map_temp, i, d1), j, d2)
                ph_maps.append(ph_map)
            tpb_map = torch.ones_like(img)
            for ph_map in ph_maps:
                tpb_map *= ph_map
            tpb_map[tpb_map>1] = 1
            tpb_map = tpb_map[1:-1, 1:-1, 1:-1]
            tpb += torch.sum(tpb_map)

    return tpb/total_edges

def label_periodic(field, grayscale_value, neighbour_structure, periodic, debug=False):
    # Initialize phi field whith enlarged dimensions in periodic directions. Boundary values of
    # array are copied into ghost cells which are necessary to impose boundary conditions.
    padx = int(periodic[0])
    pady = int(periodic[1])
    padz = int(periodic[2])
    mask = np.pad(field, ((padx, padx), (pady, pady), (padz, padz)), mode='wrap')
    labeled_mask, num_labels = label(mask==grayscale_value, structure=neighbour_structure)
    count = 1
    for k in range(100):
        # Find indices where labels are different at the boundaries and create swaplist
        swap_list = np.zeros((1,2))
        if periodic[0]:
            # right x
            indices = np.where((labeled_mask[0,:,:]!=labeled_mask[-2,:,:]) & (labeled_mask[0,:,:]!=0) & (labeled_mask[-2,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[0,:,:][indices], labeled_mask[-2,:,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # left x
            indices = np.where((labeled_mask[1,:,:]!=labeled_mask[-1,:,:]) & (labeled_mask[1,:,:]!=0) & (labeled_mask[-1,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[1,:,:][indices], labeled_mask[-1,:,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        if periodic[1]:
            # top y
            indices = np.where((labeled_mask[:,0,:]!=labeled_mask[:,-2,:]) & (labeled_mask[:,0,:]!=0) & (labeled_mask[:,-2,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,0,:][indices], labeled_mask[:,-2,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # bottom y
            indices = np.where((labeled_mask[:,1,:]!=labeled_mask[:,-1,:]) & (labeled_mask[:,1,:]!=0) & (labeled_mask[:,-1,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,1,:][indices], labeled_mask[:,-1,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        if periodic[2]:
            # front z
            indices = np.where((labeled_mask[:,:,0]!=labeled_mask[:,:,-2]) & (labeled_mask[:,:,0]!=0) & (labeled_mask[:,:,-2]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,0][indices], labeled_mask[:,:,-2][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # back z
            indices = np.where((labeled_mask[:,:,1]!=labeled_mask[:,:,-1]) & (labeled_mask[:,:,1]!=0) & (labeled_mask[:,:,-1]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,1][indices], labeled_mask[:,:,-1][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        swap_list = swap_list[1:,:]
        # Sort swap list columns to ensure consistent ordering
        swap_list = np.sort(swap_list, axis=1)

        # Remove duplicates from swap_list
        swap_list = np.unique(swap_list, axis=0)
        # print(f"swap_list contains {swap_list.shape[0]} elements.")
        if (swap_list.shape[0]==0):
            break
        for i in range(swap_list.shape[0]):
            index = swap_list.shape[0] - i -1
            labeled_mask[labeled_mask == swap_list[index][1]] = swap_list[index][0]
        count += 1
    if(debug):
        print(f"Did {count} iterations for periodic labelling.")
    dim = labeled_mask.shape
    return labeled_mask[padx:dim[0]-padx,pady:dim[1]-pady,padz:dim[2]-padz], np.unique(labeled_mask).size-1

def find_spanning_labels(labelled_array, axis):
    """
    Find labels that appear on both ends along given axis

    Returns:
        set: Labels that appear on both ends of the first axis.
    """
    if axis == "x":
        front = np.s_[0,:,:]
        end   = np.s_[-1,:,:]
    elif axis == "y":
        front = np.s_[:,0,:]
        end   = np.s_[:,-1,:]
    elif axis == "z":
        front = np.s_[:,:,0]
        end   = np.s_[:,:,-1]
    else:
        raise ValueError("Axis should be x, y or z!")

    first_slice_labels = np.unique(labelled_array[front])
    last_slice_labels = np.unique(labelled_array[end])
    spanning_labels = set(first_slice_labels) & set(last_slice_labels)
    spanning_labels.discard(0)  # Remove the background label if it exists
    return spanning_labels

def extract_through_feature(array, grayscale_value, axis, periodic=[False,False,False], connectivity=1, debug=False):
    if array.ndim != 3:
        print(f"Expected a 3D array, but got an array with {array.ndim} dimension(s).")
        return None

    # Compute volume fraction of given grayscale value
    vol_phase = volume_fraction(array, phases={'1': grayscale_value})['1']

    # Define a list of connectivities to loop over
    connectivities_to_loop_over = [connectivity] if connectivity else range(1, 4)
    through_feature = []
    through_feature_fraction = np.zeros(len(connectivities_to_loop_over))

    # Compute the largest interconnected features depending on given connectivity
    count = 0
    for conn in connectivities_to_loop_over:
        # connectivity 1 = cells connected by sides (6 neighbours)
        # connectivity 2 = cells connected by sides & edges (14 neighbours)
        # connectivity 3 = cells connected by sides & edges & corners (26 neighbours)
        neighbour_structure = generate_binary_structure(3,conn)
        # Label connected components in the mask with given neighbour structure
        if any(periodic):
            labeled_mask, num_labels = label_periodic(array, grayscale_value, neighbour_structure, periodic, debug=debug)
        else:
            labeled_mask, num_labels = label(array == grayscale_value, structure=neighbour_structure)
        if(debug):
            print(f"Found {num_labels} labelled regions. For connectivity {conn} and grayscale {grayscale_value}.")

        through_labels = find_spanning_labels(labeled_mask,axis)
        spanning_network = np.isin(labeled_mask, list(through_labels))

        through_feature.append(spanning_network)
        through_feature_fraction[count] = volume_fraction(spanning_network, phases={'1': 1})['1']/vol_phase
        count += 1

    return through_feature, through_feature_fraction