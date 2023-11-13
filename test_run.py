import taufactor as tau
import torch
import matplotlib.pyplot as plt
import tifffile

def test_for_squires():
    # load image
    large_size = 400
    img = torch.ones(400, 400, 400)
    end_idx = 379
    beginning = (large_size - end_idx) // 2
    cube_img = torch.zeros(end_idx, end_idx, end_idx)
    img[beginning:beginning+end_idx,beginning:beginning+end_idx,beginning:beginning+end_idx] = cube_img
    # ensure 1s for conductive phase and 0s otherwise.
    plt.imshow(img[0].cpu().numpy())
    plt.show()
    # create a solver object with loaded image
    s = tau.Solver(img)

    # call solve function
    s.solve()

    # view effective diffusivity and tau
    print(s.D_eff, s.tau)

def test_for_basil(filepath="basil.tif"):
    img = tifffile.imread(filepath)
    # ensure 1s for conductive phase and 0s otherwise.
    img[img == 255] = 1
    img[img == 2] = 1
    # s = tau.Solver(img)
    # # call solver 
    # s.solve(iter_limit=10000)
    img = img.transpose(1,0,2)
    s = tau.Solver(img)
    # call solver 
    s.solve(verbose = 'per_iter')

if __name__ == "__main__":
    test_for_basil()  