# Using Taufactor

This documentation covers the common usage of TauFactor. For a more detailed description see the [module API documentation](./taufactor.rst).

## Core solver

### Tau and D_eff

The main output of the solve function is `tau` and `D_eff`. `tau` is the tortuosity factor, a measure of the reduction in diffusive transport caused by convolution in the geometry of the material. `D_eff` is the effective diffusivity resulting from the tortuous nature of the material. The relationship between these values is given by:

{math}`D_{eff}=D\frac{\epsilon}{\tau}`

For more see [Cooper _et al._](https://doi.org/10.1016/j.softx.2016.09.002)

```python
import taufactor as tau
import tifffile

# load segmented image
img = tifffile.imread('path/filename')
s = tau.Solver(img)
s.solve()
# tau
s.tau
# D_eff
s.D_eff
```

The iteration limit, convergence criteria and verbosity of the solver can be adjusted. Setting `verbose='per_iter'` logs the output of the solver every 100 steps whilst solving. The `conv_crit` controls the value at which convergence is met.

```python
s.solve(verbose='per_iter', conv_crit='1e-5')
```

### Flux direction

The direction of flow by default is the first index of the loaded image. If a different direction is required, the image must be permuted before solving. To visualise this and give guidance, the utility function `flux_direction` can be used

```python
from taufactor.utils import flux_direction
import tifffile

img = tifffile.imread('path/filename')
flux_direction(img)
```

When the flux direction has been chosen, the image can be permuted using `torch.permute` as outlined in the output from `flux_direction`, for example

```python
import taufactor as tau
import tifffile

# load segmented image
img = tifffile.imread('path/filename')
flux_direction(img)
img = torch.permute(torch.tensor(img), (1,2,0))
s = tau.Solver(img)
s.solve()
```

## Other Solvers

### Periodic solver

The periodic solver applies periodic boundary conditions instead of mirror boundaries.

```python
import taufactor as tau
import tifffile

# load 2 phase periodic segmented image
img = tifffile.imread('path/filename')

# create a Periodic solver object and set an iteration limit
# and convergence limit of 1%
s = tau.PeriodicSolver(img, iter_limit=1000, conv_limit = 0.01)
# call solve function
s.solve()
```

### Multi-phase solver

The multi-phase solver allows for more than 2 conductive phases per image. The conductivity of each phase is given as an input to the solver along with the phase label

```python
import taufactor as tau
import tifffile

# load n phase segmented image
img = tifffile.imread('path/filename')
# assign conductivity values, where key is segmented label in 'img'
# and value is conductivity
cond = {1:0.32, 2:0.44}

# create a multiphase solver object and set an iteration limit
s = tau.MultiPhaseSolver(img, cond=cond, iter_limit=1000)
# call solve function
s.solve()
```

### Electrode tortuosity factor solver

This solver implements the electrode tortuosity method from [Nguyen _et al._](https://doi.org/10.1038/s41524-020-00386-4)

```python
import taufactor as tau
import tifffile

# load n phase segmented image
img = tifffile.imread('path/filename')

# create an electrode solver object and set an iteration limit
s = tau.ElectrodeSolver(img)
# call solve function
s.solve()
```

## Metrics

Metrics can be calculated using the metrics module

```python
from taufactor.metrics import *
```

### Volume fraction

Volume fraction is calculated for each phase in a segmented image:

```python
from taufactor.metrics import volume_fraction

# calculate the volume fraction
vf = volume_fraction(img)

# consider a three phase image with pore, particle and binder
# where 0, 1, 2 correspond to pore, particle and binder respectively
# calculate the volume fraction
vf = volume_fraction(img, phases={'pore':0, 'particle':1, 'binder':2})
```

### Surface area

Surface area is calculated for each phase in a segmented image. The surface area returned is the fraction of interfacial faces with respect to the total number of faces n the image

```python
from taufactor.metrics import surface_area
# calculate the surface area of a single phase in an img
sa = surface_area(img, phases=1)

# consider a three phase image with pore, particle and binder
# where 0, 1, 2 correspond to pore, particle and binder respectively
# calculate the surface area between pore and binder with periodic boundaries in y and z axes
sa = surface_area(img, phases=[0,2], periodic=[0,1,1])
```

### Triple phase boundary

Triple phase boundary is calculated on a segmented image with exactky three phases. The value returned is the fraction of triple phase edges with respect to the total number of edges

```python
from taufactor.metrics import triple_phase_boundary
# calculate the triple phase boundareies
tpb = triple_phase_boundary(img)
```
