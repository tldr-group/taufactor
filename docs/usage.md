# Using Taufactor

This documentation covers the full usage of TauFactor

## Core solver

### Tau and D_eff

The main output of the solve function is `tau` and `D_eff`. `tau` is the tortuoisty factor, a measure of the reduction in diffusive transport caused by convolution in the geometry of the material. `D_eff` is the effective diffusivity resulting from the tortuous nature of the material. The relationship between these values is given by:

$D_{eff}=D\frac{\epsilon}{\tau}$

For more see [Cooper _et al._](https://doi.org/10.1016/j.softx.2016.09.002)

```
import taufactor as tau

# load segmented image
img = tifffile.imread('path/filename')
s = tau.Solver(img)
# tau
s.tau
# D_eff
s.D_eff
```

### Steady-state maps

Steady-state maps can be visualised after solving. These maps show the steady state solution to the flux field and concentration field after convergence.

```
import taufactor as tau

# load segmented image
img = tifffile.imread('path/filename')
s = tau.Solver(img)
# flux map
s.flux_map(lay=5, filename='example.png')
# concentration map
s.conc_map()
```

## Other Solvers

### Periodic solver

```
import taufactor as tau

# load 2 phase periodic segmented image
img = tifffile.imread('path/filename')

# create a Periodic solver object and set an iteration limit
# and convergence limit of 1%
s = tau.PeriodicSolver(img, iter_limit=1000, conv_limit = 0.01)
# call solve function
s.solve()
```

### Multi-phase solver

```
import taufactor as tau

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

```
import taufactor as tau

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

Volume fraction is calculated for each phase in a segmented image:

```python
from taufactor.metrics import surface_area
# calculate the volume fraction of a single phase in an img
vf = surface_area(img, phases=1)

# consider a three phase image with pore, particle and binder
# where 0, 1, 2 correspond to pore, particle and binder respectively
# calculate the volume fraction between pore and binder with periodic boundaries in y and z axes
vf = volume_fraction(img, phases=[0,2], periodic=[0,1,1])
```
