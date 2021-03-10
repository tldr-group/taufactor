This documentation covers more advanced usage of TauFactor

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
