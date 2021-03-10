This documentation covers more advanced usage of TauFactor

## Metrics

Metrics can be calculated using the metrics module
```python
from taufactor.metrics import *
```

### Volume fraction
Volume fraction is calculated for each phase in a segmented image:
```python
import taufactor as tau
from taufactor.metrics import volume_fraction 
# load image
img = tifffile.imread('path/filename')
# consider a two phase image with 1s and 0s
# here we perform an example segmentation on a grayscale img
img[img > 0.7] = 1
img[img != 1] = 0
# calculate the volume fraction
vf = volume_fraction(img)

# consider a three phase image with pore, particle and binder
# where 0, 1, 2 correspond to pore, particle and binder respectively
# calculate the volume fraction
vf = volume_fraction(img, phases={'pore':0, 'particle':1, 'binder':2})
```
