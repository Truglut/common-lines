import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
RotationTriplet = tuple[FloatArray, FloatArray, FloatArray]
