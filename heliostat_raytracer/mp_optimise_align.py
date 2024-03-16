import numpy as np

from apparent_inc_optimiser import get_optimized_deltas
from vector import vector_from_azimuth_elevation

if __name__ == "__main__":
    print(get_optimized_deltas(azim=0, elev=30))