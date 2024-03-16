import numpy as np

from apparent_inc_optimiser import get_optimized_vec
from vector import vector_from_azimuth_elevation

if __name__ == "__main__":
    incident_vec = -1*vector_from_azimuth_elevation(alpha=0, beta=30)
    print(get_optimized_vec(incident_vec))