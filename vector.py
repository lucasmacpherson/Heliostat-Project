import numpy as np

def magnitude(vector: np.ndarray):
    """ 
    Faster method for getting the magnitude of numpy vectors
    """
    return np.sqrt(vector.dot(vector))

def norm_vector(vector: np.ndarray):
    """
    Get the normalised vector of the direction of any given numpy vector
    """
    return vector / np.sqrt(vector.dot(vector))

def calculate_rotation_matrix(u: np.ndarray, theta):
    # Source: https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    M = [ [ctheta+u[0]*u[0]*(1-ctheta), u[0]*u[1]*(1-ctheta) - u[2]*stheta, u[0]*u[2]*(1-ctheta) + u[1]*stheta],
          [u[1]*u[0]*(1-ctheta) + u[2]*stheta, ctheta + u[1]*u[1]*(1-ctheta), u[1]*u[2]*(1-ctheta) - u[0]*stheta],
          [u[2]*u[0]*(1-ctheta) - u[1]*stheta, u[2]*u[1]*(1-ctheta) + u[0]*stheta, ctheta + u[2]*u[2]*(1-ctheta)]
    ]

    return M
    
def calculate_plane_intersection(vector_norm: np.ndarray, vector_position: np.ndarray, plane_norm: np.ndarray, plane_offset):
    # Source: https://stackoverflow.com/questions/26920705/ray-plane-intersection
    s = plane_norm.dot(vector_norm)
    if s == 0: 
        return None
    
    t = (plane_offset-(plane_norm.dot(vector_position)))/s
    return np.array([vector_position[0] + vector_norm[0]*t, 
                     vector_position[1] + vector_norm[1]*t,
                     vector_position[2] + vector_norm[2]*t])