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

def rotation_matrix_from_vectors(vec1, vec2):
    # Source: https://stackoverflow.com/a/59204638
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
    
def calculate_plane_intersection(vector_norm: np.ndarray, vector_position: np.ndarray, plane_norm: np.ndarray, plane_offset):
    # Source: https://stackoverflow.com/questions/26920705/ray-plane-intersection
    s = plane_norm.dot(vector_norm)
    if s == 0: 
        return None
    
    t = (plane_offset-(plane_norm.dot(vector_position)))/s
    return np.array([vector_position[0] + vector_norm[0]*t, 
                     vector_position[1] + vector_norm[1]*t,
                     vector_position[2] + vector_norm[2]*t])

def line_plane_intersection(point, vector, plane_point, plane_normal):
    vector_norm = norm_vector(vector)
    dist = np.dot(plane_point - point, plane_normal) / np.dot(vector_norm, plane_normal)
    intersection_point = point + dist * vector_norm

    return intersection_point