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

def distance(vec1: np.ndarray, vec2: np.ndarray):
    return magnitude(vec2 - vec1)

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
    if s.all() == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def vector_plane_distance(vector_norm: np.ndarray, vector_position: np.ndarray, plane_norm: np.ndarray, plane_offset):
    s = vector_norm.dot(plane_norm)
    if s == 0: 
        return None
    
    u = plane_offset - vector_position
    return u.dot(plane_norm) / s

def calculate_plane_intersection(vector_norm: np.ndarray, vector_position: np.ndarray, plane_norm: np.ndarray, plane_offset, epsilon=1e-9):
    d = vector_plane_distance(vector_norm, vector_position, plane_norm, plane_offset)
    if d is None or abs(d) < epsilon:
        return None
    
    return np.array([vector_position[0] + vector_norm[0]*d, 
                     vector_position[1] + vector_norm[1]*d,
                     vector_position[2] + vector_norm[2]*d])

def vector_from_elevation_azimuth(alpha, beta):
    # Convert degree to radians
    alpha, beta = alpha*np.pi/180, beta*np.pi/180

    f = np.array((1, 0, 0))

    # rotate f by alpha about the normal to the plane
    plane_normal = np.array((0, 0, 1))
    R = calculate_rotation_matrix(plane_normal, alpha)
    f = np.matmul(R, f)

    # elevate f by beta
    u = np.cross(f, plane_normal)
    R = calculate_rotation_matrix(u, beta)
    f = np.matmul(R, f)

    return f