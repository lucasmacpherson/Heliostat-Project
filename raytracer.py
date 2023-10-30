import numpy as np

def vector_to_receiver(hstat, recevier):
    """
    Calculate the normalised direction vector pointing torwards receiver
    """
    return np.array(norm_vector(recevier - hstat))

def calculate_mirror_normal(receiver_vec, incident_vec):
    """
    Calculate the mirror normal vector required for reflection of incident rays to the target
    """
    xproduct = np.cross(-incident_vec, receiver_vec)
    theta = np.arcsin(magnitude(xproduct))
    if magnitude(xproduct != 0):
        R = calculate_rotation_matrix(xproduct/magnitude(xproduct), theta/2)
    else:
        R = calculate_rotation_matrix(xproduct, theta/2)
        
    if theta != 0:
        return np.array(np.matmul(R, -incident_vec))
    else:
        return -incident_vec

def calculate_reflection(mirror_norm, incident_vec):
    """
    Calculate the normalised direction vector of the reflected ray
    """
    return norm_vector(np.array(incident_vec - 2*mirror_norm.dot(incident_vec.dot(mirror_norm))))

def calculate_mirror_positions(hstat, mirror_normal, receiver_vec, reflecting_width):
    """
    Calculate the positions of the 2 heliostat mirrors, seperated from the central
    position by a distance of reflecting_width/4
    """
    step = reflecting_width/4
    xproduct = norm_vector(np.cross(mirror_normal, receiver_vec))
    return np.array([hstat - step*xproduct, hstat + step*xproduct])

def calculate_target_intersection(reflected_vec, hstat, receiver_pos):
    """
    Calculate the intersection point of the reflected ray and the target plane
    """
    plane_normal = np.array([0, 0, 1])
    plane_offset = receiver_pos[2]

    return calculate_plane_intersection(reflected_vec, hstat, plane_normal, plane_offset)

def tilt_mirror_normal(mirror_position, receiver_pos, mirror_norm, incident_vec):
    receiver_vec = vector_to_receiver(mirror_position, receiver_pos)
    initial_reflect = calculate_reflection(mirror_norm, incident_vec)
    xproduct = np.cross(initial_reflect, receiver_vec)
    print(magnitude(xproduct))
    tilt =  np.arcsin(magnitude(xproduct)) / 2
    
    R = calculate_rotation_matrix(xproduct/magnitude(xproduct), tilt)
    return np.matmul(R, mirror_norm)

# Generalised vector methods

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