import numpy as np

from vector import *

def vector_to_receiver(hstat, recevier) -> np.ndarray:
    """
    Calculate the normalised direction vector pointing torwards receiver
    """
    return np.array(norm_vector(recevier - hstat))

def calculate_mirror_normal(receiver_vec, incident_vec):
    """
    Calculate the mirror normal vector required for reflection of incident rays to the target
    """
    dproduct = np.dot(-incident_vec, receiver_vec)
    xproduct = np.cross(-incident_vec, receiver_vec)
    theta = np.arccos(dproduct)
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

def calculate_mirror_positions(hstat, mirror_normal, receiver_vec, mirror_sep):
    """
    Calculate the positions of the 2 heliostat mirrors, seperated from the central
    position by a distance of mirror_sep/2.

    Function returns a tuple of (mirror_positions, offset_vectors)
    """
    step = mirror_sep/2
    xproduct = norm_vector(np.cross(mirror_normal, receiver_vec))
    return (np.array([hstat - step*xproduct, hstat + step*xproduct]),
            np.array([-xproduct, xproduct]))

def calculate_target_intersection(reflected_vec, mirror, receiver_pos):
    """
    Calculate the intersection point of the reflected ray and the target plane
    """
    plane_normal = np.array([0, 0, 1])
    plane_offset = receiver_pos[2]

    return calculate_plane_intersection(reflected_vec, mirror, plane_normal, plane_offset)

def calculate_ideal_tilt(mirror_position, receiver_pos, mirror_norm, incident_vec):
    receiver_vec = vector_to_receiver(mirror_position, receiver_pos)
    initial_reflect = calculate_reflection(mirror_norm, incident_vec)
    dproduct = np.dot(initial_reflect, receiver_vec)
    tilt = -np.arccos(dproduct) / 2
    
    return tilt

def tilt_mirror_normal(mirror_normal, offset_vec, tilt):
    xproduct = norm_vector(np.cross(mirror_normal, offset_vec))
    R = calculate_rotation_matrix(xproduct, tilt)
    return np.matmul(R, mirror_normal)