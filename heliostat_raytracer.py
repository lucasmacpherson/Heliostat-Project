import numpy as np

def create_heliostat_field(size, layout):
    """
    Generate the heliostat field as an array of vector positions, based on the provided
    layout. Scale provided is the width of the rectangular array.
    """
    heliostats = []
    try:
        xstep = size / (layout[1] - 1)
    except ZeroDivisionError: xstep = 0
    try:
        ystep = size / (layout[0] - 1)
    except ZeroDivisionError: ystep = 0

    for i in range(layout[0]):
        for j in range(layout[1]):
            xpos = xstep * j - size/2
            ypos = ystep * i - size/2
            zpos = 0
            heliostats.append(np.array([xpos, ypos, zpos]))
    
    heliostats = np.array(heliostats)
    return heliostats

def get_directions_to_receiver(heliostats, receiver):
    receiver_vecs = []
    for hstat in heliostats:
        receiver_vecs.append(np.array(norm_vector(receiver - hstat)))

    return np.array(receiver_vecs)

def calculate_mirror_normals(receiver_vecs, incident_vec):
    mirror_norms = []
    for vector in receiver_vecs:
        xproduct = np.cross(-incident_vec, vector) # Note: assuming incident_vec normalised
        theta = np.arcsin(magnitude(xproduct))
        if magnitude(xproduct != 0):
            R = calculate_rotation_matrix(xproduct/magnitude(xproduct), theta/2)
        else:
            R = calculate_rotation_matrix(xproduct, theta/2)
        
        if theta != 0:
            mirror_norms.append(np.array(np.matmul(R, -incident_vec)))
        else:
            mirror_norms.append(-incident_vec)

    return np.array(mirror_norms)

def calculate_reflected_rays(mirror_norms, incident_vec, num_mirrors=1):
    reflected_vecs = []
    for normal in mirror_norms:
        for n in range(num_mirrors):
            reflected_vecs.append(
                norm_vector(np.array(incident_vec - 2*normal.dot(incident_vec.dot(normal))))
            )
    
    return np.array(reflected_vecs)

def calculate_mirror_positions(heliostats, mirror_normals, receiver_vecs, reflecting_width):
    mirror_positions = []
    step = reflecting_width/4
    for i, hstat in enumerate(heliostats):
        xproduct = norm_vector(np.cross(mirror_normals[i], receiver_vecs[i]))
        mirror_positions.append(hstat - step*xproduct)
        mirror_positions.append(hstat + step*xproduct)

    return np.array(mirror_positions)

def calculate_target_intersections(reflected_vecs, heliostats, receiver_pos):
    plane_normal = np.array([0, 0, 1])
    plane_offset = receiver_pos[2]

    intersections = []
    for i, hstat in enumerate(heliostats):
        intersections.append(calculate_plane_intersection(reflected_vecs[i], hstat, plane_normal, plane_offset))

    return np.array(intersections)

def calculate_idealised_tilts(mirror_positions, mirror_normals, target_pos,:
    alphas = []


    return np.array(alphas)

def magnitude(vector: np.ndarray):
    """ 
    Faster method for getting the magnitude of numpy vectors.
    """
    return np.sqrt(vector.dot(vector))

def norm_vector(vector: np.ndarray):
    """
    Get the normalised vector of the direction of any given numpy vector.   
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