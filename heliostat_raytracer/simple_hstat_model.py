from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np

from heliostat_field import *
from plotting import *
from images import *

from experimental_params import experimental_params as exp

def calculate_3D_angle(receiver_vec: np.ndarray, incident_vec: np.ndarray):
    return np.arccos(receiver_vec.dot(incident_vec))

def create_simplified_model(hstats: list, incident_vec: np.ndarray, receiver_pos: np.ndarray):
    mirror_norms = []
    reflected_vecs = []
    mirror_positions = []
    incident_angle = 0

    for i, hstat in enumerate(hstats):
        receiver_vec = vector_to_receiver(hstat, receiver_pos)
        mirror_norm = calculate_mirror_normal(receiver_vec, incident_vec)
        
        if (len(hstats) == 1):
            incident_angle = calculate_3D_angle(receiver_vec, incident_vec)

        mirror_positions.append(hstat)
        mirror_norms.append(mirror_norm)
        reflected_vecs.append(calculate_reflection(mirror_norm, incident_vec))

    return {'heliostat_positions': hstats,
        'receiver_position': receiver_pos,
        'incident_vector': incident_vec,
        'mirror_normals': np.array(mirror_norms),
        'reflected_vectors': np.array(reflected_vecs),
        'mirror_positions': np.array(mirror_positions),
        'incident_angle': incident_angle
    }

def efficiency_plots(eff):
    elev0 = 45   # 45 degrees
    azim0 = 0   # 0 degrees
    
    elevs = []
    elev_effs = []
    azims = []
    azim_effs = []
    for key in eff:
        elev, azim = key
        eff_value = eff[key]

        if elev == elev0:
            azims.append(azim)
            azim_effs.append(eff_value)

        if azim == azim0:
            elevs.append(elev)
            elev_effs.append(eff_value)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(elevs, elev_effs)
    plt.show()

def mphelper_simple_image(hstats, elevation, azimuth, receiver_pos, receiver_size,
                          mirror_size, beam_size, raycasts, start_height,
                          fname=''):
        incident_vec = vector_from_azimuth_elevation(azimuth, elevation)
        model = create_simplified_model(hstats, incident_vec, receiver_pos)
        model = create_geometry(model, receiver_size, mirror_size)
        model = raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height)
        collection_frac = calculate_collection_fraction(model)
        
        if fname != '':
            img = intensity_image(model, exp.CAMERA_IMAGESIZE.value, sigma=4)
            img.save(fname)
        
        return collection_frac
    

if __name__ == "__main__":
    receiver_pos = np.array(exp.RECEIVER_POSITION.value)
    receiver_size = exp.RECEIVER_SIZE.value
    mirror_size = exp.MIRROR_RADIUS.value
    # hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
    hstats = [[0.226, 0, 0]]

    raycasts = 500**2
    beam_size = 3.0
    start_height = 0.20

    elevations = np.array((15, 30, 45, 55, 60))
    azimuths = np.array((-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70))

    angles = np.zeros(shape=(len(elevations), len(azimuths)))
    for i, elevation in enumerate(elevations):
        for j, azimuth in enumerate(azimuths):
            incident_vec = vector_from_azimuth_elevation(azimuth, elevation)
            model = create_simplified_model(hstats, incident_vec, receiver_pos)
            angles[i, j] = 180/np.pi * model['incident_angle']

    print(angles)
    np.savetxt('data/simple_incident_angles.csv', angles, delimiter=',')

    model = create_simplified_model(hstats, incident_vec, receiver_pos)
    model = create_geometry(model, receiver_size, mirror_size)
    model = raytrace_uniform_incidence(model, incident_vec, beam_size, raycasts, start_height)

    fig, ax = show_system(model)