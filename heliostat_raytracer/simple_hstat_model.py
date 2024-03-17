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

def calculate_incident_angles(hstats, receiver_pos, elevations, azimuths):
    angles = np.zeros(shape=(len(elevations), len(azimuths)))
    for i, elevation in enumerate(elevations):
        for j, azimuth in enumerate(azimuths):
            incident_vec = -1*vector_from_azimuth_elevation(azimuth, elevation)
            model = create_simplified_model(hstats, incident_vec, receiver_pos)
            angles[i, j] = 180/np.pi * model['incident_angle']

    return angles

def mphelper_simple_eff(hstats, elevation, azimuth, receiver_pos, receiver_size,
                          mirror_size, source_dist, system_extent, raycasts,
                          fname=''):
        incident_vec = -1*vector_from_azimuth_elevation(azimuth, elevation)
        model = create_simplified_model(hstats, incident_vec, receiver_pos)
        model = create_geometry(model, receiver_size, mirror_size)
        print(f"Raytracing system with elev: {elevation}, azim: {azimuth} ({raycasts[0]*raycasts[1]} rays)...")
        model = raytrace_source_incidence(model, source_dist, incident_vec, system_extent, raycasts)
        collection_frac = calculate_collection_fraction(model)
        
        if fname != '':
            img = intensity_image(model, exp.CAMERA_IMAGESIZE.value, sigma=4)
            img.save(fname)

        fig, ax = show_system(model)
        plt.show()

        return collection_frac

def realscale_simple_effwrapper(azimuth, elevation):
    receiver_pos = np.array(exp.RECEIVER_POSITION.value)
    receiver_size = exp.RECEIVER_SIZE.value
    # receiver_size = (1, 1)
    mirror_size = exp.MIRROR_RADIUS.value
    hstats = [[0.226, -0.226, 0], [0.226, 0.226, 0]]
    # hstats = [[0.226, 0, 0]]

    raycasts = (100, 2000)
    system_extent = np.array([
        np.array((0.3, 0.5, -0.3)),
        np.array((-0.2, -0.5, 0.2))
    ])

    return mphelper_simple_eff(hstats, elevation, azimuth, receiver_pos, 
                               receiver_size, mirror_size, exp.SOURCE_DISTANCE.value, 
                               system_extent, raycasts)

def datagen_angle_range(elevations, azimuths, worker_threads=8):
    eff = np.zeros(shape=(len(elevations), len(azimuths)))

    for i, elev in enumerate(elevations):
        args = []

        for j, azim in enumerate(azimuths):
            args.append([azim, elev])

        with mp.Pool(worker_threads) as pool:
            results = pool.starmap(realscale_simple_effwrapper, args)

        for j, result in enumerate(results):
            eff[i, j] = result

    return eff


def single_elev_efficiency_plot(elevations, azimuths, eff):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, elev in enumerate(elevations):
        ax.scatter(azimuths, eff[i, :], label=f"Beta={elev}")

    return fig, ax

if __name__ == "__main__":
    elevations = np.array((15, 30, 45, 55, 60))
    azimuths = np.array((-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70))

    # eff = datagen_angle_range(elevations, azimuths, worker_threads=12)
    # print(eff)
    # fig, ax = single_elev_efficiency_plot(elevations, azimuths, eff)
    # plt.show()

    print(realscale_simple_effwrapper(azimuth=30, elevation=30))

    # angles = calculate_incident_angles(elevations, azimuths)
    # np.savetxt('data/simple_incident_angles.csv', angles, delimiter=',')
