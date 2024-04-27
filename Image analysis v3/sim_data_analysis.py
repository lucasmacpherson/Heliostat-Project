import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from edge_detector_v3 import *

def plot_collections(data, tilts, azimuthals, colours):

    colls = data["collection_fractions"]

    for i, tilt in enumerate(tilts):
        for azim in azimuthals:
            frac = colls[(tilt, azim)] * 100
            plt.scatter(azim, frac, label = str(tilt), color = colours[i])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Collection fractions")
    plt.xlabel("Azimuthal tilt")
    plt.ylabel("Collection % (a.u.)")

    plt.show()

def plot_heliostat_thetas(data, tilt, azimuthals, colours):

    all_thetas = data["heliostat_thetas"]

    neg = azimuthals[::-1][:-1]*-1
    mirrd_azimuthals = np.concatenate((neg, azimuthals))

    key = [2, 3, 0, 1]

    for azim in mirrd_azimuthals:
        h_thetas = all_thetas[(tilt, np.absolute(azim))]/2

        for helio, theta in enumerate(h_thetas):

            if azim < 0:
                reflected_helio = key[helio]
                plt.scatter(azim, theta, label = ("Heliostat " + str(reflected_helio +1)), color = colours[reflected_helio])

            else:
                plt.scatter(azim, theta, label = ("Heliostat " + str(helio +1)), color = colours[helio])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Azimuthal angle")
    plt.ylabel("Theta")
    plt.title(f"Elevational tilt {str(tilt)}")

    plt.show()

def plot_all_heliostat_thetas(data, tilts, azimuthals, colours):

    key = [2, 3, 0, 1]

    neg = azimuthals[::-1][:-1]*-1
    mirrd_azimuthals = np.concatenate((neg, azimuthals))

    all_thetas = data["heliostat_thetas"]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    for tilt, ax in zip(tilts, axs.ravel()):
        by_helio = [[],[],[],[]]
        for azim in mirrd_azimuthals:
            h_thetas = all_thetas[(tilt, np.absolute(azim))]/2

            for helio, theta in enumerate(h_thetas):
                if azim < 0:
                    reflected_helio = key[helio]
                    by_helio[reflected_helio].append(theta)
                else:
                    by_helio[helio].append(theta)

                # ax.scatter(azim, theta, label = ("Heliostat " + str(helio +1)), color = colours[helio], marker = ".")
        
        for i, angles in enumerate(by_helio):
            # angles = np.array(angles)
            # opp_angles = angles[::-1][:-1]
            # all_angles = np.concatenate((opp_angles, angles))
            ax.plot(mirrd_azimuthals, angles, label = ("Heliostat " + str(i +1)), color = colours[i])
            # ax.set_xlabel("Azimuthal angle")
            # ax.set_ylabel("Theta")
            ax.set_title((f"{str(tilt)}$^\circ$ Elevation"))


    fig.supxlabel("Azimuth ($^\circ$)", fontsize = 12)
    fig.supylabel("Theta ($^\circ$)", fontsize = 12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1))

    plt.tight_layout()

    #plt.savefig("Image analysis v3/report graphs/theta for all heliostats and tilt angles.png", dpi = 1500)
    plt.show()

def plot_mirror_thetas(data, tilt, azimuthals, colours):

    all_thetas = data["heliostat_thetas"]
    mirr_thetas = data["mirror_thetas"]

    neg = azimuthals[::-1][:-1]*-1
    mirrd_azimuthals = np.concatenate((neg, azimuthals))

    by_helio = [[],[],[],[]]

    markers = ["x", "^"]

    for azim in mirrd_azimuthals:
        h_thetas = all_thetas[(tilt, np.absolute(azim))]/2
        m_thetas = mirr_thetas[(tilt, np.absolute(azim))]/2

        print(m_thetas)
        print(h_thetas)
        print()

        for j, theta in enumerate(m_thetas):
            index = int(j//2)
            plt.scatter(azim, theta, marker = markers[j%2], label = ("Heliostat " + str(index +1)), color = colours[index])

        for helio, theta in enumerate(h_thetas):
            by_helio[helio].append(theta)

    for i, angles in enumerate(by_helio):
        plt.plot(mirrd_azimuthals, angles, label = ("Heliostat " + str(i +1)), color = colours[i])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Azimuthal angle")
    plt.ylabel("Theta")
    plt.title(f"Elevational tilt {str(tilt)}")

    plt.show()

def compare_collections(datas, tilts, azimuthals, colours, names, symbols):

    for k, data in enumerate(datas):
        colls = data["collection_fractions"]
        name = names[k]
        symbol = symbols[k]
        print(symbol, name)

        for i, tilt in enumerate(tilts):
            for azim in azimuthals:
                frac = colls[(tilt, azim)] * 100
                plt.scatter(azim, frac, label = (name + " " + str(tilt)), color = colours[i], marker = symbol)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Collection fractions")
    plt.xlabel("Azimuthal tilt")
    plt.ylabel("Collection % (a.u.)")
    
    plt.show()
    
def collections_and_cosine(datas, tilts, azimuthals, colours, names, symbols):

    for k, data in enumerate(datas):

        if k == 0:
            thetas = data["heliostat_thetas"]

        colls = data["collection_fractions"]
        name = names[k]
        symbol = symbols[k]
        print(symbol, name)

        for i, tilt in enumerate(tilts):
            cos_vals = []

            if k == 0:
                coll_norm = colls[(tilt, 0)]
                ang_norm = np.mean(np.cos(thetas[(tilt, 0)]*np.pi/360))
                norm = (coll_norm/ang_norm) * 100

            for azim in azimuthals:
                frac = colls[(tilt, azim)] * 100
                plt.scatter(azim, frac, label = (name + " " + str(tilt)), color = colours[i], marker = symbol)

                if azim != 0:
                    plt.scatter(azim*-1, frac, label = (name + " " + str(tilt)), color = colours[i], marker = symbol)

                angle = thetas[(tilt, azim)] *np.pi/360
                cos = np.cos(angle)
                mean_cos = np.mean(cos) * norm
                
                cos_vals.append(mean_cos)
            
            print(cos_vals)
            plt.plot(azimuthals, cos_vals, label = (name + " " + str(tilt)), color = colours[i])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())
    plt.title("Collection fractions")
    plt.xlabel("Azimuthal tilt")
    plt.ylabel("Collection % (a.u.)")

    plt.show()

def sim_binary(image):

    dim = np.shape(image)
    binary = image.copy()

    for x in range(0, dim[0]):
            for y in range(0, dim[1]):
                if image[x,y] > 0:
                    binary[x,y] = True
                else: binary[x,y] = False

    return binary


def analyse_sim_images(folder_name, tilts, sim_data):
    colours = ["red", "gold", "darkgreen", "blue"]
    fig = plt.figure(figsize = (7,5))

    #azimuthals = np.array([-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70])
    azimuthals = np.array([0, 15, 30, 45, 60, 70])

    colls = sim_data["collection_fractions"]
    
    plt.figure(figsize = (6,4))
    for i, t in enumerate(tilts):
        intensities = []
        sim_intensities = []

        for a in azimuthals:
            image_name = "Image analysis v3/simulation pictures/" + folder_name + f"/{str(t)}_{str(np.abs(a))}_intensity.png"
            image = im.imread(image_name)
            binary = sim_binary(image)

            #plt.imshow(binary)
            #plt.show()

            dim = np.shape(image)
            t_loc = [dim[0]//2, dim[1]//2]
            #print(t_loc)
            total_intensity, pixels_illuminated, separation, variance = image_data_extractor(binary, image, t_loc)

            #print(total_intensity)
            intensities.append(total_intensity)

            sim_int = colls[(t, a)]
            sim_intensities.append(sim_int)
        
        sim_intensities = np.array(sim_intensities)
        intensities = np.array(intensities) * (sim_intensities[0]/intensities[0])
        maximum = np.max(intensities)
        intensities *= (1/maximum) /1.05
        sim_intensities *= (1/maximum) /1.05
            
        # print(azimuthals)
        # print(intensities)
        plt.scatter(azimuthals, intensities, label = f"Image analysis {str(t)}$^\circ$", color = colours[i], marker = "^")
        plt.plot(azimuthals, sim_intensities, label = f"Collection fraction {str(t)}$^\circ$", color = colours[i], marker = ".")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Azimuthals ($^\circ$)", fontsize = 12)
    plt.ylabel("Incident intensity (a.u.)", fontsize = 12)
    plt.tight_layout()
    #plt.savefig("Image analysis v3/report graphs/Simulation collection fraction vs. image analysis.png", dpi = 1500)
    plt.show()
    #return total_intensity, pixels_illuminated, separation, variance

if __name__ == "__main__":
    file = "Image analysis v3/sim data/exprange_idealtilt_25Mrays_last.pkl"

    f  = open(file, "rb")
    data = pkl.load(f)

    tilts = np.arange(15, 75, 15)
    azimuthals = np.array([0, 15, 30, 45, 60, 70])
    colours = ["red", "gold", "darkgreen", "blue"]

    f2 = open("Image analysis v3/sim data/exprange_10degtilt_25Mrays_last.pkl", "rb")
    non_ideal_data = pkl.load(f2)

    #plot_collections(data, tilts, azimuthals, colours)
    #plot_heliostat_thetas(data, 15, azimuthals, colours)
    plot_all_heliostat_thetas(data, tilts, azimuthals, colours)
    #plot_mirror_thetas(data, 60, azimuthals, colours)

    #compare_collections([data, non_ideal_data], [15], azimuthals, colours, names = ["Ideal", "Non-ideal"], symbols = ["o", "x"])
    #collections_and_cosine([data, non_ideal_data], [30], azimuthals, colours, names = ["Ideal", "Non-ideal"], symbols = ["o", "x"])

    #tilts = [15]
    #analyse_sim_images("2hst_exprange_10degtilt_25Mrays", tilts, data)