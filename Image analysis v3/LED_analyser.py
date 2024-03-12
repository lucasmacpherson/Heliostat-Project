from edge_detector_v3 import *
import matplotlib.cm as cm


def plot_by_tilt(tilts, azimuthals, folder, iterations, target_loc, colours):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i, tilt in enumerate(tilts):

        col = colours[i]

        for j, azim in enumerate(azimuthals):

            for iteration in range(0, iterations):

                filename = folder + str(tilt) + "_" + str(azim) + "_" + str(iteration) + ".png"
                backname = folder + str(tilt) + "_" + str(azim) + "_back" + str(iteration) + ".png"

                print(filename)
                image = im.imread(filename)
                background = im.imread(backname)

                image = np.array(image, dtype=np.int16)
                background = np.array(background, dtype=np.int16) #is this necessary? check

                removed = np.array(image) - np.array(background)

                binary = gen_binary(removed, threshold=10)
                plt.imsave((folder + "/binary images/" + filename[len(folder):]), binary, cmap=cm.gray)

                total_intensity, pixels_illuminated, separation, variance = image_data_extractor(binary, removed, target_loc)

                ax1.scatter(azim, total_intensity, color = col, label = str(tilt))
                ax2.scatter(azim, pixels_illuminated, color = col, label = str(tilt))

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
                ax2.legend(by_label.values(), by_label.keys())

                ax1.set_title("Total Intensity")
                ax2.set_title("Pixels illuminated")


    plt.show()

def plot_by_object_num(tilt, azimuthals, folder: str, mirror_numbers, iterations: int, target_loc, colours, normalise = False, average = False):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for j, azim in enumerate(azimuthals):
        if average:
            intensities = []
            pixels = []

            if not normalise: print("Careful: Averaging without normalising")

        for iteration in range(0, iterations):

            filename = folder + str(tilt) + "_" + str(azim) + "_" + str(iteration) + ".png"
            backname = folder + str(tilt) + "_" + str(azim) + "_back" + str(iteration) + ".png"

            print(filename)
            image = im.imread(filename)
            background = im.imread(backname)

            image = np.array(image, dtype=np.int16)
            background = np.array(background, dtype=np.int16) #is this necessary? check

            removed = np.array(image) - np.array(background)

            binary = gen_binary(removed, threshold=10)

            # im.imsave((folder + "/binary images/" + filename[len(folder):]), binary.astype(float))
            plt.imsave((folder + "/binary images/" + filename[len(folder):]), binary, cmap=cm.gray)

            row = j*3 + iteration
            num_objs = int(mirror_numbers[row])

            col = colours[num_objs-1]

            total_intensity, pixels_illuminated, separation, variance = image_data_extractor(binary, removed, target_loc)

            if normalise:
                if num_objs != 4:
                    total_intensity = total_intensity * 4/(num_objs)
                    pixels_illuminated = pixels_illuminated * 4/(num_objs)

            if not average:
                ax1.scatter(azim, total_intensity, color = col, label = str(num_objs))
                ax2.scatter(azim, pixels_illuminated, color = col, label = str(num_objs))

            elif average:
                intensities.append(total_intensity)
                pixels.append(pixels_illuminated)

            #print(pixels_illuminated)

        if average:
            ax1.scatter(azim, np.mean(intensities), color = col, label = str(num_objs))
            ax2.scatter(azim, np.mean(pixels), color = col, label = str(num_objs))

        #print()
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax2.legend(by_label.values(), by_label.keys())

    ax1.set_title("Total Intensity")
    ax2.set_title("Pixels illuminated")
    plt.show()

def sim_plot_by_tilt(tilts, azimuthals, folder, target_loc, colours):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i, tilt in enumerate(tilts):

        col = colours[i]

        for j, azim in enumerate(azimuthals):

            filename = folder + str(tilt) + "_" + str(azim) + "_25Mrays_intensity.png"

            print(filename)
            image = im.imread(filename)
            image = np.array(image, dtype=np.int16)

            binary = gen_binary(image, threshold=10)

            total_intensity, pixels_illuminated, separation, variance = image_data_extractor(binary, image, target_loc)

            ax1.scatter(azim, total_intensity, color = col, label = str(tilt))
            ax2.scatter(azim, pixels_illuminated, color = col, label = str(tilt))

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys())
            ax2.legend(by_label.values(), by_label.keys())

            ax1.set_title("Total Intensity")
            ax2.set_title("Pixels illuminated")


    plt.show()

tilts = np.arange(15, 75, 15)
azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]
colours = ["red", "orange", "green", "blue"]


#for full data set
folder = "Image analysis v3/full data set/"
mirr_15, mirr_30, mirr_45, mirr_60 = np.loadtxt((folder + "mirror_numbers.csv"), skiprows=1, delimiter = ",", unpack = True, usecols=range(1,5))

plot_by_tilt(tilts, azimuthals, folder, 3, [640, 544], colours)

average = False
normalise = True

# plot_by_object_num(15, azimuthals, folder, mirr_15, 3, [640, 544], colours, normalise, average)
# plot_by_object_num(30, azimuthals, folder, mirr_30, 3, [640, 544], colours, normalise, average)
# plot_by_object_num(45, azimuthals, folder, mirr_45, 3, [640, 544], colours, normalise, average)
# plot_by_object_num(60, azimuthals, folder, mirr_60, 3, [640, 544], colours, normalise, average)

#for simulation data
#folder = "Image analysis v3/simages/"
#sim_plot_by_tilt(tilts, azimuthals, folder, [512, 640], colours)



# for finding target
# image = im.imread("Image analysis v3/full data set/15_-30_back1.png")
# im.imshow(image)
# plt.show()