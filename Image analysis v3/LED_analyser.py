from edge_detector_v3 import *
import matplotlib.cm as cm
import pickle as pkl
import seaborn as sns


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
                #plt.imsave((folder + "/binary images/" + filename[len(folder):]), binary, cmap=cm.gray)

                total_intensity, pixels_illuminated, separation, variance = image_data_extractor(binary, removed, target_loc)

                ax1.scatter(azim, total_intensity, color = col, label = str(tilt))
                ax2.scatter(azim, pixels_illuminated, color = col, label = str(tilt))

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
                ax2.legend(by_label.values(), by_label.keys())

                ax1.set_title("Total Intensity")
                ax2.set_title("Pixels illuminated")

def plot_by_object_num(tilt, azimuthals, folder: str, mirror_numbers, iterations: int, target_loc, colours, normalise = False, average = False, total_mirrs = 4):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ints = []
    areas = []
    seps = []
    vars = []
    azims = []
    normalisation = []

    for j, azim in enumerate(azimuthals):

        if average:
            intensities = []
            pixels = []

            if not normalise: print("Careful: Averaging without normalising")

        for iteration in range(0, iterations):
            
            filename = folder + str(tilt) + "_" + str(azim) + "_" + str(iteration) + ".png"
            backname = folder + str(tilt) + "_" + str(azim) + "_back" + str(iteration) + ".png"

            #print(filename)
            image = im.imread(filename)
            background = im.imread(backname)

            image = np.array(image, dtype=np.int16)
            background = np.array(background, dtype=np.int16) #is this necessary? check

            removed = np.array(image) - np.array(background)

            binary = gen_binary(removed, threshold=10)

            # im.imsave((folder + "/binary images/" + filename[len(folder):]), binary.astype(float))
            plt.imsave((folder + "/binary images/" + filename[len(folder):]), binary, cmap=cm.gray)

            row = j*iterations + iteration
            num_objs = int(mirror_numbers[row])

            if num_objs > 4:
                vals = num_objs - 4
            else: vals = num_objs
    
            col = colours[vals - 1]

            total_intensity, pixels_illuminated, separation, variance = image_data_extractor(binary, removed, target_loc)
            seps.append(separation)
            vars.append(variance)
            azims.append(azim)

            if azim == 0:
                normalisation.append(total_intensity)

            if normalise:
                if num_objs != total_mirrs:
                    total_intensity = total_intensity * total_mirrs/(num_objs)
                    pixels_illuminated = pixels_illuminated * total_mirrs/(num_objs)

            ints.append(total_intensity)
            areas.append(pixels_illuminated)

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

    data = np.array([azims, ints, areas, seps, vars])
    if normalise:
        np.savetxt(("Image analysis v3/"+ str(tilt) + " " + str(total_mirrs) + " nnorm data.csv"), data.T, delimiter=",")

    # else:
    #     np.savetxt(("Image analysis v3/"+ str(tilt) + " " + str(total_mirrs) + " data.csv"), data.T, delimiter=",")
    # #plt.show()
    
    # print(normalisation)
    # norm = np.mean(normalisation)
    
    # return norm

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

def quick_plot(tilt, azimuthals, cos, norm): #pre normalised data
    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " norm data.csv"), delimiter=",")
    norm_ex = norm * 1/cos[5]
    data = data.T

    plt.scatter(data[0], data[1], label = str(tilt) + " data")
    plt.plot(azimuthals, cos*norm_ex, label = str (tilt) + " cosine losses", marker = ".", color = "red")
    plt.xlabel("Azimuthal Tilt")
    plt.ylabel("Total Incident Intensity")
    plt.legend()
    plt.show()

def normalised_tilt_by_azim(tilt, azimuthals, cos, norm, object_num, colors, int_factor = 1.4e6, fontsize = 12, save = False):
    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
    norm_ex = norm * 1/cos[5] * 1/int_factor
    data = data.T

    intensities = data[1]/int_factor
    azimuths = data[0]

    for i, num in enumerate(object_num):
        if num == 4:
            plt.scatter(azimuths[i], intensities[i], label = "Experimental data", color = colors[0])

        elif num != 4:
            plt.scatter(azimuths[i], intensities[i], label = "Unnormalised data", marker = 'x', color = colors[0])
            plt.scatter(azimuths[i], intensities[i]*4/num, label = "Normalised data", color = colors[0], alpha = 0.3)


    plt.plot(azimuthals, cos*norm_ex, label = "Cosine Prediction", marker = ".", color = "red")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel(r"Azimuthal tilt ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)

    if save:
        plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " azimuth by tilt graph.png"), dpi = 1500)

    plt.show()

def all_averaged_tilt_by_azim(tilts, azimuthals, coss, norms, object_nums, colors, int_factor = 1.4e6, fontsize = 12):
    plt.figure(figsize=(8, 5), dpi=100)

    for i, tilt in enumerate(tilts):
        data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
        cos = coss[i]
        norm = norms[i]
        object_num = object_nums[i]

        norm_ex = norm * 1/cos[5] * 1/int_factor
        data = data.T

        intensities = data[1]/int_factor
        azimuths = data[0]
        y = []
        err = []

        for j in range(0, len(azimuthals)):
            a = intensities[(j*3)] * (4/object_num[(j*3)])
            b = intensities[(j*3 + 1)] * (4/object_num[(j*3 + 1)])
            c = intensities[(j*3 + 2)] * (4/object_num[(j*3 + 2)])
            y.append(np.mean([a,b,c]))
            err.append(np.std([a,b,c]))

        err = np.array(err)*0.8
            
        plt.errorbar(azimuthals, y, yerr = err, xerr = 4, label = str(tilt) + " data", color = colors[i], marker = ".", ls = 'none', capsize = 3, alpha = 0.7)
        plt.plot(azimuthals, cos*norm_ex, label = str(tilt) + " Cosine", color = colors[i])
    
    #plt.ylim(0.4, 1.2)
    
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    plt.xlabel(r"Azimuthal tilt ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)
    plt.tight_layout()

    plt.savefig(("Image analysis v3/Final graphs/All averaged azimuth by tilt graph.png"), dpi = 1500)
    plt.show()

def averaged_tilt_by_azim(tilt, azimuthals, cos, norm, object_num, colors, int_factor = 1.4e6, fontsize = 12):

    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
    norm_ex = norm * 1/cos[5] * 1/int_factor
    data = data.T

    intensities = data[1]/int_factor
    azimuths = data[0]
    y = []
    err = []

    for j in range(0, len(azimuthals)):
        a = intensities[(j*3)] * (4/object_num[(j*3)])
        b = intensities[(j*3 + 1)] * (4/object_num[(j*3 + 1)])
        c = intensities[(j*3 + 2)] * (4/object_num[(j*3 + 2)])
        y.append(np.mean([a,b,c]))
        err.append(np.std([a,b,c]))

    err = np.array(err)*0.8
        
    plt.errorbar(azimuthals, y, yerr = err, xerr = 4, label = str(tilt) + " data", color = colors[0], marker = "o", ls = 'none', capsize = 3)
    plt.plot(azimuthals, cos*norm_ex, label = "Cosine Prediction", marker = ".", color = "red")

    if tilt < 40:
        plt.legend(loc = "upper center") 

    else: plt.legend(loc = "lower center")

    plt.xlabel(r"Azimuthal tilt ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)
    
    plt.savefig(("Image analysis v3/Final graphs/" + str(tilt) + " averaged azimuth by tilt graph.png"), dpi = 1500)
    plt.show()

def sep_and_variance(tilt, azimuthals):
    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " norm data.csv"), delimiter=",")
    data = data.T

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(data[0], data[3])
    ax2.scatter(data[0], data[4])

    ax1.set_title("Separation from target")
    ax2.set_title("Variance")

def sep(tilts, azimuthals, colors, average = False):

    for i, t in enumerate(tilts):
        data = np.loadtxt(("Image analysis v3/"+ str(t) + " norm data.csv"), delimiter=",")
        data = data.T

        seps = np.array(data[3])/45

        if not average:
            plt.scatter(data[0], seps, color = colors[i], label = str(t) + " data")

        elif average:
            y = np.mean(seps.reshape(-1, 3), axis = 1)
            plt.scatter(azimuthals, y, color = colors[i], marker = "o", label = str(t) + " data")

            total_mean = np.mean(y)
            plt.plot([-70, 70], [total_mean, total_mean], color = colors[i], linestyle = "-")


    plt.xlabel(r"Azimuthal tilt ($^\circ$)")
    plt.ylabel("Separation of CoM from target (cm)")
    plt.legend()
    plt.show()

def averaged_tilt_with_sim(tilt, azimuthals, cos, norm, object_num, colors, sim_data, sim_num, tilt_index, int_factor = 1.4e6, fontsize = 12, scale = True):

    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
    norm_ex = norm * 1/cos[5] * 1/int_factor
    data = data.T

    intensities = data[1]/int_factor
    azimuths = data[0]
    y = []
    err = []
    sim_ys = []

    for j, azim in enumerate(azimuthals):
        a = intensities[(j*3)] * (4/object_num[(j*3)])
        b = intensities[(j*3 + 1)] * (4/object_num[(j*3 + 1)])
        c = intensities[(j*3 + 2)] * (4/object_num[(j*3 + 2)])
        y.append(np.mean([a,b,c]))
        err.append(np.std([a,b,c]))

        mirr_num = sim_num[j+1][tilt_index+1]

        if scale:
            sim_y = sim_data[tilt, azim] * 4/mirr_num
        
        else: sim_y = sim_data[tilt, azim]
        sim_ys.append(sim_y)
        
    sim_ys = np.array(sim_ys)*y[5]/sim_ys[5]
    
    err = np.array(err)*0.8
        
    plt.errorbar(azimuthals, y, yerr = err, xerr = 4, label = str(tilt) + " data", color = "blue", marker = "o", ls = 'none', capsize = 3)
    plt.plot(azimuthals, cos*norm_ex, label = "Cosine Prediction", marker = ".", color = "m")
    plt.plot(azimuthals, sim_ys, label = "Scaled sim. data", marker = "^", color = "red", ls = "none")

    if tilt < 40:
        plt.legend(loc = "upper center") 

    else: plt.legend(loc = "lower center")

    plt.xlabel(r"Azimuthal tilt ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)
    
    plt.savefig(("Image analysis v3/Final graphs/" + str(tilt) + " averaged azimuth by tilt graph.png"), dpi = 1500)
    plt.show()

def normalised_tilt_with_sim(tilt, azimuthals, cos, norm, object_num, colors, sim_data, int_factor = 1.4e6, fontsize = 12):
    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
    norm_ex = norm * 1/cos[5] * 1/int_factor
    data = data.T

    intensities = data[1]/int_factor
    azimuths = data[0]
    sim_ys = []
    avg = []

    for i, num in enumerate(object_num):
        if i == 15 or i == 16 or i ==17:
            avg.append(intensities[i])

        if num == 4:
            plt.scatter(azimuths[i], intensities[i], label = "Unscaled data", marker = "o", color = colors[0])

        elif num != 4:
            plt.scatter(azimuths[i], intensities[i], label = "Unscaled data", marker = 'o', color = colors[0])
            plt.scatter(azimuths[i], intensities[i]*4/num, label = "Scaled data", marker = "x", color = colors[0], alpha = 0.7)

    
    for azim in azimuthals:
        sim_y = sim_data[tilt, azim]
        sim_ys.append(sim_y) 

    sim_ys = np.array(sim_ys)*np.mean(avg)/sim_ys[5]
    plt.scatter(azimuthals, sim_ys, label = "Simulation data", marker = "^", color = "red")

    plt.plot(azimuthals, cos*norm_ex, label = "Cosine Prediction", color = "m")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    if tilt < 40:
        plt.legend(by_label.values(), by_label.keys(), loc = "upper center")

    else:
        plt.legend(by_label.values(), by_label.keys(), loc = "lower center")

    plt.xlabel(r"Azimuthal tilt ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)

    plt.savefig(("Image analysis v3/Final graphs/" + str(tilt) + " azimuth by tilt graph w sim.png"), dpi = 1500)

    plt.show()

def heatmap(tilts, azimuthals, fs = 10, fourhst = False):    

    map = []
    for tilt in tilts:
        if not fourhst:
            data = np.loadtxt(("Image analysis v3/" + str(tilt) + " 4 nnorm data.csv"), delimiter = ",") 
        elif fourhst:
            data = np.loadtxt(("Image analysis v3/" + str(tilt) + " 8 norm data.csv"), delimiter = ",")
        
        data = data.T
        intensities = data[1]/1.4e6

        if not fourhst:
            y = np.mean(intensities.reshape(-1, 3), axis = 1)
        elif fourhst:
            y = intensities.copy()
        map.append(y)

    az_labs = ["-70", "-60", "-45", "-30", "-15", "0", "15", "30", "45", "60", "70"]
    ti_labs = ["15", "30", "45", "60"]
        
    map = np.array(map)
    s = sns.heatmap(map, xticklabels=az_labs, yticklabels=ti_labs)

    s.set_xlabel('Azimuth ($^\circ$)', fontsize = fs)
    s.set_ylabel('Elevation ($^\circ$)', fontsize = fs)
    s.collections[0].colorbar.set_label("Energy Incident (a.u.)", fontsize = fs)

    if fourhst:
        plt.savefig("Image analysis v3/report graphs/heatmap 4hst.png", dpi= 1000)

    elif not fourhst:
        plt.savefig("Image analysis v3/report graphs/heatmap 2hst.png", dpi= 1000)
    plt.show()


if __name__ == "__main__":
    norms = [690122.0, 1066143.7, 1078526.0, 1277268.3]
    norm_15, norm_30, norm_45, norm_60 = norms[0], norms[1], norms[2], norms[3]

    norms_8 = [1729972, 2110938, 2451893, 2738462]

    tilts = np.arange(15, 75, 15)
    azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]
    colours = ["red", "gold", "darkgreen", "blue"]
    incident_angles = np.loadtxt("simple_incident_angles.csv", delimiter = ",")

    cos_15 = np.cos(incident_angles[1][1:]*np.pi/360)
    cos_30 = np.cos(incident_angles[2][1:]*np.pi/360)
    cos_45 = np.cos(incident_angles[3][1:]*np.pi/360)
    cos_60 = np.cos(incident_angles[4][1:]*np.pi/360)
    cos_all = [cos_15, cos_30, cos_45, cos_60]

    #quick_plot(15, azimuthals=azimuthals, cos = cos_15, norm = norms[0])
    # #quick_plot(30, azimuthals=azimuthals, cos = cos, norm = norms[1])
    # #quick_plot(45, azimuthals=azimuthals, cos = cos, norm = norms[2])
    # quick_plot(60, azimuthals=azimuthals, cos = cos, norm = norms[3])

    # sep(tilts, azimuthals, colours)

    #for full data set
    folder = "Image analysis v3/full data set/"
    mirr_15, mirr_30, mirr_45, mirr_60 = np.loadtxt((folder + "mirror_numbers.csv"), skiprows=1, delimiter = ",", unpack = True, usecols=range(1,5))
    # mirr_all = [mirr_15, mirr_30, mirr_45, mirr_60]
    # print(len(mirr_15))

    with open('raytracer_data-18.03/all_simages_25Mrays_uniform.pkl', 'rb') as f:
        sim_data = pkl.load(f)

    sim_num = np.loadtxt("sim_mirr_numbs.csv", delimiter=",")

    #try new data set
    folder_8 = "Image analysis v3/4 heliostat 8_04/"
    # plot_by_tilt(tilts, [-30, 0, 15, 30, 45, 60, 70], folder_8, 1, [640, 544], colours)
    # plt.show()


    m8_15 = [8,8,8,8,8,6,6]
    m8_30 = [8,8,7,7,7,7,6]
    m8_45 = [8,8,8,7,7,6,6]
    m8_60 = [7,8,8,8,7,7,6]


    #plot_by_object_num(15, [-30, 0, 15, 30, 45, 60, 70], folder_8, m8_15, 1, [744,564], colours, normalise = True, total_mirrs= 8)
    # plot_by_object_num(30, [-30, 0, 15, 30, 45, 60, 70], folder_8, m8_30, 1, [744,564], colours, normalise = True, total_mirrs= 8)
    # plot_by_object_num(45, [-30, 0, 15, 30, 45, 60, 70], folder_8, m8_45, 1, [744,564], colours, normalise = True, total_mirrs= 8)
    # plot_by_object_num(60, [-30, 0, 15, 30, 45, 60, 70], folder_8, m8_60, 1, [744,564], colours, normalise = True, total_mirrs= 8)


    #THIS FOR GRAPHS W UNNORMALISED DATA
    #normalised_tilt_by_azim(15, azimuthals, cos_15, norm_15, mirr_15, colors = ['blue', 'green'], save = True)
    #normalised_tilt_by_azim(30, azimuthals, cos_30, norm_30, mirr_30, colors = ['blue', 'green'], save = True)
    #normalised_tilt_by_azim(45, azimuthals, cos_45, norm_45, mirr_45, colors = ['blue', 'green'], save = True)
    #normalised_tilt_by_azim(60, azimuthals, cos_60, norm_60, mirr_60, colors = ['blue', 'green'], save = True)

    #normalised_tilt_with_sim(15, azimuthals, cos_15, norm_15, mirr_15, ['blue', 'green'], sim_data)
    # normalised_tilt_with_sim(30, azimuthals, cos_30, norm_30, mirr_30, ['blue', 'green'], sim_data)
    # normalised_tilt_with_sim(45, azimuthals, cos_45, norm_45, mirr_45, ['blue', 'green'], sim_data)
    # normalised_tilt_with_sim(60, azimuthals, cos_60, norm_60, mirr_60, ['blue', 'green'], sim_data)
    
    col = "blue"
    # averaged_tilt_by_azim(15, azimuthals, cos_15, norm_15, mirr_15, colors = [col])
    # averaged_tilt_by_azim(30, azimuthals, cos_30, norm_30, mirr_30, colors = [col])
    # averaged_tilt_by_azim(45, azimuthals, cos_45, norm_45, mirr_45, colors = [col])
    # averaged_tilt_by_azim(60, azimuthals, cos_60, norm_60, mirr_60, colors = [col])


    #averaged_tilt_with_sim(15, azimuthals, cos_15, norm_15, mirr_15, [col], sim_data, sim_num, 0)
    # averaged_tilt_with_sim(30, azimuthals, cos_30, norm_30, mirr_30, [col], sim_data, sim_num, 1)
    # averaged_tilt_with_sim(45, azimuthals, cos_45, norm_45, mirr_45, [col], sim_data, sim_num, 2)
    # averaged_tilt_with_sim(60, azimuthals, cos_60, norm_60, mirr_60, [col], sim_data, sim_num, 3)

    # sep(tilts, azimuthals, colours, average = True)

    #all_averaged_tilt_by_azim(tilts, azimuthals, cos_all, norms, mirr_all, colors = colours)

    #heatmap(tilts, azimuthals)

    average = False
    normalise = True

    #plot_by_object_num(15, azimuthals, folder, mirr_15, 3, [640, 544], colours, True, False)
    # plt.show()
    #plot_by_object_num(30, azimuthals, folder, mirr_30, 3, [640, 544], colours, normalise, average)
    #plot_by_object_num(45, azimuthals, folder, mirr_45, 3, [640, 544], colours, normalise, average)
    plot_by_object_num(60, azimuthals, folder, mirr_60, 3, [640, 544], colours, normalise, average)

    #for simulation data
    # folder = "raytracer_data-18.03/images/"
    # sim_plot_by_tilt(tilts, azimuthals, folder, [512, 640], colours)



    # for finding target
    # image = im.imread("Image analysis v3/full data set/15_-30_back1.png")
    # im.imshow(image)
    # plt.show()