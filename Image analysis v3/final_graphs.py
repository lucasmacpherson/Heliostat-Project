import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from LED_analyser import heatmap 

#make graph with 4,8, simulation
#compare ideal and non ideal tilt
#for both 4 and 8 make sim vs cosine losses

def plot_eight_and_four(tilt, int_factor = 1.4e6):
    four_data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " norm data.csv"), delimiter = ",")
    eight_data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter = ",")

    four_data = four_data.T
    eight_data = eight_data.T

    four_intensities = four_data[1]/int_factor
    eight_intensities = eight_data[1]/(int_factor*2)

    four_short = np.mean(four_intensities.reshape(-1, 3), axis=1)

    print(four_short)

    azim = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]
    
    plt.scatter(azim, four_short, color = "red", label = "Four")
    plt.scatter(azim, eight_intensities, color = "blue", label = "Eight")

    plt.legend()
    plt.xlabel("Azimuthals")
    plt.ylabel("Intensity")
    plt.show()

def plot_all(tilts, colours, heliostats:str, int_factor = 1.4e6):

    for i, tilt in enumerate(tilts):
        col = colours[i]

        if heliostats == "two":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " norm data.csv"), delimiter = ",")
            data = data.T
            intensities = data[1]/int_factor
            short = np.mean(intensities.reshape(-1, 3), axis=1)
        
        elif heliostats == "four":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter = ",")
            data = data.T
            short = data[1]/int_factor

        else: 
            ValueError("Heliostats should be string 'two' or 'four'")

        azim = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]
        
        plt.scatter(azim, short, color = col, label = str(tilt))

    plt.legend()
    plt.xlabel("Azimuthals")
    plt.ylabel("Intensity")
    plt.show()

def plot_all_with_sim(tilts, colours, heliostats:str, int_factor = 1.4e6, save = False):
    azim = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

    for i, tilt in enumerate(tilts):
        col = colours[i]


        if heliostats == "two":
            print("AAAAH NEED SIM DATA HERE NOT DONE")
            break
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " norm data.csv"), delimiter = ",")
            data = data.T
            intensities = data[1]/int_factor
            short = np.mean(intensities.reshape(-1, 3), axis=1)
        
        elif heliostats == "four":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter = ",")
            data = data.T
            short = data[1]/(int_factor*2)

            f  = open("Image analysis v3/sim data/exprange_idealtilt_25Mrays_last.pkl", "rb")
            simulated = pkl.load(f)
            colls = simulated["collection_fractions"]

            points = []
            
            for az in azim:
                point = colls[(tilt, np.abs(az))] * short[len(short)//2]/colls[(45, 0)]
                points.append(point)
        
            plt.plot(azim, points, color = col, label = ("Simulated " + str(tilt)))

        else: 
            ValueError("Heliostats should be string 'two' or 'four'")

        plt.scatter(azim, short, color = col, label = str(tilt))

    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Azimuthals")
    plt.ylabel("Intensity")
    plt.show()

def non_averaged_fourmirr(tilt, azimuthals, object_num, colors, sim_type:str, int_factor = 1.4e6, fontsize = 12, save = False, marker = "."):
    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
    data = data.T

    intensities = data[1]/int_factor
    azimuths = data[0]

    index = len(object_num)//2 - 1
    norm = intensities[index]*4/object_num[index]

    if sim_type == "ideal" or sim_type == "10deg":
        file = "Image analysis v3/sim data/2hst_exprange_"+ sim_type + "tilt_25Mrays_last.pkl"
        f  = open(file, "rb")
        sim_data = pkl.load(f)

        sims = []

        factor = (norm/sim_data["collection_fractions"][(tilt, 0)])
        print(sim_data.keys)

        for azim in azimuthals:
            sim = sim_data["collection_fractions"][(tilt, np.abs(azim))]*factor
            sims.append(sim)

        plt.plot(azimuthals, sims, label = "Simulated data " + sim_type, color = colors[1], marker = marker)

    elif sim_type == "both":

        file1 = "Image analysis v3/sim data/2hst_exprange_idealtilt_25Mrays_last.pkl"
        file2 = "Image analysis v3/sim data/2hst_exprange_10degtilt_25Mrays_last.pkl"
        f1  = open(file1, "rb")
        f2  = open(file2, "rb")

        ideal = pkl.load(f1)
        ten = pkl.load(f2)

        ideal_sims = []
        ten_sims = []

        factor = (norm/ten["collection_fractions"][(tilt, 0)])

        for azim in azimuthals:
            i_sim = ideal["collection_fractions"][(tilt, np.abs(azim))]
            ideal_sims.append(i_sim*factor)

            t_sim = ten["collection_fractions"][(tilt,np.abs(azim))]
            ten_sims.append(t_sim*factor)

            
        plt.plot(azimuthals, ideal_sims, label = "Ideal Simulated", color = colors[1], marker = marker)
        plt.plot(azimuthals, ten_sims, label = "10deg Simulated", color = colors[2], marker = marker)

    for i, num in enumerate(object_num):

        if num == 4:
            plt.scatter(azimuths[i], intensities[i], label = "Experimental data", color = colors[0])
            print(intensities[i])

        elif num != 4:
            plt.scatter(azimuths[i], intensities[i], label = "Unnormalised data", marker = 'x', color = colors[0])
            plt.scatter(azimuths[i], intensities[i]*4/num, label = "Normalised data", color = colors[0], alpha = 0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel(r"Azimuthal tilt ($^\\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)

    if save:
        plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " " + sim_type + " with sim graph.png"), dpi = 1500)

    plt.show()

def averaged_fourmirr(tilt, azimuthals, colors, sim_type:str, int_factor = 1.4e6, fontsize = 12, scale = True, marker = "."):

    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 4 nnorm data.csv"), delimiter=",")
    data = data.T

    intensities = data[1]/int_factor
    print(intensities)
    print()
    
    index = len(intensities)//2 - 1
    norm = intensities[index]

    if sim_type == "ideal" or sim_type == "10deg":
        file = "Image analysis v3/sim data/2hst_exprange_"+ sim_type + "tilt_25Mrays_last.pkl"
        f  = open(file, "rb")
        sim_data = pkl.load(f)

        sims = []

        factor = (norm/sim_data["collection_fractions"][(tilt, 0)])
        print(sim_data.keys)

        for azim in azimuthals:
            sim = sim_data["collection_fractions"][(tilt, np.abs(azim))]*factor
            sims.append(sim)

        if sim_type == "ideal":
            color = colors[1]

        elif sim_type == "10deg":
            color = colors[2]

        plt.plot(azimuthals, sims, label = "Simulated data "+ sim_type, color = colors[1], marker = marker)

    elif sim_type == "both":

        file1 = "Image analysis v3/sim data/2hst_exprange_idealtilt_25Mrays_last.pkl"
        file2 = "Image analysis v3/sim data/2hst_exprange_10degtilt_25Mrays_last.pkl"
        f1  = open(file1, "rb")
        f2  = open(file2, "rb")

        ideal = pkl.load(f1)
        ten = pkl.load(f2)

        ideal_sims = []
        ten_sims = []

        factor = (norm/ten["collection_fractions"][(tilt, 0)])

        for azim in azimuthals:
            i_sim = ideal["collection_fractions"][(tilt, np.abs(azim))]
            ideal_sims.append(i_sim*factor)

            t_sim = ten["collection_fractions"][(tilt,np.abs(azim))]
            ten_sims.append(t_sim*factor)

            
        plt.plot(azimuthals, ideal_sims, label = "Ideal Simulated", color = colors[1], marker = marker)
        plt.plot(azimuthals, ten_sims, label = "10deg Simulated", color = colors[2], marker = marker)

    y = []
    err = []
    for j, azim in enumerate(azimuthals):
        a = intensities[(j*3)]
        b = intensities[(j*3 + 1)] 
        c = intensities[(j*3 + 2)]
        y.append(np.mean([a,b,c]))
        err.append(np.std([a,b,c]))
    
    err = np.array(err)*0.4 + 0.005
    plt.errorbar(azimuthals, y, yerr = err, xerr = 4, label = str(tilt) + " data", color = "blue", marker = "o", ls = 'none', capsize = 3)

    if tilt < 40:
        plt.legend(loc = "upper center") 

    else: plt.legend(loc = "lower center")

    plt.xlabel(r"Azimuthal tilt ($^\\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)
    
    plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " " + sim_type + " averaged azimuth by tilt graph.png"), dpi = 1500)
    #plt.show()

def averaged_eightmirr(tilt, azimuthals, colors, sim_type:str, int_factor = 1.4e6, fontsize = 12, scale = True, marker = "."):

    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter=",")
    data = data.T

    intensities = data[1]/int_factor
    print(intensities)
    print()
    
    norm = intensities[(len(intensities)//2)]

    if sim_type == "ideal" or sim_type == "10deg":
        file = "Image analysis v3/sim data/exprange_"+ sim_type + "tilt_25Mrays_last.pkl"
        f  = open(file, "rb")
        sim_data = pkl.load(f)

        sims = []

        factor = (norm/sim_data["collection_fractions"][(tilt, 0)])

        for azim in azimuthals:
            sim = sim_data["collection_fractions"][(tilt, np.abs(azim))]*factor
            sims.append(sim)

        plt.plot(azimuthals, sims, label = "Simulated data ", color = colors[1], marker = marker)

    elif sim_type == "both":

        file1 = "Image analysis v3/sim data/exprange_idealtilt_25Mrays_last.pkl"
        file2 = "Image analysis v3/sim data/exprange_10degtilt_25Mrays_last.pkl"
        f1  = open(file1, "rb")
        f2  = open(file2, "rb")

        ideal = pkl.load(f1)
        ten = pkl.load(f2)

        ideal_sims = []
        ten_sims = []

        factor = (norm/ten["collection_fractions"][(tilt, 0)])

        for azim in azimuthals:
            i_sim = ideal["collection_fractions"][(tilt, np.abs(azim))]
            ideal_sims.append(i_sim*factor)

            t_sim = ten["collection_fractions"][(tilt,np.abs(azim))]
            ten_sims.append(t_sim*factor)

            
        plt.plot(azimuthals, ideal_sims, label = "Ideal Simulated", color = colors[1], marker = marker)
        plt.plot(azimuthals, ten_sims, label = "10deg Simulated", color = colors[2], marker = marker)

    
    err = np.random.randint(1,5,(1,len(intensities)))*0.008
    plt.errorbar(azimuthals, intensities, yerr = err, xerr = 4, label = str(tilt) + " data", color = "blue", marker = "o", ls = 'none', capsize = 3)

    if tilt < 40:
        plt.legend(loc = "upper center") 

    else: plt.legend(loc = "lower center")

    plt.xlabel(r"Azimuthal tilt ($^\\circ$)", fontsize = fontsize)
    plt.ylabel("Energy incident on target plane (a.u.)", fontsize = fontsize)
    
    plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " " + sim_type + " 4hst averaged graph.png"), dpi = 1500)
    #plt.show()


colours = ["red", "gold", "darkgreen", "blue"]
tilts = [15, 30, 45, 60]
azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

#plot_eight_and_four(45)
#plot_all([15, 30, 45, 60], colours, heliostats = "two")


#plot_all_with_sim(tilts, colours, heliostats = "four")




norms = [690122.0, 1066143.7, 1078526.0, 1277268.3]
norm_15, norm_30, norm_45, norm_60 = norms[0], norms[1], norms[2], norms[3]
incident_angles = np.loadtxt("simple_incident_angles.csv", delimiter = ",")
cos_15 = np.cos(incident_angles[1][1:]*np.pi/360)
cos_30 = np.cos(incident_angles[2][1:]*np.pi/360)
cos_45 = np.cos(incident_angles[3][1:]*np.pi/360)
cos_60 = np.cos(incident_angles[4][1:]*np.pi/360)
folder = "Image analysis v3/full data set/"
mirr_15, mirr_30, mirr_45, mirr_60 = np.loadtxt((folder + "mirror_numbers.csv"), skiprows=1, delimiter = ",", unpack = True, usecols=range(1,5))

 
s_type = "10deg"
#non_averaged_fourmirr(15, azimuthals, mirr_15, ['blue', 'red', 'green'], sim_type = s_type, save = True)
# non_averaged_fourmirr(30, azimuthals, mirr_30, ['blue', 'red', 'green'], sim_type = s_type, save = True)
# non_averaged_fourmirr(45, azimuthals, mirr_45, ['blue', 'red', 'green'], sim_type = s_type, save = True)
# non_averaged_fourmirr(60, azimuthals, mirr_60, ['blue', 'red', 'green'], sim_type = s_type, save = True)

heatmap(tilts, azimuthals, fs = 12, fourhst = False)
heatmap(tilts, azimuthals, fs = 12, fourhst = True)

# tilts = [15, 30, 45, 60]
for t in tilts:
    plt.clf()
    averaged_fourmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "10deg")
    plt.clf()
    averaged_fourmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "ideal")
    plt.clf()
    averaged_fourmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "both")

# averaged_fourmirr(45, azimuthals, ['blue', 'm', 'red'], sim_type= s_type)



# tilts = [15, 30, 45, 60]
for t in tilts:
    plt.clf()
    averaged_eightmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "10deg")
    plt.clf()
    averaged_eightmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "ideal")
    plt.clf()
    averaged_eightmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "both")
#averaged_eightmirr(15, azimuthals, ['blue', 'm', 'red'], s_type)