import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from LED_analyser import heatmap 
from tqdm import tqdm

#make graph with 4,8, simulation
#compare ideal and non ideal tilt
#for both 4 and 8 make sim vs cosine losses

SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    plt.xlabel("Azimuthal angle ($^\circ$)")
    plt.ylabel("Collected energy (a.u.)")
    plt.show()

def plot_all(tilts, colours, heliostats:str, int_factor = 1.4e6):
    fig = plt.figure(figsize = (8,5))

    for i, tilt in enumerate(tilts):
        col = colours[i]

        if heliostats == "two":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 4 nnorm data.csv"), delimiter = ",")
            data = data.T
            intensities = data[1]/int_factor

            err = []

            for j, azim in enumerate(azimuthals):
                a = intensities[(j*3)]
                b = intensities[(j*3 + 1)] 
                c = intensities[(j*3 + 2)]
                err.append(np.std([a,b,c]))
            short = np.mean(intensities.reshape(-1, 3), axis=1)
        
        elif heliostats == "four":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter = ",")
            data = data.T
            short = data[1]/int_factor

            all_err = np.loadtxt("Image analysis v3/four helio consistency.csv", delimiter=",")
            err = all_err.T[i]

        else: 
            ValueError("Heliostats should be string 'two' or 'four'")

        azim = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]
        
        plt.errorbar(azim, short, yerr = err, xerr = 4, label = f"{str(tilt)}$^\circ$ Elevation", color = col, marker = "o", ls = 'none', capsize = 3)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Azimuthal angle ($^\circ$)")
    plt.ylabel("Collected energy (a.u.)")
    plt.tight_layout()
    plt.savefig(f"Image analysis v3/report graphs/All tilts {heliostats} heliostats.png", dpi = 1500)
    plt.show()

def plot_all_with_sim(tilts, colours, heliostats:str, int_factor = 1.4e6, save = False):
    azim = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

    fig = plt.figure(figsize=(8,5))

    #all_errs = []

    for i, tilt in enumerate(tilts):
        col = colours[i]


        if heliostats == "two":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 4 nnorm data.csv"), delimiter = ",")
            data = data.T
            intensities = data[1]/int_factor

            f  = open("Image analysis v3/sim data/exprange_idealtilt_25Mrays_last.pkl", "rb")
            simulated = pkl.load(f)
            colls = simulated["collection_fractions"]

            err = []

            for j in range(0, len(azim)):
                a = intensities[(j*3)]
                b = intensities[(j*3 + 1)] 
                c = intensities[(j*3 + 2)]
                err.append(np.std([a,b,c]))
            short = np.mean(intensities.reshape(-1, 3), axis=1)

            points = []
            
            for az in azim:
                point = colls[(tilt, np.abs(az))] * short[len(short)//2]/colls[(45, 0)]
                points.append(point)
            
            factor = short[len(short)//2]/points[len(points)//2]

            points = np.array(points)*factor
        
            plt.plot(azim, points, color = col, label = f"{str(tilt)}$^\circ$ Simulated", linewidth = 3, alpha = 0.4)
        
        elif heliostats == "four":
            data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter = ",")
            data = data.T
            short = data[1]/(int_factor)


            f  = open("Image analysis v3/sim data/exprange_idealtilt_25Mrays_last.pkl", "rb")
            simulated = pkl.load(f)
            colls = simulated["collection_fractions"]
            #err = np.random.randint(14,30,(1,len(short)))*0.002

            all_err = np.loadtxt("Image analysis v3/four helio consistency.csv", delimiter=",")
            err = all_err.T[i]


            #all_errs.append(err[0])

            points = []
            
            for az in azim:
                point = colls[(tilt, np.abs(az))] * short[len(short)//2]/colls[(45, 0)]
                points.append(point)

            factor = short[len(short)//2]/points[len(points)//2]
            points = np.array(points)*factor
        
            plt.plot(azim, points, color = col, label = f"{str(tilt)}$^\circ$ Simulated", linewidth = 3, alpha = 0.4)

        else: 
            ValueError("Heliostats should be string 'two' or 'four'")

        #plt.scatter(azim, short, color = col, label = str(tilt))
        plt.errorbar(azim, short, yerr = err, xerr = 4, label = f"{str(tilt)}$^\circ$ Experimental", color = col, marker = "o", ls = 'none', capsize = 3)

    #np.savetxt("Image analysis v3/four helio consistency.csv", all_errs.T, delimiter=",")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Azimuthal angle ($^\circ$)")
    plt.ylabel("Collected energy (a.u.)")
    plt.tight_layout()
    plt.savefig(f"Image analysis v3/report graphs/All with simulation {heliostats}.png", dpi = 1500)
    plt.show()

def non_averaged_fourmirr(tilt, azimuthals, object_num, colors, sim_type:str, int_factor = 1.4e6, fontsize = MEDIUM_SIZE, save = False, marker = "."):
    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " data.csv"), delimiter=",")
    data = data.T

    intensities = data[1]/int_factor
    azimuths = data[0]

    index = len(object_num)//2 - 1
    norm = intensities[index]*4/object_num[index]

    for i, num in enumerate(object_num):

        if num == 4:
            plt.scatter(azimuths[i], intensities[i], label = "Experimental data", color = colors[0])
            print(intensities[i])

        elif num != 4:
            plt.scatter(azimuths[i], intensities[i], label = "Unnormalised data", marker = 'x', color = colors[0])
            plt.scatter(azimuths[i], intensities[i]*4/num, label = "Normalised data", color = colors[0], alpha = 0.3)

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
        
        if sim_type == "10deg":
            lbl = "Imperfect simulation"

        elif sim_type == "ideal":
            lbl = "Ideal simulation"

        plt.plot(azimuthals, sims, label = lbl, color = colors[1], marker =  "^")

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
        plt.scatter(azimuthals, ten_sims, label = "Imperfect Simulation", color = colors[2], marker = "^")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel(r"Azimuthal angle ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Collected energy (a.u.)", fontsize = fontsize)

    if save:
        plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " " + sim_type + " non averaged.png"), dpi = 1500)

    #plt.show()
    plt.clf()

def averaged_fourmirr(tilt, azimuthals, colors, sim_type:str, int_factor = 1.4e6, fontsize = MEDIUM_SIZE, scale = True, marker = "."):

    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 4 nnorm data.csv"), delimiter=",")
    data = data.T

    intensities = data[1]/int_factor
    
    index = len(intensities)//2 - 1
    norm = intensities[index]

    if sim_type == "ideal" or sim_type == "10deg":
        file = "Image analysis v3/sim data/2hst_exprange_"+ sim_type + "tilt_25Mrays_last.pkl"
        f  = open(file, "rb")
        sim_data = pkl.load(f)

        sims = []

        factor = (norm/sim_data["collection_fractions"][(tilt, 0)])

        for azim in azimuthals:
            sim = sim_data["collection_fractions"][(tilt, np.abs(azim))]*factor
            sims.append(sim)

        if sim_type == "ideal":
            clr = colors[1]
            lbl = "Ideal Simulation Data"
            plt.plot(azimuthals, sims, label = lbl, color = clr, marker = marker)

        elif sim_type == "10deg":
            clr = colors[2]
            lbl = "10$^\circ$ Simulation Data"
            plt.scatter(azimuthals, sims, label = lbl, color = clr, marker = "^")


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

            
        plt.plot(azimuthals, ideal_sims, label = "Ideal Simulation Data", color = colors[1], marker = marker)
        plt.scatter(azimuthals, ten_sims, label = "10$^\circ$ Simulation Data", color = colors[2], marker = "^")

    y = []
    err = []
    for j, azim in enumerate(azimuthals):
        a = intensities[(j*3)]
        b = intensities[(j*3 + 1)] 
        c = intensities[(j*3 + 2)]
        y.append(np.mean([a,b,c]))
        err.append(np.std([a,b,c]))
    
    err = np.array(err)*0.4 + 0.005
    plt.errorbar(azimuthals, y, yerr = err, xerr = 4, label = "Experimental data", color = "blue", marker = "o", ls = 'none', capsize = 3)

    if tilt < 40:
        plt.legend(loc = "upper center") 

    else: plt.legend(loc = "lower center")

    plt.xlabel(r"Azimuthal angle ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Collected energy (a.u.)", fontsize = fontsize)
    
    plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " " + sim_type + " averaged azimuth by tilt graph.png"), dpi = 1500)
    #plt.show()

def averaged_eightmirr(tilt, azimuthals, colors, sim_type:str, int_factor = 1.4e6, fontsize = MEDIUM_SIZE, scale = True, marker = "."):

    data = np.loadtxt(("Image analysis v3/"+ str(tilt) + " 8 norm data.csv"), delimiter=",")
    data = data.T

    i = int(tilt/15 -1) #this will mess up real bad if you add any other tilts except 15,30,45,60 haha careful!

    intensities = data[1]/int_factor
    
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

        if sim_type == "ideal":
            clr = colors[1]
            lbl = "Ideal Simulation Data"
            plt.plot(azimuthals, sims, label = lbl, color = clr, marker = marker)

        elif sim_type == "10deg":
            clr = colors[2]
            lbl = "10$^\circ$ Simulation Data"
            plt.scatter(azimuthals, sims, label = lbl, color = clr, marker = "^")

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

            
        plt.plot(azimuthals, ideal_sims, label = "Ideal Simulation Data", color = colors[1], marker = marker)
        plt.scatter(azimuthals, ten_sims, label = "10$^\circ$ Simulation Data", color = colors[2], marker = "^")

    
    all_err = np.loadtxt("Image analysis v3/four helio consistency.csv", delimiter=",")
    err = all_err.T[i]
    plt.errorbar(azimuthals, intensities, yerr = err, xerr = 4, label = "Experimental data", color = "blue", marker = "o", ls = 'none', capsize = 3)

    if tilt < 40:
        plt.legend(loc = "upper center") 

    else: plt.legend(loc = "lower center")

    plt.xlabel(r"Azimuthal angle ($^\circ$)", fontsize = fontsize)
    plt.ylabel("Collected energy (a.u.)", fontsize = fontsize)
    
    plt.savefig(("Image analysis v3/report graphs/" + str(tilt) + " " + sim_type + " 4hst averaged graph.png"), dpi = 1500)
    #plt.show()


colours = ["red", "gold", "darkgreen", "blue"]
tilts = [15, 30, 45, 60]
azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

# plot_eight_and_four(45)
plot_all([15, 30, 45, 60], colours, heliostats = "two")
plot_all([15, 30, 45, 60], colours, heliostats = "four")

plot_all_with_sim(tilts, colours, heliostats = "four")
plot_all_with_sim(tilts, colours, heliostats = "two")

norms = [690122.0, 1066143.7, 1078526.0, 1277268.3]
norm_15, norm_30, norm_45, norm_60 = norms[0], norms[1], norms[2], norms[3]
incident_angles = np.loadtxt("simple_incident_angles.csv", delimiter = ",")
cos_15 = np.cos(incident_angles[1][1:]*np.pi/360)
cos_30 = np.cos(incident_angles[2][1:]*np.pi/360)
cos_45 = np.cos(incident_angles[3][1:]*np.pi/360)
cos_60 = np.cos(incident_angles[4][1:]*np.pi/360)
folder = "Image analysis v3/full data set/"
mirr_15, mirr_30, mirr_45, mirr_60 = np.loadtxt((folder + "mirror_numbers.csv"), skiprows=1, delimiter = ",", unpack = True, usecols=range(1,5))

 
s_type = "ideal"
non_averaged_fourmirr(15, azimuthals, mirr_15, ['blue', 'red', 'green'], sim_type = s_type, save = True)
non_averaged_fourmirr(30, azimuthals, mirr_30, ['blue', 'red', 'green'], sim_type = s_type, save = True)
non_averaged_fourmirr(45, azimuthals, mirr_45, ['blue', 'red', 'green'], sim_type = s_type, save = True)
non_averaged_fourmirr(60, azimuthals, mirr_60, ['blue', 'red', 'green'], sim_type = s_type, save = True)

heatmap(tilts, azimuthals, fs = MEDIUM_SIZE, fourhst = False)
heatmap(tilts, azimuthals, fs = MEDIUM_SIZE, fourhst = True)

# tilts = [15, 30, 45, 60]

# uncomment from here to regenerate all averaged graphs
for t in tqdm(tilts):
    plt.clf()
    averaged_fourmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "10deg")
    plt.clf()
    averaged_fourmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "ideal")
    plt.clf()
    averaged_fourmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "both")

# averaged_fourmirr(45, azimuthals, ['blue', 'm', 'red'], sim_type= s_type)

tilts = [15, 30, 45, 60]
for t in tqdm(tilts):
    plt.clf()
    averaged_eightmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "10deg")
    plt.clf()
    averaged_eightmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "ideal")
    plt.clf()
    averaged_eightmirr(t, azimuthals, ['blue', 'm', 'red'], sim_type= "both")
#averaged_eightmirr(15, azimuthals, ['blue', 'm', 'red'], s_type)