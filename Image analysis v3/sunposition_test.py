import numpy as np
import matplotlib.pyplot as plt
from sunposition import sunpos
from datetime import datetime, timezone, timedelta
import seaborn as sns
import pickle as pkl
import pandas as pd
from tqdm import tqdm

def gen_times(start, end, steps_per_day, step_size):

    if (steps_per_day +1)*step_size != 24:
        raise ValueError("Steps per day and step size do not make an exact day. Need (steps_per_day + 1)*step_size to equal 24")

    else:
        days = [start + timedelta(days=x) for x in range((end-start).days + 1)]
        times = []
        for day in days:
            for inc in range(0, steps_per_day+1):
                time = day + timedelta(hours = step_size*inc)
                #print(time)
                times.append(time)

        return times

def elev_az_time_heatmap(lon, lat, times, bins = 50, cmap = None):

    az,zen = sunpos(times,lat,lon,0)[:2] #discard RA, dec, H
    elev = 90 - zen

    data = {"Elevation": elev, "Azimuthal": az} 
    df = pd.DataFrame(data=data)

    df_day = df[df.Elevation > 0] # only those during the day! 

    if cmap is None:
        sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8))
    else: 
        sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), cmap = cmap)

    plt.show()

def latitude_heatmaps(lon, lats, times, bins = 50, cols = 3, cmap = None):

    rows = int(np.ceil(len(lats)/cols))

    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(("By latitude, longitude " + str(lon)))
    iter = 1
    
    #for lat in lats:   
    for lat, ax in zip(lats, axs.ravel()):
        az,zen = sunpos(times,lat,lon,0)[:2] #discard RA, dec, H
        elev = 90 - zen

        data = {"Elevation": elev, "Azimuthal": az} 
        df = pd.DataFrame(data=data)

        df_day = df[df.Elevation > 0] # only those during the day! 

        ax.set_title(("Latitude " + str(lat)))
        
        if cmap is None:
            sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), ax = ax)
        else:
            sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), ax = ax, cmap = cmap)

        print(f"Calculated for latitude {iter}/{len(lats)}")
        iter += 1

    plt.show() 

def long_and_lat_heatmaps(longs, lats, times, bins = 50, cols = 3, cmap = None):

    #use to confirm same over different longitudes. CONFIRMED! yay :)

    for i, lon in enumerate(longs):

        rows = int(np.ceil(len(lats)/cols))
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
        plt.figure((i+1))

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(("By latitude, longitude " + str(lon)))
            
        for lat, ax in zip(lats, axs.ravel()):
            az,zen = sunpos(times,lat,lon,0)[:2] #discard RA, dec, H
            elev = 90 - zen

            data = {"Elevation": elev, "Azimuthal": az} 
            df = pd.DataFrame(data=data)

            df_day = df[df.Elevation > 0] # only those during the day! 

            ax.set_title(("Latitude " + str(lat)))

            if cmap is None:
                sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), ax = ax)
            else:
                sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), ax = ax, cmap = cmap)
        
    plt.show()    

def heatmap_vals(lon, lat, times, xy_bins, show = False):

    az,zen = sunpos(times,lat,lon,0)[:2]
    elev = 90 - zen
    
    data = {"Elevation": elev, "Azimuthal": az} 
    df = pd.DataFrame(data=data)
    df_day = df[df.Elevation > 0]


    bin_height = plt.hist2d(df_day["Elevation"], df_day["Azimuthal"], xy_bins)[0]

    if show:
        plt.colorbar()
        plt.show()

    return bin_height

def heatmap_vals_restricted(lat, times, azimuthals, tilts, dir_offset = 0, show = False, lon = 0):

    az,zen = sunpos(times,lat,lon,0)[:2]
    elev = 90 - zen
    
    rotated_azimuths = az - dir_offset
    neg_azimuths = []
    for a in rotated_azimuths:
        if a > 180:
            a -= 360
        neg_azimuths.append(a)

    min_az = np.min(azimuthals)
    max_az = np.max(azimuthals)
    
    data = {"Elevation": elev, "Azimuthal": neg_azimuths} 
    df = pd.DataFrame(data=data)
    df_day = df[df.Elevation > 0]
    df_range = df_day[df_day.Elevation < 60]
    df_range = df_range[df_range.Azimuthal < max_az]
    df_range = df_range[df_range.Azimuthal > min_az]

    bin_height = plt.hist2d(df_range["Azimuthal"], df_range["Elevation"], [len(azimuthals), 4])[0]

    # plt.set_xlabel('Azimuth ($^\\circ$)')
    # plt.set_ylabel('Elevation ($^\\circ$)')

    if show:
        plt.colorbar()
        plt.show()

    # elif not show:
    #     plt.clf()

    return np.array(bin_height)

def heatmap_collection_frac(tilts, azimuthals, fourhst = False, cut_off = True, show = True):

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

        if cut_off:
            y = y[1:-1]

        map.append(y)

    if not cut_off:
        az_labs = ["-70", "-60", "-45", "-30", "-15", "0", "15", "30", "45", "60", "70"]
    elif cut_off: 
        az_labs = ["-60", "-45", "-30", "-15", "0", "15", "30", "45", "60"]
    ti_labs = ["15", "30", "45", "60"]
        
    map = np.array(map)

    if show:
        s = sns.heatmap(map,xticklabels=az_labs, yticklabels=ti_labs)

        s.set_xlabel('Azimuth ($^\\circ$)')
        s.set_ylabel('Elevation ($^\\circ$)')

        # plt.savefig("Image analysis v3/Final graphs/heatmap.png", dpi= 1000)
        #plt.show()

    return map

def heatmap_sim(data, tilts, azimuthals):

    colls = data["collection_fractions"]
    map = []

    az_labs = []
    ti_labs = []

    for i, tilt in enumerate(tilts):
        points = []
        ti_labs.append(str(tilt))

        for azim in azimuthals:
            point = colls[(tilt, np.abs(azim))] * 0.77037571/colls[(45, 0)]
            points.append(point)

            if i == 0:
                az_labs.append(str(azim))
            
        map.append(points)

    # az_labs = ["-70", "-60", "-45", "-30", "-15", "0", "15", "30", "45", "60", "70"]
    # ti_labs = ["15", "30", "45", "60"]
        
    map = np.array(map)
    s = sns.heatmap(map,xticklabels=az_labs, yticklabels=ti_labs)

    s.set_xlabel('Azimuth ($^\\circ$)')
    s.set_ylabel('Elevation ($^\\circ$)')

    # plt.savefig("Image analysis v3/Final graphs/heatmap.png", dpi= 1000)
    #plt.show()

    return map

def latitude_efficiency(map, tilts, azimuthals, longitude, latitude, times, dir_offset = 0):

    #dir_offset 0 for north, 90 for east, 180 for south, 270 for west

    rotated_azimuths = []
    for az in azimuthals:
        r_az = az + dir_offset
        if r_az < 0:
            r_az += 360
        rotated_azimuths.append(r_az)

    hours = heatmap_vals(longitude, latitude, times, [len(tilts), len(azimuthals)])
    efficiency = "NOT DONE YET"
    return efficiency

def save_efficiency_data(offset_directions, latitudes, sim_data):
    map = heatmap_collection_frac(tilts, azimuthals, show = False)
    sim_map = heatmap_sim(data, tilts, azimuthals)

    all_exp = []
    all_sim = []
    for dir_off in offset_directions:
        sums = []
        sim_sums = []

        for lat in tqdm(latitudes):
            hours = heatmap_vals_restricted(lat, times, azimuthals, tilts, dir_offset = dir_off)
            
            efficiency = hours * map.T
            sum = np.sum(efficiency)
            sums.append(sum)

            sim_efficiency = hours * sim_map.T
            sim_sums.append(np.sum(sim_efficiency))

        all_sim.append(sim_sums)
        all_exp.append(sums)

    np.savetxt("Image analysis v3/Efficiency data/Experimental efficiency data.csv", all_exp, delimiter = ',')
    np.savetxt("Image analysis v3/Efficiency data/Ideal simulated efficiency data.csv", all_sim, delimiter = ',')

def plot_efficiency_data(colours, labels, test_latitudes):

    fig = plt.figure(figsize = (8,5))
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    all_exp = np.loadtxt("Image analysis v3/Efficiency data/Experimental efficiency data.csv", delimiter = ",")
    all_sim = np.loadtxt("Image analysis v3/Efficiency data/Ideal simulated efficiency data.csv", delimiter = ',')
    
    all_exp = all_exp/np.max(all_sim)
    all_sim = all_sim/np.max(all_sim)
    
    for i, off in enumerate(all_exp):
        offset = offsets[i]
        label = labels[i]
        plt.plot(test_latitudes, off, label = f"{label} Facing", color = colours[i], linestyle = "solid", linewidth = 4, alpha = 0.8)
        plt.plot(test_latitudes, all_sim[i], color = colours[i], linestyle = "dashed", alpha = 0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Latitude ($^\\circ$)", fontsize = 12)
    plt.ylabel("Efficiency (a.u.)", fontsize = 12)
    plt.savefig("Image analysis v3/report graphs/Performance.png")
    plt.tight_layout()
    plt.show()

start_time = 0 #if you change this from 0 make sure to do the math that steps per day and size work out
# steps_per_day = 47
# step_size = 0.5
bin_number = 100

# steps_per_day = 23
# step_size = 1

steps_per_day = 95
step_size = 0.25

xy_bins = [9, 36] #number of x and y bins respectively

start = datetime(2023, 1, 1, start_time) 
end = datetime(2023, 12, 31, start_time)

lats = np.linspace(-70, 70, 6)
longs = [-180, -90, 0, 90]

tilts = np.arange(15, 75, 15)
azimuthals = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

times = gen_times(start, end, steps_per_day, step_size)

#elev_az_time_heatmap(lon, lat, times, bins = 50, cmap = "flare") 
#latitude_heatmaps(lon, lats, times, bins = bin_number, cols = 3)
#long_and_lat_heatmaps(longs, lats, times, bins = bin_number, cols = 3)

#bin_height = heatmap_vals(lon, lat, times, xy_bins)



# file = "Image analysis v3/sim data/fullrange_idealtilt_25Mrays_last.pkl"
file = "Image analysis v3/sim data/fullrange_idealtilt_25Mrays_last.pkl"
f  = open(file, "rb")
data = pkl.load(f)

# full_tilts = np.arange(0, 65, 5)
# full_azims = np.arange(-70, 75, 5)

#map = heatmap_collection_frac(tilts, azimuthals)
#sim_map = heatmap_sim(data, full_tilts, full_azims)

#heatmap_vals_restricted(lat, times, azimuthals, tilts, show = True, dir_offset= 90)


#testing latitudes
offsets = np.arange(0, 360, 45)
test_latitudes = np.arange(-90, 100, 5)
colours = ["red", "orange", "gold", "green", "#00ccff", "blue", "#ff66ff", "#8a2be2"]
labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

#save_efficiency_data(offsets, test_latitudes, data)
plot_efficiency_data(colours, labels, test_latitudes)

