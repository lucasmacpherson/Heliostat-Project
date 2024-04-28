import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sunposition import sunpos
from datetime import datetime, timezone, timedelta
import seaborn as sns
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from skimage.measure import block_reduce

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

    #sns.set(font_scale=1.4)
    sns.set_theme(style="ticks", font_scale = 1.4)

    az,zen = sunpos(times,lat,lon,0)[:2] #discard RA, dec, H
    elev = 90 - zen

    data = {"Elevation": elev, "Azimuthal": az} 
    df = pd.DataFrame(data=data)

    df_day = df[df.Elevation > 0] # only those during the day! 

    if cmap is None:
        s = sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8, label ="5min increments"))
    else: 
        s = sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8, label ="5min increments"), cmap = cmap)

    s.set(ylabel='Azimuth ($^\\circ$)', xlabel='Elevation ($^\\circ$)')
    # set_xlabel('Azimuth ($^\\circ$)', fontize = 12)
    # s.set_ylabel('Elevation ($^\\circ$)', fontize = 12)

    plt.tight_layout()
    plt.savefig(f"Image analysis v3/Sun pos heat map for {lat} deg", dpi = 1500)
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
            s = sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), ax = ax)
        else:
            s = sns.histplot(df_day, x="Elevation", y="Azimuthal", bins = bins, cbar = True, cbar_kws=dict(shrink=.8), ax = ax, cmap = cmap)

        print(f"Calculated for latitude {iter}/{len(lats)}")
        iter += 1

    s.set_xlabel('Azimuth ($^\\circ$)')
    s.set_ylabel('Elevation ($^\\circ$)')
    
    plt.tight_layout()
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

def heatmap_vals_restricted(lat, times, azimuthals, tilts, dir_offset = 0, show = False, lon = 0, highres = False):

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

    if not highres:
        bin_height = plt.hist2d(df_range["Azimuthal"], df_range["Elevation"], bins = [[-67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5], [7.5, 22.5, 37.5, 52.5, 67.5]])[0]

    elif highres:
        x_bin_edges = np.arange(-60, 70, 5) - 2.5
        y_bin_edges = np.arange(0, 70, 5) -2.5
        np.append(x_bin_edges, x_bin_edges[-1] + 5)
        np.append(y_bin_edges, y_bin_edges[-1] + 5)
        x_bin_edges.tolist()
        y_bin_edges.tolist()

        bin_height = plt.hist2d(df_range["Azimuthal"], df_range["Elevation"], bins = [x_bin_edges, y_bin_edges])[0]
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
        plt.show()

    return map

def heatmap_sim(data, tilts, azimuthals, show = False):

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

    # az_labs = ["-60", "-45", "-30", "-15", "0", "15", "30", "45", "60"]
    # ti_labs = ["15", "30", "45", "60"]
        
    map = np.array(map)
    s = sns.heatmap(map,xticklabels=az_labs, yticklabels=ti_labs)

    if show:
        s.set_xlabel('Azimuth ($^\\circ$)')
        s.set_ylabel('Elevation ($^\\circ$)')

    # plt.savefig("Image analysis v3/Final graphs/heatmap.png", dpi= 1000)
        plt.show()

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

def save_efficiency_data(offset_directions, latitudes, sim_data, fourhst = False):

    tilts = np.arange(15, 75, 15)
    azimuthals = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

    map = heatmap_collection_frac(tilts, azimuthals, show = False, fourhst= fourhst)
    sim_map = heatmap_sim(data, tilts, azimuthals, show = False)

    factor = map[1][4]/sim_map[1][4]
    sim_map = sim_map * factor

    all_exp = []
    all_sim = []
    for dir_off in offset_directions:
        sums = []
        sim_sums = []

        for lat in tqdm(latitudes):
            plt.clf()
            hours = heatmap_vals_restricted(lat, times, azimuthals, tilts, dir_offset = dir_off, show = False)
            
            efficiency = hours * map.T
            sum = np.sum(efficiency)
            sums.append(sum)

            sim_efficiency = hours * sim_map.T
            sim_sums.append(np.sum(sim_efficiency))

        all_sim.append(sim_sums)
        all_exp.append(sums)

    if not fourhst:
        np.savetxt("Image analysis v3/Efficiency data/Experimental efficiency data new.csv", all_exp, delimiter = ',')
    if fourhst:
        np.savetxt("Image analysis v3/Efficiency data/Experimental 4hst efficiency data new.csv", all_exp, delimiter = ',')
    
    #np.savetxt("Image analysis v3/Efficiency data/Ideal simulated efficiency data.csv", all_sim, delimiter = ',')

def highres_efficiency_data(offset_directions, latitudes, sim_data):
    highres_tilts = np.arange(10, 70, 5)
    highres_azimuthals = np.arange(-65, 70, 5)
    sim_map = heatmap_sim(data, highres_tilts, highres_azimuthals)
    print(np.shape(sim_map))
    
    sim_smaller = block_reduce(sim_map, block_size=(3,3), func=np.mean)
    print(np.shape(sim_smaller))

    tilts = np.arange(15, 75, 15)
    azimuthals = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

    all_sim = []
    for dir_off in offset_directions:
        sim_sums = []

        for lat in tqdm(latitudes):
            hours = heatmap_vals_restricted(lat, times, azimuthals, tilts, dir_offset = dir_off)

            sim_efficiency = hours * sim_smaller.T
            sim_sums.append(np.sum(sim_efficiency))

        all_sim.append(sim_sums)

    np.savetxt("Image analysis v3/Efficiency data/High res simulated efficiency data new.csv", all_sim, delimiter = ',')


def plot_efficiency_data(colours, labels, test_latitudes, fourhst = False, highres = False):

    fig = plt.figure(figsize = (8,5))
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    if highres:
        all_sim = np.loadtxt("Image analysis v3/Efficiency data/High res simulated efficiency data.csv", delimiter = ',')
        all_sim *= 1.8
    elif not highres:
        all_sim = np.loadtxt("Image analysis v3/Efficiency data/Ideal simulated efficiency data.csv", delimiter = ',')
    
    if fourhst:
        all_exp = np.loadtxt("Image analysis v3/Efficiency data/Experimental 4hst efficiency data new.csv", delimiter = ",")
        all_sim *= 2.2
    elif not fourhst:
        all_exp = np.loadtxt("Image analysis v3/Efficiency data/Experimental efficiency data new.csv", delimiter = ",")

    # factor = (all_exp[-1][0]/all_sim[-1][0])*1
    # exp_max = np.max(all_exp)
    # all_exp = (all_exp/exp_max)
    # all_sim = (all_sim* 1/np.max(all_sim))
    #all_sim = (all_sim/exp_max)*factor

    sim_max = np.max(all_sim)
    all_exp = (all_exp/sim_max)
    all_sim = (all_sim/sim_max)
    

    sim_latitudes = np.arange(-90, 95, 5)
    
    for i, off in enumerate(all_exp):
        offset = offsets[i]
        label = labels[i]
        plt.plot(test_latitudes, off, label = f"{label} Facing", color = colours[i], linestyle = "solid", linewidth = 4, alpha = 0.8)
        plt.plot(sim_latitudes, all_sim[i], color = colours[i], linestyle = "dashed", alpha = 0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Latitude ($^\\circ$)", fontsize = 12)
    plt.ylabel("Efficiency (a.u.)", fontsize = 12)
    plt.tight_layout()

    lbl = ""
    if highres:
        lbl = "High res"
    if fourhst:
        plt.savefig(f"Image analysis v3/report graphs/Performance 4hst new {lbl}.png", dpi = 1500)
    else:
        plt.savefig(f"Image analysis v3/report graphs/Performance 2hst new {lbl}.png", dpi = 1500)
    plt.show()

start_time = 0 #if you change this from 0 make sure to do the math that steps per day and size work out
# steps_per_day = 47
# step_size = 0.5
bin_number = 100

# steps_per_day = 23
# step_size = 1

# steps_per_day = 95
# step_size = 0.25
steps_per_day = 239
step_size = 0.1

xy_bins = [9, 36] #number of x and y bins respectively

start = datetime(2023, 1, 1, start_time) 
end = datetime(2023, 12, 31, start_time)

lats = np.linspace(-70, 70, 6)
longs = [-180, -90, 0, 90]

tilts = np.arange(15, 75, 15)
azimuthals = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

times = gen_times(start, end, steps_per_day, step_size)

# elev_az_time_heatmap(0, 55, times, bins = 50, cmap = "flare") 
# elev_az_time_heatmap(0, 15, times, bins = 50, cmap = "flare") 
#latitude_heatmaps(0, [55, 14], times, bins = bin_number, cols = 2)
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
# offsets = [0]
test_latitudes = np.arange(-90, 92, 2)
# test_latitudes = np.arange(-90, 95, 5)
colours = ["red", "orange", "gold", "green", "#00ccff", "blue", "#ff66ff", "#8a2be2"]
labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# save_efficiency_data(offsets, test_latitudes, data)
# save_efficiency_data(offsets, test_latitudes, data, fourhst = True)
# print("Done with 4hst data")
# highres_efficiency_data(offsets, test_latitudes, data)
# print("Done with high res data")

plot_efficiency_data(colours, labels, test_latitudes, fourhst = False, highres = False)
