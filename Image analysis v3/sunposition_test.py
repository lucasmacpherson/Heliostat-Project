import numpy as np
import matplotlib.pyplot as plt
from sunposition import sunpos
from datetime import datetime, timezone, timedelta
import seaborn as sns
import pickle as pkl

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

def heatmap_collection_frac(tilts, azimuthals):

    map = []
    for tilt in tilts:
        data = np.loadtxt(("Image analysis v3/" + str(tilt) + " norm data.csv"), delimiter = ",") 
        data = data.T
        intensities = data[1]/1.4e6
        y = np.mean(intensities.reshape(-1, 3), axis = 1)
        print(tilt)
        print(y)
        map.append(y)

    az_labs = ["-70", "-60", "-45", "-30", "-15", "0", "15", "30", "45", "60", "70"]
    ti_labs = ["15", "30", "45", "60"]
        
    map = np.array(map)
    s = sns.heatmap(map,xticklabels=az_labs, yticklabels=ti_labs)

    s.set_xlabel('Azimuth')
    s.set_ylabel('Elevation')

    # plt.savefig("Image analysis v3/Final graphs/heatmap.png", dpi= 1000)
    plt.show()

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

    s.set_xlabel('Azimuth')
    s.set_ylabel('Elevation')

    # plt.savefig("Image analysis v3/Final graphs/heatmap.png", dpi= 1000)
    plt.show()

    return map

lon = 40
lat = 0
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
azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

times = gen_times(start, end, steps_per_day, step_size)

#elev_az_time_heatmap(lon, lat, times, bins = 50, cmap = "flare") 
#latitude_heatmaps(lon, lats, times, bins = bin_number, cols = 3)
#long_and_lat_heatmaps(longs, lats, times, bins = bin_number, cols = 3)

#bin_height = heatmap_vals(lon, lat, times, xy_bins)



# file = "Image analysis v3/sim data/fullrange_idealtilt_25Mrays_last.pkl"
file = "Image analysis v3/sim data/fullrange_idealtilt_25Mrays_last.pkl"


f  = open(file, "rb")
data = pkl.load(f)

full_tilts = np.arange(0, 70, 5)
full_azims = np.arange(0, 75, 5)

#map = heatmap_collection_frac(tilts, azimuthals)
sim_map = heatmap_sim(data, full_tilts, full_azims)


