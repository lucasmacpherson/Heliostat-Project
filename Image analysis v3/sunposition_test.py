import numpy as np
import matplotlib.pyplot as plt
from sunposition import sunpos
from datetime import datetime, timezone, timedelta
import seaborn as sns
import pandas as pd

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

times = gen_times(start, end, steps_per_day, step_size)

elev_az_time_heatmap(lon, lat, times, bins = bin_number, cmap = "flare") 
#latitude_heatmaps(lon, lats, times, bins = bin_number, cols = 3)
#long_and_lat_heatmaps(longs, lats, times, bins = bin_number, cols = 3)

#bin_height = heatmap_vals(lon, lat, times, xy_bins)





# #evaluate on a 2 degree grid
# lon  = np.linspace(-180,180,181)
# lat = np.linspace(-90,90,91)
# LON, LAT = np.meshgrid(lon,lat)
# #at the current time
# now = datetime.now(timezone.utc)
# az,zen = sunpos(now,LAT,LON,0)[:2] #discard RA, dec, H
# #convert zenith to elevation
# elev = 90 - zen

# az_single, zen_single = sunpos(now, 40, 30, 0)[:2]
# print(az_single, 90- zen_single)

# #convert azimuth to vectors
# #u, v = np.cos((90-az)*np.pi/180), np.sin((90-az)*np.pi/180)
# #plot
# fig, ax = plt.subplots(figsize=(6,3),layout='constrained')
# img = ax.imshow(elev,cmap=plt.cm.CMRmap,origin='lower',vmin=-90,vmax=90,extent=(-181,181,-91,91))
# s = slice(5,-1,5) # equivalent to 5:-1:5
# #ax.quiver(lon[s],lat[s],u[s,s],v[s,s],pivot='mid',scale_units='xy')

# ax.contour(lon,lat,elev,[0])
# ax.set_aspect('equal')
# ax.set_xticks(np.arange(-180,181,45))
# ax.set_yticks(np.arange(-90,91,45))
# ax.set_xlabel('Longitude (deg)')
# ax.set_ylabel('Latitude (deg)')
# cb = plt.colorbar(img,ax=ax,shrink=0.8,pad=0.03)
# cb.set_label('Sun Elevation (deg)')
# plt.show()