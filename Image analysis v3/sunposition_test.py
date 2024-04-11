import numpy as np
import matplotlib.pyplot as plt
from sunposition import sunpos
from datetime import datetime, timezone

#evaluate on a 2 degree grid
lon  = np.linspace(-180,180,181)
lat = np.linspace(-90,90,91)
LON, LAT = np.meshgrid(lon,lat)
#at the current time
now = datetime.now(timezone.utc)
az,zen = sunpos(now,LAT,LON,0)[:2] #discard RA, dec, H
#convert zenith to elevation
elev = 90 - zen

# az_single, zen_single = sunpos(now, 40, 30, 0)[:2]
# print(az_single, 90- zen_single)

#convert azimuth to vectors
#u, v = np.cos((90-az)*np.pi/180), np.sin((90-az)*np.pi/180)
#plot
fig, ax = plt.subplots(figsize=(6,3),layout='constrained')
img = ax.imshow(elev,cmap=plt.cm.CMRmap,origin='lower',vmin=-90,vmax=90,extent=(-181,181,-91,91))
s = slice(5,-1,5) # equivalent to 5:-1:5
#ax.quiver(lon[s],lat[s],u[s,s],v[s,s],pivot='mid',scale_units='xy')

ax.contour(lon,lat,elev,[0])
ax.set_aspect('equal')
ax.set_xticks(np.arange(-180,181,45))
ax.set_yticks(np.arange(-90,91,45))
ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Latitude (deg)')
cb = plt.colorbar(img,ax=ax,shrink=0.8,pad=0.03)
cb.set_label('Sun Elevation (deg)')
plt.show()