from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from experimental_params import experimental_params as exp
from raytracer import *

def target_image_points(image_shape, model, pointsize):
    receiver_pos = model['receiver_position']
    model = prune_rays(model)
    rays = get_rays_at_target(model)

    img = Image.new("RGB", image_shape, "black")
    draw = ImageDraw.Draw(img)
    # draw.ellipse((100, 100, 150, 200), fill=(255, 0, 0), outline=(0, 0, 0))

    for ray in rays:
        pos = ray[-1][0] - receiver_pos
        pos = pos * 1000 * exp.CAMERA_PIXEL_TO_MM.value # metres to pixel conversion
        pos = pos + np.array((exp.CAMERA_IMAGESIZE.value[0]/2, exp.CAMERA_IMAGESIZE.value[1]/2, 0))
        
        draw.ellipse((pos[0]-pointsize, pos[1]-pointsize, pos[0]+pointsize, pos[1]+pointsize),
        fill=(255, 255, 255))

    return img

def target_image_hist(image_shape, model):
    receiver_pos = model['receiver_position']
    model = prune_rays(model)
    rays = get_rays_at_target(model)

    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(rays[:][-1][0][0], rays[:][-1][0][1], bins=image_shape)

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Display the 2D histogram as an image
    ax.imshow(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], origin='lower', cmap='viridis')

    # Add a colorbar for reference
    cbar = fig.colorbar()
    cbar.set_label('Intensity')

    # Customize the plot if needed
    plt.title('2D Histogram of rays at Target Plane')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')

    return fig, ax
