from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from experimental_params import experimental_params as exp
from raytracer import *

def position_to_pixels(pos, receiver_pos=exp.RECEIVER_POSITION.value, image_size=exp.CAMERA_IMAGESIZE.value, pixel_to_mm=exp.CAMERA_PIXEL_TO_MM.value):
    pos = pos - receiver_pos
    pxl_pos = pos * 1000 * pixel_to_mm # metres to pixel conversion
    pxl_pos = pxl_pos + np.array((image_size[0]/2, image_size[1]/2, 0))
    return pxl_pos

def target_image_points(image_shape, model, pointsize):
    receiver_pos = model['receiver_position']
    model = prune_rays(model)
    rays = get_rays_at_target(model)

    img = Image.new("RGB", image_shape, "black")
    draw = ImageDraw.Draw(img)
    # draw.ellipse((100, 100, 150, 200), fill=(255, 0, 0), outline=(0, 0, 0))

    for ray in rays:
        pos = position_to_pixels(ray[-1][0], receiver_pos)
        
        draw.ellipse((pos[0]-pointsize, pos[1]-pointsize, pos[0]+pointsize, pos[1]+pointsize),
        fill=(255, 255, 255))

    return img

def target_image_hist(image_shape, model):
    receiver_pos = model['receiver_position']
    receiver_size = model['receiver_size']
    model = prune_rays(model)
    rays = get_rays_at_target(model)

    pixels = []
    for ray in rays:
        pixels.append(position_to_pixels(ray[-1][0], receiver_pos, receiver_size))
    pixels = np.array(pixels)

    hist, xedges, yedges = np.histogram2d(pixels[:, 0], pixels[:, 1], 
                                          bins=image_shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], origin='lower', cmap='viridis')

    plt.title('2D Histogram of rays at Target Plane')
    plt.xlabel('x-pixels')
    plt.ylabel('y-pixels')

    return fig, ax
