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

def target_image_smoothed(image_shape, model, smoothscale):
    receiver_pos = model['receiver_position']
    model = prune_rays(model)
    rays = get_rays_at_target(model)