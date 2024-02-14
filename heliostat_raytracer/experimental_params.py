from enum import Enum

class experimental_params(Enum):
    # All values given in metres
    HELIOSTAT_SEPERATION = 0.452
    HELIOSTAT_WIDTH = 0.57
    MIRROR_RADIUS = 0.013
    RECEIVER_POSITION = (-0.365, 0, 0.585)
    RECEIVER_SIZE = (0.23, 0.28)
    YLIM = (-1, 2)
    CAMERA_IMAGESIZE = (1280, 1080)
    CAMERA_PIXEL_TO_MM = 4.5
