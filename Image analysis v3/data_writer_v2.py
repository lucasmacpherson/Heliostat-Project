from edge_detector_v3 import *

# data set 1 variables
# degrees = [5,15,25,35,45,55,65]
# degrees = [45]
# location = "Image analysis v3/Images 15_12/"
# image_labels = [1,2,3,4,5,6]
# date = "15_12"
# xy_range = [[60, 1024], [0, 1280]] # in order y, x
# avg_backs = False
# target_loc = [644, 566]

# data set 2 variables
degrees = [5,15,25,35,45,55]
location = "Image analysis v3/Images 23_1/"
image_labels = [1,2,3,4,5]
date = "23_1"
xy_range = [[90, 1024], [0, 1280]] # in order y, x
avg_backs = True
target_loc = [616, 571]


back_label = "back.jpg"


for deg in degrees:
    distances = []
    w_distances = []

    for num in image_labels:
        img = location + str(deg) + "_" + str(num) + ".jpg"

        back_name = location + str(deg) + "_back" 

        if avg_backs:
            back = avg_background(back_name, 4)

        elif not avg_backs:
            back = im.imread((back_name + ".jpg"))

        title =  date + " data/locs/" + str(deg) + " " + str(num) + ".csv"
        imgtitle = date + " data/figs/" + str(deg) + " " + str(num) + ".png"
        
        image, background = trim(img, back, xy_range[0], xy_range[1])
        image = np.array(image, dtype=np.int16)
        background = np.array(background, dtype=np.int16)

        removed = np.array(image) - np.array(background)

        binary = gen_binary(removed, imgtitle, threshold=10)

        data = obj_data_extractor(binary, image)

        pixel_dist = pixel_dist_from_target(binary, target_loc)
        w_pixel_dist = weighted_pixel_dist(binary, image, target_loc)
        distances.append(pixel_dist)
        w_distances.append(w_pixel_dist)

        header = "areas, intensities, x, y"

        np.savetxt(title, data, header = header, delimiter = ",")

    np.savetxt((title[:-6] + " distances.csv"), distances, delimiter = ",")
    np.savetxt((title[:-6] + " w distances.csv"), w_distances, delimiter = ",")

 