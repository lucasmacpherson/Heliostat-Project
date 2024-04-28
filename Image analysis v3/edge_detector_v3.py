import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import closing, disk
import skimage.io as im

def trim(img, background, y_range, x_range, back_given = True):
    
    y1 = y_range[0]
    y2 = y_range[1]

    x1 = x_range[0]
    x2 = x_range[1]
    
    image = im.imread(img)

    if not back_given:
        b_image = im.imread(background)
    
    elif back_given:
        b_image = background
    
    image = image[y1:y2,x1:x2]
    b_image = b_image[y1:y2,x1:x2]
    
    return image, b_image

def gen_binary(image, imgtitle = "Title not given", savefig = False, threshold = 10, disk_size:int = 5, saveclosing = False):
    """generates and saves binary version of image, 1 for object 0 for background"""

    dim = np.shape(image)
    avg = np.mean(image)

    binary = image.copy()

    for x in range(0, dim[0]):
            for y in range(0, dim[1]):
                if image[x,y] > avg + threshold:
                    binary[x,y] = True
                else: binary[x,y] = False

    footprint = disk(disk_size)
    closed = closing(binary, footprint)

    if saveclosing:
        plt.imshow(binary)
        plt.savefig("Image analysis v3/binary.pdf")
        plt.show()

        plt.imshow(closed)
        plt.savefig("Image analysis v3/closed.pdf")
        plt.show()

    if savefig: 
        plt.figure()
        plt.title(imgtitle[-8:-4])
        plt.imshow(closed)
        plt.savefig(imgtitle)
        #plt.show()
        plt.close()

    return closed

def obj_data_extractor(data, og_image: str): #background already removed!

    objects, num_objs = label(data, return_num=True)
    # plt.figure()
    # im.imshow(objects)
    # plt.show()

    blob_coords = {}
    blob_coords[0] = [] #for all coordinates
    num_valid_objs = 0

    areas = []
    intensities = []
    x_centres = []
    y_centres = []

    total_x_centre = []
    total_y_centre = []

    for i in range(1, num_objs+1):
        area = (objects == i).sum()

        if area > 1000:
            blob_coords[i] = []
            num_valid_objs += 1
            areas.append(area)

    dim = np.shape(objects)

    for x in range(0, dim[0]):
            for y in range(0, dim[1]):
                if objects[x,y] != 0:
                    val = objects[x,y]
                    try:
                        blob_coords[val].append([x,y])
                    except:
                        continue

                    blob_coords[0].append([x,y])
    
    for i in blob_coords.keys():
        if i == 0:
             continue
        
        intensity_vals = []
        x_coords = []
        y_coords = []
        
        for j in blob_coords[i]:
            x = j[0]
            y = j[1]

            # int_val = np.float32(og_image[x,y]) - np.float32(og_background[x,y])
            int_val = np.float32(og_image[x,y])
            intensity_vals.append(int_val)

            x_coords.append(x*int_val)
            y_coords.append(y*int_val)

        total_intensity = np.sum(intensity_vals)
        intensities.append(total_intensity)
        x_centres.append(np.sum(x_coords)/total_intensity) 
        y_centres.append(np.sum(y_coords)/total_intensity)

    info = np.array([areas, intensities, x_centres, y_centres])

    return info.T


def image_data_extractor(binary, removed_image, target_loc): #extracts data for whole image, ignoring object separation
    #removed image has background removed. try without? should make small difference to intensity, otherwise?!!!!

    total_intensity = 0
    pixels_illuminated = 0
    #weighted_pos = []
    weighted_dist = []

    x = target_loc[0]
    y = target_loc[1]

    dim = np.shape(binary)

    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
             
            if binary[i,j]:
                intensity = removed_image[i,j]

                pixels_illuminated += 1
                total_intensity += intensity
                
                x_sep = (i - x)*(i - x)
                y_sep = (j - y)*(j - y)
                weighted_dist.append((np.sqrt((x_sep + y_sep)) * intensity))
                #weighted_pos.append((np.sqrt((i*i + j*j)) * intensity))

    separation = np.sum(weighted_dist)/total_intensity

    mean = np.mean(weighted_dist)

    var_vals = (weighted_dist - mean)**2
    var = np.sum(var_vals)/total_intensity

    return total_intensity, pixels_illuminated, separation, var




def avg_background(name, num_backs, file_type):

    title = name + file_type
    img_1 = im.imread(title)
    total = np.array(img_1.copy())
    used = 0

    for i in range(1, num_backs):
        title = name + "_" + str(i) + file_type

        try:
            img = np.array(im.imread(title))
            total += img
            used += 1
        except:
            continue
        
    avg = total/used
    
    # print(avg[178,635])
    return avg
     
def pixel_dist_from_target(data, target_loc):
    dim = np.shape(data)

    distances = []

    for x in range(0, dim[0]):
            for y in range(0, dim[1]):
                if data[x, y] == 1:
                    dist = np.sqrt((x - target_loc[0])*(x - target_loc[0]) + (y - target_loc[1])*(y - target_loc[1]))
                    distances.append(dist)

    return np.mean(distances)

def weighted_pixel_dist(data, image, target_loc):
    dim = np.shape(data)

    distances = []
    intensities_used = []

    for x in range(0, dim[0]):
            for y in range(0, dim[1]):
                if data[x, y] == 1:
                    dist = np.sqrt((x - target_loc[0])*(x - target_loc[0]) + (y - target_loc[1])*(y - target_loc[1]))
                    dist_weighted = dist * image[x,y]
                    distances.append(dist_weighted)
                    intensities_used.append(image[x,y])
    
    weighted_distance = np.sum(distances)/np.sum(intensities_used)

    return weighted_distance

# avg = avg_background("Images 23_1/55_back", 4)
# back_1 = im.imread("Images 23_1/5_back.jpg")
# back_2 = im.imread("Images 23_1/5_back_1.jpg")
# back_3 = im.imread("Images 23_1/5_back_2.jpg")
# back_4 = im.imread("Images 23_1/5_back_3.jpg")
# back_5 = im.imread("Images 23_1/5_back_4.jpg")

if __name__ == "__main__":
    image = im.imread("Image analysis v3/full data set/15_-30_1.png")
    gen_binary(image, threshold = 15, disk_size = 40, saveclosing= True)


# fig = plt.figure(figsize=(10, 7)) 
# fig.add_subplot(2, 3, 1) 
# plt.title("Average")
# plt.imshow(avg)
# plt.show()

# fig.add_subplot(2, 3, 2) 
# plt.title("1")
# plt.imshow(back_1)

# fig.add_subplot(2, 3, 3) 
# plt.title("2")
# plt.imshow(back_2)

# fig.add_subplot(2, 3, 4) 
# plt.title("3")
# plt.imshow(back_3)

# fig.add_subplot(2, 3, 5) 
# plt.title("4")
# plt.imshow(back_4)

# fig.add_subplot(2, 3, 6) 
# plt.title("5")
# plt.imshow(back_5)
# plt.show()