# from edge_detector_v3 import *
# from LED_analyser import quick_plot
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as im
from skimage.measure import block_reduce

# test = im.imread("Image analysis v3/4 heliostat 8_04/60_70_0.png")
# plt.imshow(test, norm = "linear")
# plt.show()

# a = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

# print(a[len(a)//2])

# file = "Image analysis v3/sim data/fullrange_idealtilt_25Mrays_last.pkl"
# f  = open(file, "rb")
# data = pkl.load(f)

# print(data["collection_fractions"].keys())

print(np.random.randint(-10,10, size = 11)*0.01 + 1)

# N_points = 1000
# x = np.random.randn(N_points) 
# y = 4 * x + np.random.randn(1000) + 50
   
# bin_height = plt.hist2d(x, y, bins = [[-67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5], [7.5, 22.5, 37.5, 52.5, 67.5]])

# print(bin_height)
# plt.show()


# colours = ["red", "orange", "green", "blue"]
# with open('raytracer_data-18.03/all_simages_25Mrays_uniform.pkl', 'rb') as f:
#     data = pickle.load(f)

# mirr_nums = np.loadtxt("sim_mirr_numbs.csv", delimiter=",")
# # print(mirr_nums[1][1])

# tilts = np.arange(15, 75, 15)
# azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

# for i, tilt in enumerate(tilts):
#     for j, azim in enumerate(azimuthals):
#         mirr_num = mirr_nums[j+1][i+1]
#         print(mirr_num)
#         y = data[tilt, azim] * 4/mirr_num

#         if mirr_num == 4:
#             plt.scatter(azim, data[tilt, azim], label = str(tilt) + " unscaled", color = colours[i])

#         else:
#             plt.scatter(azim, data[tilt, azim], label = str(tilt) + " unscaled", marker = "x", color = colours[i])
#             plt.scatter(azim, y, label = str(tilt) + " scaled", color = colours[i])

# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# #plt.ylim(top = 0.0005)
# plt.legend(by_label.values(), by_label.keys())

# plt.show()

# fig1 = plt.figure()
# colours = ["red", "orange", "green", "blue"]

# norms = [690122.0, 1066143.7, 1078526.0, 1277268.3]
# incident_angles = np.loadtxt("simple_incident_angles.csv", delimiter = ",")


# for i, tilt in enumerate(tilts):
#     cos = np.cos(incident_angles[i+1][1:]*np.pi/360)
#     quick_plot(tilt, azimuthals, cos, norms[i])

#     if data[tilt, 0] != 0:
#         norm = norms[i] * 1/data[tilt, 0] 
    
#     else: norm = norms[i]

#     for azim in azimuthals:

#         col = colours[i]

#         collection = data[tilt, azim] * norm

#         plt.scatter(azim, collection, color = col, label = str(tilt))


#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys())

#     plt.title((str(tilt) + " Tilt"))
#     plt.show()


# data = np.loadtxt("used_incident_angles.csv", delimiter="," )

# print(data[1][1:])
 
# print(len(data))
# colours = ["red", "orange", "green", "blue"]
# plt.figure(figsize=(8, 6))

# for i in range(1, len(data)):
#     plt.scatter(data[0][1:], data[i][1:]/2, label = (str(int(data[i][0])) + "$^\circ$ elevational tilt"), color = colours[i-1])

# plt.legend(loc = "lower center")
# plt.ylim(bottom = 25)
# plt.xlabel("Azimuthal tilt ($^\circ$)", fontsize = 12)
# plt.ylabel("Resultant Angle ($^\circ$)", fontsize = 12)

# plt.savefig("Image analysis v3/Final graphs/Angle demonstrator.png", dpi = 1500)
# plt.show()


# # data set 1 variables
# # degrees = [5,15,25,35,45,55]
# # location = "Images 23_1/"
# # image_labels = [1,2,3,4,5]
# # date = "23_1"
# # xy_range = [[90, 1024], [0, 1280]] # in order y, x
# # avg_backs = True
# # target_loc = [616, 571]

# #data set 2 variables
# degrees = [5,15,25,35,45,55]
# location = "Images 23_1/"
# # image_labels = [1,2,3,4,5]
# image_labels = [1]
# date = "23_1"
# xy_range = [[90, 1024], [0, 1280]] # in order y, x
# avg_backs = False
# target_loc = [616, 571]

# back_label = "back.jpg"


# plt.close()
# size = 12

# img = "Image analysis v3/Images 13_2/45_20_2.png"

# image = im.imread(img)[150:, 0:1140]
# print(image.shape)

# #plt.figure(figsize = [50,80])
# plt.imshow(image, norm = "linear")
# cbar = plt.colorbar()

# cbar.ax.tick_params(labelsize=size)
# plt.xticks(fontsize = size)
# plt.yticks(fontsize = size)

# plt.show()

# def cosine(deg, factor, shift, amplitude, offset):
#     y = np.cos(deg/factor - shift)*amplitude + offset
#     return y


# degrees = [5,15,25,35,45,55,65]
# rads = np.array(degrees)*np.pi/180

# xrad = np.linspace(rads[0], rads[-1], 100)
# x = np.linspace(degrees[0], degrees[-1], 100)

# predicted_params = [1, np.pi/2, 15000, 10000]
# ypred = cosine(xrad, *predicted_params)
# plt.plot(x, ypred, label = "Theoretical prediction")
# print(xrad)

# plt.show()
