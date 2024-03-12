from edge_detector_v3 import *
import pickle


with open('Image analysis v3/16Mrays_last.pkl', 'rb') as f:
    data = pickle.load(f)

fig1 = plt.figure()
colours = ["red", "orange", "green", "blue"]
tilts = np.arange(15, 75, 15)
azimuthals = [-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70]

for i, tilt in enumerate(tilts):
    for azim in azimuthals:

        col = colours[i]

        collection = data[tilt, azim]

        plt.scatter(azim, collection, color = col, label = str(tilt))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

plt.show()



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
