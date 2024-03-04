from edge_detector_v3 import *

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

# deg = 5

# for num in image_labels:
#     img = location + str(deg) + "_" + str(num) + ".jpg"

#     back_name = location + str(deg) + "_back" 

#     if avg_backs:
#         back = avg_background(back_name, 4)

#     elif not avg_backs:
#         back = im.imread((back_name + ".jpg"))

#     title =  date + " data/locs/" + str(deg) + " " + str(num) + ".csv"
#     imgtitle = date + " data/figs/" + str(deg) + " " + str(num) + ".png"

#     image, background = trim(img, back, xy_range[0], xy_range[1])

#     image = np.array(image, dtype=np.int16)
#     background = np.array(background, dtype=np.int16)
#     removed = image - background

#     fig2 = plt.figure()

#     plt.subplot(1,3,1)
#     plt.imshow(background)
#     plt.title("Background image for " + str(deg) + ", iteration " + str(num))

#     plt.subplot(1,3,2)
#     plt.imshow(image)
#     plt.title("Image")

#     plt.subplot(1,3,3)
#     plt.imshow(removed)
#     plt.title("Removed")
    

#     print(background[0,0])
#     print(image[0,0])
#     print(removed[0,0])

#     # print(np.shape(background))
#     # print(np.shape(image))
#     # print(np.shape(removed))

#     plt.show()
#     binary = gen_binary(removed, imgtitle, threshold=10)
#     #plt.show()

#874/1140

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

def cosine(deg, factor, shift, amplitude, offset):
    y = np.cos(deg/factor - shift)*amplitude + offset
    return y


degrees = [5,15,25,35,45,55,65]
rads = np.array(degrees)*np.pi/180

xrad = np.linspace(rads[0], rads[-1], 100)
x = np.linspace(degrees[0], degrees[-1], 100)

predicted_params = [1, np.pi/2, 15000, 10000]
ypred = cosine(xrad, *predicted_params)
plt.plot(x, ypred, label = "Theoretical prediction")
print(xrad)

plt.show()