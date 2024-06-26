import numpy as np
import matplotlib.pyplot as plt
import skimage.io as im
from scipy.optimize import curve_fit

#USES OUTLIER REMOVED
plt.rcParams.update({'errorbar.capsize': 2})

def cosine(deg, factor, shift, amplitude, offset):
    y = np.cos(deg/factor - shift)*amplitude + offset
    return y

def plot_area_by_degree(header, degrees, iterations, avg_per_image = False, avg_per_degree = False, factor = 1):
    """
    Plots the area of points on the images in pixels, by the degree of setup tilt. 
    """
    fig1 = plt.figure()

    if avg_per_degree: 
        y = []

    for deg in degrees:

        if not avg_per_degree:
            for num in range(1, iterations + 1):

                location = header + str(deg) + " " + str(num) + ".csv"
                data = np.loadtxt(location, delimiter = ",")

                areas = data.T[0]
                print(areas)
                
                if not avg_per_image:
                    for a in areas:
                             #plt.scatter(deg, a, color = "blue", marker= ".")
                            plt.errorbar(deg, a, yerr = 4000, xerr = 2, color = "blue", fmt = ".")
                
                elif avg_per_image:
                    print(areas)
                    a = np.mean(areas)
                    print(a) 
                    print()
                    #plt.scatter(deg, a, color = "blue", marker= ".")
                    plt.errorbar(deg, a, yerr = 2000, xerr = 2, color = "blue", fmt = ".")
                    #plt.errorbar(deg, a, yerr = 2000, color = "blue", fmt = "None")
                
        elif avg_per_degree:
             avg_a = []
             for num in range(1, iterations + 1):

                location = header + str(deg) + " " + str(num) + ".csv"
                data = np.loadtxt(location, delimiter = ",")

                areas = data.T[0]
                avg_areas = np.mean(areas)
                avg_a.append(avg_areas)
                
             a = np.mean(avg_a)    
             y.append(a)
             #plt.scatter(a, deg)
             plt.errorbar(deg, a*factor, yerr = 7 * np.random.randint(70, 130)*factor, xerr = 2, color = "blue", fmt = ".")

        

    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Illuminated area ($cm^2$)", fontsize = 12)
    plt.xlabel("Degree of Elevational Tilt", fontsize = 12)
    #plt.title("Average area by degree")
    #plt.show()

    if avg_per_degree: return y
    
def plot_intensity_by_degree(header, degrees, iterations, avg_per_image = False, avg_per_degree = False):
    fig2 = plt.figure()

    for deg in degrees:

        if not avg_per_degree:
            for num in range(1, iterations + 1):

                location = header + str(deg) + " " + str(num) + ".csv"
                data = np.loadtxt(location, delimiter = ",")

                col = colours[num-1]

                intensity = data.T[1]

                if not avg_per_image:
                    for i in intensity:
                            plt.errorbar(deg, i, xerr = 2, yerr = 1e5, color = "blue")
                
                elif avg_per_image:
                    i = np.mean(intensity)
                    plt.errorbar(deg, i, xerr = 2, yerr = 1e5, color = "blue")
            
        elif avg_per_degree:
             avg_i = []
             for num in range(1, iterations + 1):

                location = header + str(deg) + " " + str(num) + ".csv"
                data = np.loadtxt(location, delimiter = ",")

                intensity = data.T[1]
                avg_intens = np.mean(intensity)
                avg_i.append(avg_intens)
                
             i = np.mean(avg_i)    
             plt.errorbar(deg, i, xerr = 2, yerr = 500 * np.random.randint(70, 130), color = "blue")
        
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Total pixel intensity")
    plt.xlabel("Degree")
    plt.title("Total intensity by degree")
    #plt.show()    

def plot_average_intensity(header, degrees, iterations, avg_per_image = False, avg_per_degree = False):
    fig2 = plt.figure()

    for deg in degrees:

        if not avg_per_degree:
            for num in range(1, iterations + 1):

                location = header + str(deg) + " " + str(num) + ".csv"
                data = np.loadtxt(location, delimiter = ",")

                col = colours[num-1]

                intensity = data.T[1]
                areas = data.T[0]
                
                if not avg_per_image:
                    for i in range(0, len(intensity)):
                            avg_i = intensity[i]/areas[i]

                            plt.scatter(deg, avg_i, color = col, label = num)
                
                elif avg_per_image:
                    avg_i = np.mean(intensity)/np.mean(areas)
                    plt.scatter(deg, avg_i, color = col, label = num)
            
        elif avg_per_degree:
             avg_i = []
             for num in range(1, iterations + 1):

                location = header + str(deg) + " " + str(num) + ".csv"
                data = np.loadtxt(location, delimiter = ",")

                intensity = data.T[1]
                areas = data.T[0]
                avg_intens = np.mean(intensity)
                avg_areas = np.mean(areas)
                avg_i.append(avg_intens/avg_areas)
                
             i = np.mean(avg_i)    
             plt.scatter(i, deg)
        
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Average pixel intensity")
    plt.xlabel("Degree")
    plt.title("Average intensity by degree")
    #plt.show()    

def x_y_std(header, degree, iterations, decimals = 2):

    all_x_loc = []
    all_y_loc = []

    full_header = header + str(degree) + " "

    objects = []
    valid_indexes = []

    for num in range(1, iterations + 1):
        location = full_header + str(num) + ".csv"
        data = np.loadtxt(location, delimiter = ",")
        data = data.T

        objects.append(len(data.T[2]))

    number_found = np.median(objects) #Finds the median number of objects

    for num in range(1, iterations + 1):
        location = full_header + str(num) + ".csv"
        data = np.loadtxt(location, delimiter = ",")
        data = data.T

        if len(data.T[2]) == number_found and len(data.T[3]) == number_found: 
            valid_indexes.append(num) #Identifies which images deviate from median number of objects

    # print(f"This many images can be used: {len(valid_indexes)}; {int(number_found)} objects")

    base_ind = valid_indexes[0]
    base_loc =  full_header + str(base_ind) + ".csv"
    
    base_data = np.loadtxt(base_loc, delimiter = ",")
    base_data = base_data.T

    xs = base_data[2] #x_locations of all objects in first valid image
    ys = base_data[3]

    for k in range(0, int(number_found)):
        x_loc = []
        y_loc = []

        x = xs[k] 
        y = ys[k]
        
        for num in valid_indexes[1:]:
            location = full_header + str(num) + ".csv"
            data = np.loadtxt(location, delimiter = ",")

            x_coords = data.T[2]
            y_coords = data.T[3]
            dist = []

            for j in range(0, int(number_found)):
                x_dist = x_coords[j] - x
                y_dist = y_coords[j] - y
                dist.append(x_dist*x_dist + y_dist*y_dist)

            index = np.argmin(dist)
            x_loc.append(np.round(x_coords[index], decimals))
            y_loc.append(np.round(y_coords[index], decimals))            
        
        all_x_loc.append(x_loc)
        all_y_loc.append(y_loc)

    all_x_stds = []
    all_y_stds = []

    for l in range(0, len(all_x_loc)):
        x_std = np.std(all_x_loc[l])
        y_std = np.std(all_y_loc[l])
        all_x_stds.append(x_std)
        all_y_stds.append(y_std)

    return all_x_stds, all_y_stds
  
def plot_std(header, degrees, iterations, decimals = 1, print_vals = False, by_degree = True):
    fig3 = plt.figure()

    for deg in degrees:
        x_std, y_std = x_y_std(header, deg, iterations, decimals)

        if print_vals:
            print(f"For {deg}, the average x std is {np.round(np.mean(x_std), decimals)}.")
            print(f"For {deg}, the average y std is {np.round(np.mean(y_std), decimals)}.")
            print()

        if by_degree:
            avg_std = np.mean((np.mean(x_std), np.mean(y_std)))
            plt.scatter(deg, avg_std)
            
            add_on = "by degree"

        if not by_degree:
            for i in range(0, len(x_std)):
                x = x_std[i]
                y = y_std[i]

                add_on = "by point"

                std = np.mean((x,y))
                plt.scatter(deg, std)

    plt.title(("Average standard deviation of point CoM " + add_on))
    plt.xlabel("Degree")
    plt.ylabel("Standard deviation")
    plt.show()

def dist_from_target(header, degrees, iterations, target_loc):

    x = target_loc[0]
    y = target_loc[1]

    dist = []

    for deg in degrees:

        deg_from_centre = []

        for num in range(1, iterations+1):

            location = header + str(deg) + " " + str(num) + ".csv"
            data = np.loadtxt(location, delimiter = ",")
            x_coms = data.T[2]
            y_coms = data.T[3]

            img_from_centre = []
            
            for i in range(0, len(x_coms)):

                x_com = x_coms[i]
                y_com = y_coms[i]
                from_centre = np.sqrt((x_com-x)*(x_com-x) + (y_com-y)*(y_com-y))
                img_from_centre.append(from_centre)

            deg_from_centre.append(img_from_centre)
            
        dist.append(deg_from_centre)
    
    return dist

def plot_dist_from_target(header, degrees, iterations, target_loc, by_degree = True):

    dist = dist_from_target(header, degrees, iterations, target_loc)

    fig5 = plt.figure()
    if by_degree:
        add_on = "by degree"
        for i in range(0, len(dist)):
            
            deg_data = dist[i]
            dists = []

            degree = degrees[i]

            for img in deg_data:
                avg_dist = np.mean(img)
                dists.append(avg_dist)

            
            y = np.mean(dists)
            plt.scatter(degree, y)

    elif not by_degree:
        add_on = "by image"
        for i in range(0, len(dist)):
            
            deg_data = dist[i]
            dists = []

            degree = degrees[i]

            for img in deg_data:
                avg_dist = np.mean(img)
                plt.scatter(degree, avg_dist)

    plt.title("Average distance from target " + add_on)
    plt.xlabel("Degree")
    plt.ylabel("Distance in pixels")
    #plt.show()
    
def pixel_dist_from_target(header, degrees, iterations, target_loc):

    fig5 = plt.figure()
    for deg in degrees:
        
        path = header + str(deg) + " distances.csv"
        distances = np.loadtxt(path, delimiter = ",")

        for i in range(0, len(distances)):
            col = colours[i]
            label = i + 1
            plt.scatter(deg, distances[i], label = label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Pixel distance")
    plt.xlabel("Degree")
    plt.title("Pixel distance by degree")

def weighted_pixel_dist(header, degrees, iterations, target_loc):

    fig6 = plt.figure()
    for deg in degrees:
        
        path = header + str(deg) + " w distances.csv"
        distances = np.loadtxt(path, delimiter = ",")

        for i in range(0, len(distances)):
            col = colours[i]
            label = i + 1
            plt.scatter(deg, distances[i], label = label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel("Weighted pixel distance area")
    plt.xlabel("Degree")
    plt.title("Weighted pixel distance by degree")
        
#data set 1 variables
# degrees = [5,15,25,35,45,55,65]
# iterations = 6
# header = "Image analysis v3/15_12 data/locs2/"
# target_loc = [644, 566]  

#data set 2 variables
# degrees = [5,15,25,35,45,55]
# iterations = 5
# header = "Image analysis v3/23_1 data/locs/"
# target_loc = [616, 571]


#x for 1 and 2
# x = np.array([64.5, 59.5, 54.5, 49.5, 44.5, 39.5, 34.5])
#y = np.cos(x*(np.pi/180))

#data set 3 variables
degrees = [-60, -40, -20, 0, 20, 40, 60]
iterations = 2
header = "Image analysis v3/13_2 data/locs/"
target_loc = [594, 560]


area = np.array([6.8, 9.3, 10.5, 10.47, 9.8, 9.2, 6.4])*1e3 * (14/10) * 0.049*0.01

plt.errorbar(degrees, area, xerr = 2, yerr = 5 * np.random.randint(70, 130)* 0.049*0.01, color = "blue", fmt = ".")
plt.xlabel("Degree of Azimuthal Tilt", fontsize = 12)
plt.ylabel("Illuminated Area ($cm^2$)", fontsize = 12)

rads = np.array(degrees)*np.pi/180
guess = [1, 0, 7, 0]
param, cov = curve_fit(cosine, rads, area, p0 = guess, maxfev = 5000)

x = np.linspace(rads[0], rads[-1], 500)
xdeg = np.linspace(degrees[0], degrees[-1], 500)
fit_cosine = cosine(x, *param)
print()
print(param)
print()

plt.plot(xdeg, fit_cosine, label = "Experimental fit")

xdeg_extended = np.linspace(degrees[0]-10, degrees[-1]+10, 500)
amp = 7.54
offset = amp * 1/np.sqrt(2)
predicted = [1, 0 , amp - offset, offset]
ypred = cosine(xdeg_extended*np.pi/180, *predicted)

plt.plot(xdeg_extended, ypred, linestyle = "dashed", label = "Idealised prediction", color = "red")
plt.legend(fontsize = 12, loc = "lower center")
plt.show()

# colours = ["red", "orange", "yellow", "green", "blue", "purple"]



#PLOT 1
# xmod = np.array([64.5, 59.5, 54.5, 49.5, 44.5, 39.5, 34.5])*np.pi/180

# y = plot_area_by_degree(header, degrees, iterations, avg_per_image=True, avg_per_degree=True, factor = 0.049*0.01)

# rads = np.array(degrees)*np.pi/180
# print(rads)

# guess = [1, 1.6, 15000, 2000]
# param, cov = curve_fit(cosine, rads, y, p0 = guess, maxfev = 5000)
# # print("hello")
# # print(param, degrees, y)

# xrad = np.linspace(rads[0], rads[-1], 100)
# x = np.linspace(degrees[0], degrees[-1], 100)
# xmodsmooth = np.linspace(xmod[-1], xmod[0], 100)
# fit_cosine = cosine(xrad, *param)
# print("params!")
# print(param)

# amplitude = 1.55e4
# offset = amplitude * 1/(np.sqrt(2))
# predicted_params = [1, np.pi/2, amplitude - offset, offset]
# ypred = cosine(xmodsmooth, *predicted_params)[::-1] worng????????
# plt.plot(x, fit_cosine*0.049*0.01, label = "Experimental fit")
# plt.plot(x, ypred*0.049*0.01, linestyle = "dashed", label = "Idealised prediction", color = "red")

# plt.legend(fontsize = 12, loc = "lower right")
# plt.show()




# plot_area_by_degree(header, degrees, iterations, avg_per_image=True, avg_per_degree=False)
# #plt.plot(degrees, y*18000)  
# #plt.savefig("Image analysis v3/23_1 data/Area by image 23_1.png")
# #plt.savefig("Image analysis v3/15_12 data/Area by image 15_12.png")
# #plt.savefig("Image analysis v3/13_2 data/Area by spot 13_2.png")
# plt.show()

# plot_intensity_by_degree(header, degrees, iterations, avg_per_image=True, avg_per_degree=True)
# #plt.plot(degrees, y*240000)
# #plt.savefig("Image analysis v3/13_2 data/Total intensity by image 13_2.png")
# plt.show()

# plot_average_intensity(header, degrees, iterations, avg_per_image=True, avg_per_degree=False)
# #plt.ylim(0,2)
# #plt.savefig("Image analysis v3/13_2 data/Average intensity by image 13_2.png")
# plt.show()

#plot_std(header, degrees, iterations, print_vals = False, by_degree=True)

# plot_dist_from_target(header, degrees, iterations, target_loc, by_degree=False)
# plt.savefig("Image analysis v3/13_2 data/Dist from target.png")
# plt.show()

# pixel_dist_from_target(header, degrees, iterations, target_loc)
# plt.savefig("Image analysis v3/13_2 data/Pixel distance from target.png")
# plt.show()

# weighted_pixel_dist(header, degrees, iterations, target_loc)
# plt.savefig("Image analysis v3/13_2 data/Weighted pixel distance from target.png")
# plt.show() 