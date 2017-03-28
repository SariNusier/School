import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import stats

def read_csvfile(filename):
    toReturn = []
    with open(filename, "r") as csvFile:
        foods = csv.DictReader(csvFile)
        for food in foods:
            for key in food.keys():
                food[key.strip()] = food.pop(key)
            toReturn.append(food)
    return toReturn

def plotting(keytodraw, countries_arr, keys):
    plt.scatter(range(len(countries_arr[:,keys == keytodraw])),countries_arr[:,keys == keytodraw])
    plt.xlim(-0.3, len(countries_arr[:,keys == keytodraw]))
    plt.ylim(-0.3, 101)
    plt.xlabel("Country")
    plt.ylabel("Value")
    plt.title(keytodraw)
    plt.savefig(("Scatter-%s")%keytodraw)
    plt.close()

def main():
    countries = read_csvfile("protein.csv")
    countries_arr = np.empty((len(countries),len(countries[0].keys())-1))
    country_names_list = [country.pop("Country") for country in countries]
    #country_keys = [key.strip() for key in countries[0].keys()]
    country_keys = ["RedMeat", "WhiteMeat", "Eggs", "Milk", "Fish", "Cereals", "Starch", "Nuts", "Fr&Veg"]
    names = np.array(country_names_list)
    keys = np.array(country_keys)

    for i in range(len(countries)):
        for j in range(len(country_keys)):
            #cur_key = countries[i].keys()[j]
            countries_arr[i][j] = countries[i][country_keys[j]]

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    #keytodraw = "Fish"
    veg = countries_arr[:, keys == "Fr&Veg"]
    print veg
    print countries_arr
    plt.boxplot(countries_arr)
    plt.xticks(range(1,len(country_keys)+1),country_keys)
    plt.ylabel("Consumption")
    plt.xlabel("Food")
    plt.show()
    '''
    for key in keys:
        #plotting(key, countries_arr, keys)
        if key != "Fr&Veg":
            cur = countries_arr[:, keys == key]
            print
            print(("%s\nMean ratio: %f")%(key, np.mean(cur/veg)))
            plt.scatter(range(len(cur/veg)),cur/veg)
            plt.show()
            print(("Mean difference: %f")%(np.mean((np.absolute(cur-veg)))))
            print(("Std ratio: %f")%(np.std(cur/veg)))
            print(("Std difference: %f")%(np.std((np.absolute(cur-veg)))))

    #redmeat = countries_arr[:, keys == "RedMeat"]
    colors = ['#f45942','#f4a142','#f4cb42','#f1f442','#b6f442','#6bf442','#42f474','#42eef4']
    cur_color = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    '''
    x = len(countries_arr[:,keys == "Fr&Veg"])
    '''
    for key in keys:
        if key != "Fr&Veg":
            ax.plot(range(x), countries_arr[:,keys == key], c=colors[cur_color], label = key)
            cur_color += 1
    '''
    '''
    plt.plot(range(x), countries_arr[:,keys == "Fr&Veg"], c="k", label = "Fr&Veg")
    plt.legend(loc = 'upper left')
    plt.show()
    plt.boxplot(countries_arr,vert=True)
    plt.xlim(-1, x+1)
    plt.xticks(range(x), keys)
    plt.show()
    '''
    #print(redmeat.T)
    #print(veg.T)
    #print("Mean:")
    #print(np.mean(redmeat/veg))
    #print(np.std(redmeat/veg))
    #print()
    '''
    print(countries_arr)
    print(keys)
    print("Mean:")
    print(countries_arr.mean(axis = 0))
    print("Median:")
    print(np.median(countries_arr, axis = 0))
    print("Variance:")
    print(np.var(countries_arr, axis = 0))
    print("Standard deviation:")
    print(np.std(countries_arr, axis = 0))
    print("Range:")
    print(np.amax(countries_arr,axis=0) - np.amin(countries_arr,axis=0))
    print("Interquartile range:")
    q75, q25 = np.percentile(countries_arr, [75 ,25], axis = 0)
    print(q75-q25)
    '''
if __name__ == "__main__":
    main()
