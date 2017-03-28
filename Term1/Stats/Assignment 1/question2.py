import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
from scipy import stats

def read_csvfile(filename):
    toReturn = []
    keys_wanted = ["schedtime","carrier","deptime","origin","weather","delay","date"]
    with open(filename, "r") as csvFile:
        flights = csv.DictReader(csvFile)
        for flight in flights:
            for key in flight.keys():
                if key not in keys_wanted:
                    del flight[key]
                elif key == "delay":
                    if flight["delay"] == "ontime":
                        flight["delay"] = 0
                    else:
                        flight["delay"] = 1
            toReturn.append(flight)
    return toReturn

def convert_dt(date, time):
    ssplit = date.split("/")
    year = ssplit[2]
    month = ssplit[0]
    day = ssplit[1]
    minute = time[-2:]
    hour = time[:-2]
    if len(month) == 1:
        month = "0"+month
    if len(day) == 1:
        day = "0"+day
    if len(hour) == 1:
        hour = "0"+hour
    if len(hour) == 0:
        hour = "00"
    return "%s-%s-%sT%s:%s"%(year,month,day,hour,minute)

def main():
    flights = read_csvfile("FlightDelays.csv")
    delays = np.array([flight["delay"] for flight in flights])
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    schedtimes = np.array([flight["schedtime"] for flight in flights])
    deptimes = np.array([flight["deptime"] for flight in flights])

    schedtime_dt = np.array([np.datetime64(convert_dt(flight["date"],flight["schedtime"])) for flight in flights])
    deptime_dt = np.array([np.datetime64(convert_dt(flight["date"],flight["deptime"])) for flight in flights])
    actdelays =  (deptime_dt - schedtime_dt).astype(int)
    # - schedtimes

    #Schedtime
    schedtimes_unique = np.unique(schedtimes)
    schedtimes_delays = np.array([np.sum(delays[schedtimes == time])
                         for time in schedtimes_unique]).astype(float)
    schedtimes_delays_total = np.array([len(delays[schedtimes == time])
                                for time in schedtimes_unique])
    schedtimes_delays_perc = (schedtimes_delays/schedtimes_delays_total)*100

    schedtimes_actdelays = np.array([np.sum(actdelays[schedtimes == time])
                               for time in schedtimes_unique])
    #print(schedtimes_actdelays)
    #fit = np.polyfit(schedtimes_unique,schedtimes_actdelays,1)
    #fit_fn = np.poly1d(fit)
    #plt.plot(schedtimes_unique,schedtimes_actdelays, 'yo', schedtimes_unique, fit_fn(schedtimes_unique), '--k')
    #plt.bar(schedtimes_unique, schedtimes_delays)
    #plt.show()
    #delays_schedtimes = np.array(schedtimes_delays_total-schedtimes_delays)
    #plt.boxplot(np.row_stack((delays_schedtimes, schedtimes_delays)))
    #plt.show()
    #plt.boxplot([0,1], schedtimes_delays)
    """
    plt.scatter(schedtimes_unique, schedtimes_actdelays)
    plt.title("Actual delays")
    plt.xlabel("Schedtime")
    plt.ylabel("Total minutes")
    #plt.xlim(500,2400)
    #plt.ylim(-0.5,50)
    plt.savefig("2c-Schedtime")
    plt.close()
    """
    """
    plt.scatter(schedtimes_unique, schedtimes_delays)
    plt.title("Total delays")
    plt.xlabel("Schedtime")
    plt.ylabel("Number of delays")
    plt.xlim(500,2400)
    plt.ylim(-0.5,50)
    plt.savefig("2b-Schedtime-1")
    plt.close()

    plt.scatter(schedtimes_unique, schedtimes_delays_perc)
    plt.title("Delays percentage")
    plt.xlabel("Schedtime")
    plt.ylabel("Percentage of delays")
    plt.xlim(500,2400)
    plt.ylim(-0.5,110)
    plt.savefig("2b-Schedtime-2")
    plt.close()
    """
    #deptime
    deptimes_unique = np.unique(deptimes)
    deptimes_delays = np.array([np.sum(delays[deptimes == time])
                         for time in deptimes_unique]).astype(float)
    deptimes_delays_total = np.array([len(delays[deptimes == time])
                                for time in deptimes_unique])
    deptimes_delays_perc = (deptimes_delays/deptimes_delays_total)*100
    deptimes_actdelays = np.array([np.sum(actdelays[deptimes == time])
                            for time in deptimes_unique])
    """
    plt.scatter(deptimes_unique, deptimes_actdelays)
    plt.title("Actual delays")
    plt.xlabel("Deptime")
    plt.ylabel("Total minutes")
    #plt.xlim(500,2400)
    #plt.ylim(-0.5,50)
    plt.savefig("2c-Deptime")
    plt.close()
    """
    #print(deptimes_actdelays)
    #print(deptimes_unique)
    #.scatter(deptimes_unique, deptimes_actdelays)
    #plt.show()
    #print(np.median(deptimes_delays_perc))
    #plt.plot(deptimes_unique, deptimes_delays)
    #plt.xlim(-1,2500)
    #plt.title("Total delays")
    #plt.show()
    #plt.plot(deptimes_unique, deptimes_delays_perc)
    #plt.show()
    """
    plt.scatter(deptimes_unique, deptimes_delays)
    plt.title("Total delays")
    plt.xlabel("Deptime")
    plt.ylabel("Number of delays")
    plt.xlim(500,2400)
    plt.ylim(-0.5,10)
    plt.savefig("2b-Deptime-1")
    plt.close

    plt.scatter(deptimes_unique, deptimes_delays_perc)
    plt.title("Delays percentage")
    plt.xlabel("Deptime")
    plt.ylabel("Percentage of delays")
    plt.xlim(500,2400)
    plt.ylim(-0.5,110)
    plt.savefig("2b-Deptime-2")
    plt.close()
    """

    #carrier
    carriers = np.array([flight["carrier"] for flight in flights])
    carriers_unique = np.unique(carriers)
    carriers_delays = np.array([np.sum(delays[carriers == carrier])
                         for carrier in carriers_unique]).astype(float)
    carriers_delays_total = np.array([len(delays[carriers == carrier])
                                     for carrier in carriers_unique])
    carriers_delays_perc = (carriers_delays/carriers_delays_total)*100

    #print(carriers)
    #print(carriers_unique)
    #print(carriers_delays)
    #print(carriers_delays_perc)
    """
    plt.bar(range(len(carriers_unique)), carriers_delays,align="center")
    plt.title("Total delays")
    plt.xlabel("Carrier")
    plt.ylabel("Number of delays")
    plt.xticks(range(len(carriers_unique)), carriers_unique)
    plt.savefig("2b-Carrier-1")
    plt.close()

    plt.title("Delays percentage")
    plt.xlabel("Carrier")
    plt.ylabel("Percentage of delays")
    plt.bar(range(len(carriers_unique)), carriers_delays_perc,align="center")
    plt.xticks(range(len(carriers_unique)), carriers_unique)
    plt.savefig("2b-Carrier-2")
    plt.close()
    """
    #print(carriers_delays)
    #print(carriers_delays.shape)
    #plt.xlim(-1,2)
    #plt.boxplot(carriers_delays)
    #plt.show()

    #origin
    origins = np.array([flight["origin"] for flight in flights])
    origins_unique = np.unique(origins)
    origins_delays = np.array([np.sum(delays[origins == origin])
                         for origin in origins_unique]).astype(float)
    origins_delays_total = np.array([len(delays[origins == origin])
                                     for origin in origins_unique])
    origins_delays_perc = (origins_delays/origins_delays_total)*100

    #print(origins)
    #print(origins_unique)
    #print(origins_delays)
    #plt.bar(range(len(origins_unique)), origins_delays)
    #plt.show()
    #plt.bar(range(len(origins_unique)), origins_delays_perc)
    #plt.show()
    """
    plt.bar(range(len(origins_unique)), origins_delays,align="center")
    plt.title("Total delays")
    plt.xlabel("Origin")
    plt.ylabel("Number of delays")
    plt.xticks(range(len(origins_unique)), origins_unique)
    plt.savefig("2b-Origin-1")
    plt.close()

    plt.title("Delays percentage")
    plt.xlabel("Origin")
    plt.ylabel("Percentage of delays")
    plt.bar(range(len(origins_unique)), origins_delays_perc,align="center")
    plt.xticks(range(len(origins_unique)), origins_unique)
    plt.savefig("2b-Origin-2")
    plt.close()
    """
    #weather
    weathers = np.array([int(flight["weather"]) for flight in flights])
    weathers_unique = np.unique(weathers)
    weathers_delays = np.array([np.sum(delays[weathers == weather])
                         for weather in weathers_unique]).astype(float)
    weathers_delays_total = np.array([len(delays[weathers == weather])
                                     for weather in weathers_unique])
    weathers_delays_perc = (weathers_delays/weathers_delays_total)*100
    weathers_actdelays = np.array([np.sum(actdelays[weathers == weather])
                            for weather in weathers_unique])

    q75, q25 = np.percentile(actdelays[weathers == 1], [75 ,25], axis = 0)
    print(q75-q25)
#    print(np.std(actdelays[weathers == 1]))

#    print(np.median(actdelays[weathers == 0]))
#    print(np.std(actdelays[weathers == 0]))
    """
    plt.bar(range(len(weathers_unique)), weathers_actdelays,align="center")
    plt.title("Actual delays")
    plt.xlabel("Weather")
    plt.ylabel("Minutes")
    plt.xticks(range(len(weathers_unique)), weathers_unique)
    plt.savefig("2c-Weather-1")
    plt.close()

    plt.title("Average minutes per delayed flight")
    plt.xlabel("Weather")
    plt.ylabel("Minutes")
    plt.bar(range(len(weathers_unique)), weathers_actdelays/weathers_delays,align="center")
    plt.xticks(range(len(weathers_unique)), weathers_unique)
    plt.savefig("2c-Weather-2")
    #lt.close()
    #plt.bar(range(len(weathers_unique)), weathers_delays)
    #plt.show()
    #plt.bar(range(len(weathers_unique)), weathers_delays_perc)
    #plt.show()
    """
    """
    plt.bar(range(len(weathers_unique)), weathers_delays,align="center")
    plt.title("Total delays")
    plt.xlabel("Weather")
    plt.ylabel("Number of delays")
    plt.xticks(range(len(weathers_unique)), weathers_unique)
    plt.savefig("2b-Weather-1")
    plt.close()

    plt.title("Delays percentage")
    plt.xlabel("Weather")
    plt.ylabel("Percentage of delays")
    plt.bar(range(len(weathers_unique)), weathers_delays_perc,align="center")
    plt.xticks(range(len(weathers_unique)), weathers_unique)
    plt.savefig("2b-Weather-2")
    plt.close()
    """




if __name__ == "__main__":
    main()
