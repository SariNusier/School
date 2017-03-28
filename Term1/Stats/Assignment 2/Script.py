import matplotlib.pyplot as plt
import csv
import numpy as np
import sys
import pandas as pd
import statsmodels.formula.api as sm
import statsmodels.api as sma
from scipy import stats


def read_csvfile(filename):
    toReturn = []
    keys_wanted = ["schedtime","carrier","deptime","origin","weather","delay","date","distance","dayweek"]
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

def toMins(time):
    return (time/100)*60 + (time%100)

def timeDif(sched, dept):
    sched_mins = toMins(sched)
    dept_mins = toMins(dept)
    return dept_mins-sched_mins


def main():
    flights = read_csvfile("FlightDelays.csv")
    delays = np.array([flight["delay"] for flight in flights])
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    schedtimes = np.array([flight["schedtime"] for flight in flights])
    deptimes = np.array([flight["deptime"] for flight in flights])
    distances = np.array([flight["distance"] for flight in flights]).astype(np.int)
    #print np.amax(distances)
    #print np.amin(distances)
    weather = np.array([flight["weather"] for flight in flights])
    dayweek = np.array([flight["dayweek"] for flight in flights])
    weather = np.array([flight["weather"] for flight in flights])

    schedtime_dt = np.array([np.datetime64(convert_dt(flight["date"],flight["schedtime"])) for flight in flights])
    deptime_dt = np.array([np.datetime64(convert_dt(flight["date"],flight["deptime"])) for flight in flights])
    schedtimes_mins = [(time/100)*60 + (time%100) for time in schedtimes.astype(np.int)]


    #print schedtimes_mins
    actdelays =  (deptime_dt - schedtime_dt).astype(int)
    delays_mins  = [timeDif(int(flight["schedtime"]), int(flight["deptime"])) for flight in flights]

    print np.amin(actdelays)
    #flights_df = pd.DataFrame(schedtimes,distances,weather,dayweek,actdelays, columns = ['schedtime','distance','weather','dayweek','delay'])
    flights_df = pd.DataFrame({'schedtime':schedtimes_mins, 'distance':distances, 'weather':weather, 'dayweek':dayweek, 'delay':delays_mins}, dtype = int)
    #print "DELAY RANGE:"
    #print np.amax(actdelays)
    #print np.amin(actdelays)
    #print np.amax(delays_mins)
    #print np.amin(delays_mins)

    flights_df['weather'] = flights_df['weather'].astype('category')
    flights_df['dayweek'] = flights_df['dayweek'].astype('category')
    model = sm.ols(formula = "delay ~ schedtime + distance + weather + dayweek", data = flights_df)
    result = model.fit()
    prediction = result.predict({'schedtime':900,'distance':400,'weather':0,'dayweek':4})
    print "THIS IS PREDICTION"
    print prediction
    #sma.graphics.plot_fit(result, 9)
    #plt.xlim(160,240)
    #plt.ylim(-200,300)
    #plt.savefig('plot%d.png'%i)
    #plt.show()
    print result.summary()
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

    #deptime
    deptimes_unique = np.unique(deptimes)
    deptimes_delays = np.array([np.sum(delays[deptimes == time])
                         for time in deptimes_unique]).astype(float)
    deptimes_delays_total = np.array([len(delays[deptimes == time])
                                for time in deptimes_unique])
    deptimes_delays_perc = (deptimes_delays/deptimes_delays_total)*100
    deptimes_actdelays = np.array([np.sum(actdelays[deptimes == time])
                            for time in deptimes_unique])

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

if __name__ == "__main__":
    main()
