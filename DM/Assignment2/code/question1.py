import numpy as np
import csv
import pandas as pd

def read_file(pathname):
    try:
        with open(pathname, "r") as fd:
            data = csv.DictReader(fd)
            dict_data = list(data)
            return dict_data
    except IOError as ioe:
        print("IOError: " + str(ioe))


def main():
    # data = np.array(read_file("../data/lastfm.csv"))
    data_frame = pd.read_csv("../data/lastfm.csv")
    print data_frame
    top_three_freq = data_frame["artist"].value_counts()[:3]
    a = data_frame.groupby(['user','artist'])['artist'].count()
    print a
if __name__ == "__main__":
    main()
