import numpy as np
import pandas as pd

DEFAULT_PATH_NAME = '../data/lastfm.csv'


def read_file(pathname):
    try:
        return pd.read_csv(pathname)
    except IOError as ioe:
        """
        Exit program? Read execption handling for this case.
        """
        print ioe


def find_rule_in_df(data_frame, l, r):
    # instances_with_l = data_frame.any('artist' == l)
    users_with_rule = []
    grouped_by_user = data_frame.groupby('user')['artist'].apply(list)
    for u in grouped_by_user.index.values:
        if {l, r}.issubset(grouped_by_user[u]):
            users_with_rule.append(u)
    return users_with_rule
    # containing_L_R = grouped_by_user[(grouped_by_user['artist'].any(l)) & (grouped_by_user['artist'].any(r))]
    # print grouped_by_user


def main():
    data_frame = read_file(DEFAULT_PATH_NAME)
    top_three_freq = data_frame["artist"].value_counts()[:3]
    a = data_frame.groupby(['user', 'artist'])['artist'].count()
    rules = find_rule_in_df(data_frame, 'red hot chili peppers', 'goldfrapp')
    print rules


if __name__ == "__main__":
    main()
