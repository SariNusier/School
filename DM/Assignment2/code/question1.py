import pandas as pd

DEFAULT_PATH_NAME = '../data/lastfm.csv'


def read_file(pathname):
    try:
        return pd.read_csv(pathname)
    except IOError as ioe:
        print(ioe)


def find_rule_in_df(data_frame, l, r):
    users_with_rule = []
    grouped_by_user = data_frame.groupby('user')['artist'].apply(list)
    for u in grouped_by_user.index.values:
        if {l, r}.issubset(grouped_by_user[u]):
            users_with_rule.append(u)
    return users_with_rule, len(users_with_rule)


def main():
    data_frame = read_file(DEFAULT_PATH_NAME)
    one_item_sets_coverage = data_frame["artist"].value_counts()
    top_three_freq = one_item_sets_coverage[:3]
    print top_three_freq
    a = top_three_freq.index.values[0]
    b = top_three_freq.index.values[1]
    c = top_three_freq.index.values[2]
    print(str("Total transactions: " + str(len(data_frame['user'].unique()))))
    print(a)
    print(b)
    print(c)
    _, ab = find_rule_in_df(data_frame, a, b)
    _, ac = find_rule_in_df(data_frame, a, c)
    _, ba = find_rule_in_df(data_frame, b, a)
    _, bc = find_rule_in_df(data_frame, b, c)
    _, ca = find_rule_in_df(data_frame, c, a)
    _, cb = find_rule_in_df(data_frame, c, b)
    print(ab)
    print(ac)
    print(ba)
    print(bc)
    print(ca)
    print(cb)

if __name__ == "__main__":
    main()
