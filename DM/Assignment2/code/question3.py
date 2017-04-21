import pandas as pd


DEFAULT_PATH_NAME = '../data/employee_tenure.csv'


def read_file(pathname):
    try:
        return pd.read_csv(pathname)
    except IOError as ioe:
        """
        Exit program? Read exception handling for this case.
        """
        print(ioe)

data = read_file(DEFAULT_PATH_NAME)
def main():
    data = read_file(DEFAULT_PATH_NAME)

    # Getting tenure:
    tenure = data[['EmployeeID','length_of_service']]

    # We rename length_of_service to tenure
    tenure.columns = ['EmployeeID', 'tenure']

    # Mapping status to boolean
    censoring_mapping = {'ACTIVE': True, 'TERMINATED': False}

    # Getting Censoring status
    censoring_status = pd.concat([data['EmployeeID'], data['STATUS'].map(censoring_mapping)], axis=1)

    # Renaming column
    censoring_status.columns = ['EmployeeID', 'Censored']

    print len(censoring_status[censoring_status['Censored'] == True])
    print len(censoring_status[censoring_status['Censored'] == False])

    print(tenure[:10])
    print(censoring_status[:10])


if __name__ == '__main__':
    main()
