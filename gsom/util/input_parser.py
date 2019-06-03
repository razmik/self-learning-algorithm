import pandas as pd
import sys
import csv
from os import listdir
from os.path import isfile, join


class InputParser:

    @staticmethod
    def parse_nba_basketball_data(data_filename):
        input_data = np.load(data_filename)
        print('wait.')

    @staticmethod
    def parse_input_mnist_autoencoder_data(data_filename, class_filename, length, header='infer'):

        input_data = pd.read_csv(data_filename, header=header)

        input_database = {
            0: input_data[:length].as_matrix()
        }

        labels = pd.read_csv(class_filename, header=header)[0].tolist()[:length]
        labels = [int(i) for i in labels]

        return input_database, labels

    @staticmethod
    def parse_input_zoo_data(filename, header='infer'):

        input_data = pd.read_csv(filename, header=header)

        classes = input_data[17].tolist()
        labels = input_data[0].tolist()
        input_database = {
            0: input_data.as_matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        }

        return input_database, labels, classes

    @staticmethod
    def parse_input_ucsd_ped_autoencoder_data(data_filename, length, test_train='any', header='infer'):

        input_data = pd.read_csv(data_filename, header=header)
        input_data.columns = [i for i in range(len(list(input_data)))]

        # Check if any null value exist for each column
        # https://stackoverflow.com/questions/29530232/python-pandas-check-if-any-value-is-nan-in-dataframe
        # temp = input_data.isnull().sum()

        input_data['test_train'] = input_data.apply(lambda row: row[0].split('_')[0], axis=1)

        # Shuffle the dataframe rows
        input_data = input_data.sample(frac=1).reset_index(drop=True)

        if test_train == 'train':
            input_data = input_data.loc[input_data['test_train'] == test_train]

        if length != -1:
            labels = input_data[0].tolist()[:length]
        else:
            labels = input_data[0].tolist()

        del input_data[0]
        del input_data['test_train']

        if length != -1:
            input_database = {
                0: input_data[:length].as_matrix()
            }
        else:
            input_database = {
                0: input_data[:].as_matrix()
            }

        return input_database, labels

    @staticmethod
    def parse_input_chasesun_data(filename, header='infer'):

        input_data = pd.read_csv(filename, header=header, low_memory=False)

        input_database = {
            0: input_data.as_matrix()
        }

        return input_database, list(input_data)

    @staticmethod
    def parse_input_adl_activity_data(filename, header='infer'):
        """
        DataSet
        http://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer
        """

        input_data = pd.read_csv(filename, header=header)

        # Filter activities
        input_data = input_data[input_data['activity'].isin(['brush_teeth', 'climb_stairs', 'eat_meat', 'eat_soup', 'liedown_bed'])]

        def compact_activity(row):
            activity = row['activity'].split('_')
            return activity[0][0] + activity[1][0]

        input_data['activity'] = input_data.apply(compact_activity, axis=1)

        # Fill missing values with 0: Resource: https://swdg.io/2015/finding-nans/
        input_data = input_data.fillna(0)

        activity_classes = input_data['activity'].tolist()
        volunteer_ids = input_data['vol_id'].tolist()
        input_database = {
            0: input_data[['vm_mean', 'vm_sd', 'vm_max', 'vm_min', 'vm_10perc', 'vm_25perc', 'vm_50perc', 'vm_75perc', 'vm_90perc', 'menmo']].as_matrix()
        }

        return input_database, volunteer_ids, activity_classes

    @staticmethod
    def output_list(data_list, filename):

        with open(filename, 'w') as f:
            wr = csv.writer(f, lineterminator='\n')
            wr.writerow(data_list)
