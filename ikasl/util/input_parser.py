import pandas as pd
from os import listdir
from os.path import isfile, join


class InputParser:

    @staticmethod
    def parse_input(folder_path, header='infer'):
        input_file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

        incrementer = 0
        input_database = {}
        for file in input_file_names:

            if file.split('0')[0] != 'frame':
                continue

            input_data = pd.read_csv(folder_path + file, header=header)
            input_database[incrementer] = input_data.as_matrix()

            incrementer += 1

        if header is not None:
            return input_database, list(input_data)
        else:
            return input_database

    @staticmethod
    def get_labels(filename):
        return list(pd.read_csv(filename))

    # @staticmethod
    # def parse_input_scene_analyzer(folder_path, header='infer'):
    #     input_file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    #
    #     incrementer = 0
    #     input_database = {}
    #     for file in input_file_names:
    #
    #         if file.split('.')[0] != 'vision_features':
    #             continue
    #
    #         input_data = pd.read_csv(folder_path + file, header=header)
    #         input_database[incrementer] = input_data.as_matrix()
    #
    #         incrementer += 1
    #
    #     if header is not None:
    #         return input_database, list(input_data)
    #     else:
    #         return input_database
    #
    # @staticmethod
    # def parse_input_fire_data(folder_path, header='infer'):
    #     input_file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    #
    #     incrementer = 0
    #     input_database = {}
    #     for file in input_file_names:
    #
    #         if file.split('.')[0] != 'features':
    #             continue
    #
    #         input_data = pd.read_csv(folder_path + file, header=header)
    #         input_database[incrementer] = input_data.as_matrix()
    #
    #         incrementer += 1
    #
    #     if header is not None:
    #         return input_database, list(input_data)
    #     else:
    #         return input_database
    #
    # @staticmethod
    # def parse_input_telco_data(folder_path, header='infer'):
    #     input_file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    #
    #     incrementer = 0
    #     input_database = {}
    #     for file in input_file_names:
    #
    #         if file.split('.')[0] != 'outage_feb9_features':
    #             continue
    #
    #         input_data = pd.read_csv(folder_path + file, header=header)
    #         input_database[incrementer] = input_data.as_matrix()
    #
    #         incrementer += 1
    #
    #     if header is not None:
    #         return input_database, list(input_data)
    #     else:
    #         return input_database
    #
    # @staticmethod
    # def parse_input_vicroads_bluetooth_travel_data(folder_path, filename, header='infer'):
    #     input_file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    #
    #     incrementer = 0
    #     input_database = {}
    #     for file in input_file_names:
    #
    #         if file.split('.')[0] != filename:
    #             continue
    #
    #         input_data = pd.read_csv(folder_path + file, header=header)
    #         del input_data['Unnamed: 0']
    #
    #         input_database[incrementer] = input_data.as_matrix()
    #
    #         incrementer += 1
    #
    #     if header is not None:
    #         return input_database, list(input_data)
    #     else:
    #         return input_database