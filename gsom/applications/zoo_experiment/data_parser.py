import pandas as pd
import re


class InputParser:

    @staticmethod
    def parse_input_zoo_data(filename, header='infer'):

        input_data = pd.read_csv(filename, header=header)

        classes = input_data[17].tolist()
        labels = input_data[0].tolist()
        input_database = {
            0: input_data.as_matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        }

        return input_database, labels, classes
