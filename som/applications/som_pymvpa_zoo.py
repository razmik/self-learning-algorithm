"""
Source: http://www.pymvpa.org/examples/som.html
Installation: https://github.com/PyMVPA/PyMVPA/blob/master/doc/source/installation.rst
"""
import sys

sys.path.append('../')

from mvpa2.suite import *
from data import input_parser as Parser


if __name__ == '__main__':

    input_filename = '../data/zoo.txt'

    input_vector_database, labels, classes = Parser.InputParser.parse_input_zoo_data(input_filename, None)

    # SOM Learning Phase
    som = SimpleSOMMapper((40, 40), 100, learning_rate=0.01)
    som.train(input_vector_database[0])
    mapped = som(input_vector_database[0])

    pl.figure(1)
    pl.title('Animals SOM')

    # SOM's kshape is (rows x columns), while matplotlib wants (X x Y)
    label_dict = {}
    class_dict = {}

    # Generate Visualization
    for i, m in enumerate(mapped):

        key = str(m[1]) + '-' + str(m[0])

        if key not in label_dict.keys():
            label_dict[key] = [labels[i]]
            class_dict[key] = [classes[i]]
        else:
            label_dict[key].append(labels[i])
            class_dict[key].append(classes[i])

    for key, val in label_dict.items():

        x, y = int(key.split('-')[0]), int(key.split('-')[1])

        pl.plot(x, y, 'o', color='k', markersize=2)

        # set labels
        val = list(set(val))
        pl.text(x, y + 0.4, ','.join(val), fontsize=5)

        # set classes
        classes_set = list(set(class_dict[key]))
        pl.text(x + 0.2, y - 0.3, ','.join(str(e) for e in classes_set), fontsize=5)

    pl.show()

    print('Completed.')
