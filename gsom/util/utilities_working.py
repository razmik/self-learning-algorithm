import pickle
import datetime, time, sys
import math
import numpy as np
from scipy import spatial
from params import params as Params


class Utilities:

    @staticmethod
    def get_distance(vector_1, vector_2, method, divider=-1):
        if method == Params.DistanceFunction.EUCLIDEAN:
            return np.linalg.norm(vector_2 - vector_1)
            # return spatial.distance.euclidean(vector_2, vector_1)
        elif method == Params.DistanceFunction.COSINE:
            if np.count_nonzero(vector_2) != 0:
                return spatial.distance.cosine(vector_2, vector_1)
            else:
                return 1.0
        elif method == Params.DistanceFunction.COMBINED:
            cosine_dist = spatial.distance.cosine(vector_2[0:(divider+1)], vector_1[0:(divider+1)])
            euclidean_dist = spatial.distance.euclidean(vector_2[(divider+1):len(vector_2)], vector_1[(divider+1):len(vector_1)])
            distance = (euclidean_dist * 10 + cosine_dist) / 2
            return distance
        else:
            print('ERROR: Undefined distance function -', method)
            sys.exit(-1)

    @staticmethod
    def get_distance_recurrent(global_context, recurrent_weights, alphas):

        gamma_distance = np.linalg.norm(np.dot(alphas.T, (global_context - recurrent_weights)))

        # gamma_distance = 0.0
        # for i in range(0, len(global_context)):
        #     gamma_distance += (alphas[i] * np.linalg.norm(global_context[i] - recurrent_weights[i]))

        # print(gamma_distance, gamma_distance2)
        # assert round(gamma_distance, 5) == round(gamma_distance2, 5)

        return gamma_distance

    @staticmethod
    def select_winner_recurrent(nodemap, global_context, alphas):

        if global_context.shape[0] < 2:

            # Copy this into an numpy matrix
            keys = list(nodemap.keys())
            dims = nodemap[keys[0]].dimensions
            values = np.asarray([nodemap[k].recurrent_weights.reshape((dims)) for k in keys])
            global_context_dim = np.tile(global_context, (values.shape[0], 1))
            distances = np.sqrt(np.sum((values - global_context_dim)**2, axis=1))

            return nodemap[keys[np.argmin(distances)]]

        else:
            _, winner = min(nodemap.items(), key=lambda node: Utilities.get_distance_recurrent(global_context, node[1].recurrent_weights, alphas))
            return winner

    @staticmethod
    def get_decremental_alphas(num_contexts):
        iWe = np.zeros(num_contexts)  # iWe[:] = 1 / nWeights
        for h in range(0, len(iWe)):
            iWe[h] = np.exp(-h)  # 1. / nWeights + np.exp(-h)
        iWe[:] = iWe[:] / sum(iWe)  # print iWe[:]
        return iWe

    @staticmethod
    def get_max_node_distance_square(node_1, node_2):
        return max(math.pow(node_1.x - node_2.x, 2), math.pow(node_1.y - node_2.y, 2))

    @staticmethod
    def generate_index(x, y):
        return str(x) + ':' + str(y)

    @staticmethod
    def select_winner(nodemap, input_vector, distance_function, distance_divider):
        _, winner = min(nodemap.items(), key=lambda node: Utilities.get_distance(node[1].weights, input_vector,
                                                                                 distance_function, distance_divider))
        return winner

    @staticmethod
    def select_input_to_closest_aggregate_node(aggr_node_list, input_weight, distance_function, distance_divider):
        min_dist = float("inf")

        list_index = -1
        itr = 0
        for aggr_node in aggr_node_list:
            curr_dist = Utilities.get_distance(aggr_node.weights, input_weight.weight,
                                               distance_function, distance_divider)
            if curr_dist < min_dist:
                min_dist = curr_dist
                list_index = itr
            itr += 1

        aggr_node_list[list_index].select_input_vector(input_weight)

    @staticmethod
    def neighbors(nx, ny, neighbour_radius):
        return [(x2, y2) for x2 in range(nx - neighbour_radius, nx + neighbour_radius + 1)
                for y2 in range(ny - neighbour_radius, ny + neighbour_radius + 1)
                if (nx != x2 or ny != y2)]

    @staticmethod
    def increment_node_ages(nodemap):
        for key, node in nodemap.items():
            node.age_increment()

    @staticmethod
    def remove_older_nodes(nodemap, age_threshold):
        for key in list(nodemap.keys()):
            if nodemap[key].age > age_threshold:
                del nodemap[key]

    @staticmethod
    def save_object(input_object, filename):
        suffix = '_'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        full_name = filename+suffix
        with open(full_name+'.pickle', 'wb') as handle:
            pickle.dump(input_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return full_name

    @staticmethod
    def load_object(filename):
        return pickle.load(open(filename+".pickle", "rb"))


class SugenoFuzzyIntregal:

    @staticmethod
    def get_sugeno_fuzzy_integral(h_values, g_values, sugeno_lambda):

        n = len(h_values)
        min_array = []
        array_list = []

        index_array = np.arange(0, n)

        for i in range(0, n):
            for j in range(1, (n-1)):
                if h_values[j-1] < h_values[j]:
                    index_array[j - 1], index_array[j] = index_array[j], index_array[j - 1]
                    h_values[j - 1], h_values[j] = h_values[j], h_values[j - 1]

        for i in range(0, n):
            array_list.append(g_values[index_array[i]])
            min_array.append(min(h_values[i], SugenoFuzzyIntregal.get_combination_value(array_list, sugeno_lambda)))

        new_min_array = sorted(min_array)

        return new_min_array[-1]

    @staticmethod
    def get_combination_value(values, sugeno_lambda):

        if len(values) == 1:
            return values[0]
        else:
            temp = SugenoFuzzyIntregal.get_combination_value(values[1:], sugeno_lambda)
            return values[0] + temp + (sugeno_lambda * values[0] * temp)
