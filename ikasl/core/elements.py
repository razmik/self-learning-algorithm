import random
import numpy as np
from enum import Enum
from util import utilities as Utils


class DistanceFunction(Enum):
    COSINE = 'cosine'
    EUCLIDEAN = 'euclidean'
    COMBINED = 'combined'


class AggregateFunction(Enum):
    AVERAGE = 'average'
    MAX = 'max'
    FUZZY = 'fuzzy'
    PROXIMITY_AVERAGE = 'proximity_average'


class InputWeight:
    def __init__(self, weight, label):
        self.weight = weight
        self.weight_label = label


class GSOMNodeMap:

    def __init__(self, gsom_node_map, pathway_id):
        self.gsom_node_map = gsom_node_map
        self.pathway_id = pathway_id

    def get_gsom_node_map(self):
        return self.gsom_node_map

    def get_pathway_id(self):
        return self.pathway_id


class AggregateNode:
    def __init__(self, ikasl_layer_id, weights, parent_pathway, current_pathway):
        self.ikasl_layer_id = ikasl_layer_id
        self.weights = weights
        self.parent_pathway_id = parent_pathway
        self.pathway_id = current_pathway
        self.children_pathway_ids = []
        self.input_vector_weights = []

    def select_input_vector(self, input_weight):
        self.input_vector_weights.append(input_weight)

    def add_child(self, child):
        self.children_pathway_ids.append(child)

    def get_input_vectors(self):
        input_weights = list(input_weight.weight for input_weight in self.input_vector_weights)
        return np.asarray(input_weights)

    def get_weights(self):
        return self.weights

    def get_input_vector_count(self):
        return len(self.input_vector_weights)

    def get_pathway_id(self):
        return self.pathway_id

    def get_parent_pathway_id(self):
        return self.parent_pathway_id

    def get_children_pathway_ids(self):
        return self.children_pathway_ids


class GSOMNode:
    R = random.Random()

    def __init__(self, x, y, weights):
        self.weights = weights
        self.x, self.y = x, y

        # Remember the error occuring at this particular node
        self.error = 0.0

        # Remember hit count for IKASL aggregation
        self.hit_count = 0

        # To be used to map labels and classes after GSOM phases are completed
        self.mappedLabels = []
        self.mappedClasses = []
        self.data = []

    def hit(self):
        self.hit_count += 1

    def adjust_weights(self, target, influence, learn_rate):
        self.weights += influence * learn_rate * (target - self.weights)

    def cal_and_update_error(self, input_vector, distance_function, distance_divider):
        self.error += Utils.Utilities.get_distance(self.weights, input_vector, distance_function, distance_divider)

    def map_label(self, input_label):
        self.mappedLabels.append(input_label)

    def map_class(self, input_class):
        self.mappedClasses.append(input_class)

    def map_data(self, input_data):
        self.data.append(input_data)

    def get_hit_count(self, ):
        return self.hit_count
