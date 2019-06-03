import random
import numpy as np
from util import utilities as Utils


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
    def __init__(self, ikasl_layer_id, weights):
        self.ikasl_layer_id = ikasl_layer_id
        self.weights = weights
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

    def get_children_pathway_ids(self):
        return self.children_pathway_ids


class GSOMNode:
    R = random.Random()

    def __init__(self, x, y, weight, context_weights):

        self.x, self.y = x, y
        self.dimensions = len(weight)
        self.num_contexts = len(context_weights) + 1

        # Remember the error occurring at this particular node
        self.error = 0.0

        # Age parameter: to keep track of the last time the neuron fired (got a hit).
        self.age = 0

        # Habituation rate
        self.habituation = 1

        # Remember hit count for IKASL aggregation
        self.hit_count = 0

        # Recurrent weights
        self.recurrent_weights = np.zeros((self.num_contexts, len(weight)))
        self.recurrent_weights[0] = weight  # current weight vector
        self.recurrent_weights[1:] = context_weights

        # To be used to map labels and classes after GSOM phases are completed
        self.mappedLabels = []
        self.mappedClasses = []
        self.data = []

        # Updated field for evaluation
        self.weights = None

    def age_increment(self):
        self.age += 1  # Increment the age by one. Due to not firing at an iteration.

    def fired_in_growing(self):
        self.age = 0  # Fired, thus age is reset to zero.

    def hit(self):
        self.hit_count += 1

    def habituate_neuron(self, tau):
        self.habituation += (tau * 1.05 * (1. - self.habituation) - tau)

    def adjust_weights(self, global_context, influence, learn_rate):  # epsilon == learning rate
        delta_weights = np.zeros((self.num_contexts, self.dimensions))
        for i in range(0, self.num_contexts):
            delta_weights[i] = influence * learn_rate * (global_context[i] - self.recurrent_weights[i]) * self.habituation

        # Update recurrent weights of the GSOM neuron
        self.recurrent_weights += delta_weights

    def cal_and_update_error(self, global_context, alphas):
        self.error += Utils.Utilities.get_distance_recurrent(global_context, self.recurrent_weights, alphas)

    def map_label(self, input_label):
        self.mappedLabels.append(input_label)

    def map_class(self, input_class):
        self.mappedClasses.append(input_class)

    def map_data(self, input_data):
        self.data.append(input_data)

    def get_hit_count(self):
        return self.hit_count

    def get_mapped_labels(self):
        return self.mappedLabels

    def setup_weights(self):
        self.weights = self.recurrent_weights[0]