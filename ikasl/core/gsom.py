import math
import random
import functools
import numpy as np
from core import growth_handler as Growth_Handler
from core import elements as Elements
from util import utilities as Utils

import time


class GSOM:

    def __init__(self, params, input_vectors, dimensions, aggregate_node=None):
        self.parameters = params
        self.inputs = np.asarray(input_vectors)
        self.growth_handler = Growth_Handler.GrowthHandler()
        self.aggregate_node = aggregate_node
        self.dimensions = dimensions
        self.learn_smooth_sample_size = self.parameters.get_learn_smooth_sample_size(len(self.inputs))
        self.gsom_nodemap = {}

    def grow(self):

        self._initialize_network(self.dimensions)
        param = self.parameters

        # Optimise python references: that are reevaluated each time through the loop
        grow = self._grow_for_single_iteration_and_single_input

        learning_rate = param.START_LEARNING_RATE
        for i in range(0, param.LEARNING_ITERATIONS):

            if i != 0:
                learning_rate = self._get_learning_rate(param, learning_rate, len(self.gsom_nodemap))

            neighbourhood_radius = self._get_neighbourhood_radius(param.LEARNING_ITERATIONS, i,
                                                                  param.MAX_NEIGHBOURHOOD_RADIUS)

            for k in random.sample(range(0, len(self.inputs)), self.learn_smooth_sample_size):
                grow(self.inputs[k], learning_rate, neighbourhood_radius)

        return self.gsom_nodemap

    def smooth(self):

        learning_rate = self.parameters.START_LEARNING_RATE * self.parameters.SMOOTHING_LEARNING_RATE_FACTOR
        reduced_neighbourhood_radius = self.parameters.MAX_NEIGHBOURHOOD_RADIUS * self.parameters.SMOOTHING_NEIGHBOURHOOD_RADIUS_FACTOR

        smooth = self._smooth_for_single_iteration_and_single_input

        for i in range(0, self.parameters.SMOOTHING_ITERATIONS):

            if i != 0:
                learning_rate = self._get_learning_rate(self.parameters, learning_rate, len(self.gsom_nodemap))

            neighbourhood_radius = self._get_neighbourhood_radius(self.parameters.SMOOTHING_ITERATIONS, i,
                                                                  reduced_neighbourhood_radius)

            for k in random.sample(range(0, len(self.inputs)), self.learn_smooth_sample_size):
                smooth(self.inputs[k], learning_rate, neighbourhood_radius)

        return self.gsom_nodemap

    def assign_hits(self):

        curr_count = 0
        for cur_input in self.inputs:
            winner = Utils.Utilities.select_winner(self.gsom_nodemap, cur_input, self.parameters.DISTANCE_FUNCTION,
                                                   self.parameters.DISTANCE_DIVIDER)
            node_index = Utils.Utilities.generate_index(winner.x, winner.y)
            self.gsom_nodemap[node_index].map_label(curr_count)
            curr_count += 1

    def evaluate_hits(self, input_vectors):

        for i in range(0, len(input_vectors)):
            input_vector = input_vectors[i]
            Utils.Utilities.select_winner(self.gsom_nodemap, input_vector, self.parameters.DISTANCE_FUNCTION, self.parameters.DISTANCE_DIVIDER).hit()

        return self.gsom_nodemap

    def _smooth_for_single_iteration_and_single_input(self, input_vector, learning_rate, neigh_radius):

        gsom_nodemap = self.gsom_nodemap
        winner = Utils.Utilities.select_winner(gsom_nodemap, input_vector, self.parameters.DISTANCE_FUNCTION, self.parameters.DISTANCE_DIVIDER)

        left = Utils.Utilities.generate_index(winner.x - 1, winner.y)
        right = Utils.Utilities.generate_index(winner.x + 1, winner.y)
        top = Utils.Utilities.generate_index(winner.x, winner.y + 1)
        bottom = Utils.Utilities.generate_index(winner.x, winner.y - 1)

        if left in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[left], winner, input_vector, neigh_radius,
                                                learning_rate)
        elif right in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[right], winner, input_vector, neigh_radius,
                                                learning_rate)
        elif top in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[top], winner, input_vector, neigh_radius,
                                                learning_rate)
        elif bottom in gsom_nodemap:
            self._adjust_weights_for_neighbours(gsom_nodemap[bottom], winner, input_vector, neigh_radius,
                                                learning_rate)

    def _grow_for_single_iteration_and_single_input(self, input_vector, learning_rate, neigh_radius):

        param = self.parameters
        gsom_nodemap = self.gsom_nodemap
        winner = Utils.Utilities.select_winner(gsom_nodemap, input_vector, param.DISTANCE_FUNCTION, param.DISTANCE_DIVIDER)

        # Update the error value of the winner node
        winner.cal_and_update_error(input_vector, param.DISTANCE_FUNCTION, param.DISTANCE_DIVIDER)

        # Weight adaptation for winner's neighborhood
        adjust = self._adjust_weights_for_neighbours
        for node_id in list(gsom_nodemap):
            # Exclude winner from the nodemap since winner's weight has already been updated in the previous step
            if not (gsom_nodemap[node_id].x == winner.x and gsom_nodemap[node_id].y == winner.y):
                adjust(gsom_nodemap[node_id], winner, input_vector, neigh_radius,
                                                    learning_rate)

        # Evaluate winner's weights and grow network it it's above Growth Threshold (GT)
        if winner.error > param.get_gt(len(input_vector)):
            self._adjust_winner_error(winner, len(input_vector))

    def _adjust_winner_error(self, winner, dimensions):

        left = Utils.Utilities.generate_index(winner.x - 1, winner.y)
        right = Utils.Utilities.generate_index(winner.x + 1, winner.y)
        top = Utils.Utilities.generate_index(winner.x, winner.y + 1)
        bottom = Utils.Utilities.generate_index(winner.x, winner.y - 1)

        if left in self.gsom_nodemap and right in self.gsom_nodemap and top in self.gsom_nodemap and bottom in self.gsom_nodemap:
            # If the network has adequate neurons to process the input data, the weight vectors of those neurons are
            #  adapted as such the distribution of the weight vectors will represent the input vector distribution.
            self._distribute_error_to_neighbours(winner, left, right, top, bottom, dimensions)
        else:
            # If the network does not have sufficient neurons, the weight will be accumulated on a single neuron.
            self.growth_handler.grow_nodes(self.gsom_nodemap, winner)

    def _distribute_error_to_neighbours(self, winner, left, right, top, bottom, dimensions):

        winner.error = self.parameters.get_gt(dimensions)
        self.gsom_nodemap[left].error = self._calc_error_for_neighbours(self.gsom_nodemap[left])
        self.gsom_nodemap[right].error = self._calc_error_for_neighbours(self.gsom_nodemap[right])
        self.gsom_nodemap[top].error = self._calc_error_for_neighbours(self.gsom_nodemap[top])
        self.gsom_nodemap[bottom].error = self._calc_error_for_neighbours(self.gsom_nodemap[bottom])

    def _calc_error_for_neighbours(self, node):
        return node.error * (1 + self.parameters.FD)

    def _adjust_weights_for_neighbours(self, node, winner, input_vector, neigh_radius, learning_rate):

        node_dist_sqr = math.pow(winner.x - node.x, 2) + math.pow(winner.y - node.y, 2)
        neigh_radius_sqr = neigh_radius * neigh_radius

        if node_dist_sqr < neigh_radius_sqr:
            influence = math.exp(- node_dist_sqr / (2 * neigh_radius_sqr))
            node.adjust_weights(input_vector, influence, learning_rate)

    def _initialize_network(self, dimensions):

        if self.aggregate_node is not None:
            # Generate the node map for aggregated nodes following from the second sequence of weights
            self.gsom_nodemap = {
                '0:0': Elements.GSOMNode(0, 0, self.aggregate_node.get_weights())
            }
        else:
            # Generate the node map for initial GSOM layer - for all the inputs
            self.gsom_nodemap = {
                '0:0': Elements.GSOMNode(0, 0, np.random.rand(dimensions)),
                '0:1': Elements.GSOMNode(0, 1, np.random.rand(dimensions)),
                '1:0': Elements.GSOMNode(1, 0, np.random.rand(dimensions)),
                '1:1': Elements.GSOMNode(1, 1, np.random.rand(dimensions)),
            }

    def _get_learning_rate(self, parameters, prev_learning_rate, nodemap_size):
        return parameters.ALPHA * (1 - (parameters.R / nodemap_size)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration, max_neighbourhood_radius):
        time_constant = total_iteration / math.log(max_neighbourhood_radius)
        return max_neighbourhood_radius * math.exp(- iteration / time_constant)
