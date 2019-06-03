import sys
import math
import numpy as np
from util import utilities as Utils
from core import elements as Elements

PATHWAY_TRACKER = 0


class GeneralisationLayer:
    def __init__(self, layer_id, gsom_layer, ikasl_params, dataset_length, dimensions, prev_gen_layer):
        self.aggregated_nodemap = []
        self.gsom_layer = gsom_layer
        self.ikasl_params = ikasl_params
        self.neighbour_radius = ikasl_params.get_aggregate_proximity()
        self.dimensions = dimensions
        self.layer_id = layer_id
        self.hit_threshold = math.ceil(dataset_length * ikasl_params.get_hit_threshold_fraction())
        self.prev_gen_layer = prev_gen_layer
        self.selected_pathway_ids = []

    def generalise(self):
        # Generate new knowledge
        for gsom_node_map in self.gsom_layer.get_node_maps():
            if self.ikasl_params.is_aggregate_inside_hitnode_proximity():
                self.aggregated_nodemap.extend(self._aggregate_nodemap(gsom_node_map))
            else:
                self.aggregated_nodemap.extend(self._aggregate_nodemap_concise(gsom_node_map))

        # Persist existing knowledge
        self.aggregated_nodemap.extend(self._persist_existing_pathways())

    def get_generalised_nodemap(self):
        return self.aggregated_nodemap

    def _persist_existing_pathways(self):
        # If previous pathway is not selected for processing, bring forth the most recent aggregate nodes of those
        # pathways.
        aggregate_nodes = []

        if self.prev_gen_layer is not None:
            aggregate_nodes_prev_gen_layer = self.prev_gen_layer.get_generalised_nodemap()
            for aggr_node in aggregate_nodes_prev_gen_layer:
                aggr_pathway_id = aggr_node.get_pathway_id()
                if aggr_pathway_id not in self.selected_pathway_ids:
                    aggregate_nodes.append(
                        Elements.AggregateNode(self.layer_id, aggr_node.get_weights(), aggr_pathway_id,
                                               aggr_pathway_id))
                    self.selected_pathway_ids.append(aggr_pathway_id)

        return aggregate_nodes

    def _aggregate_nodemap_concise(self, gsom_node_map_wrapper):

        """
        Criteria of aggregation:
            Select the nodes that are above hit_threshold(HT).
            Sort them into descending order of hit_count and add them into a queue.
            Process the queue nodes into aggregation.
                If an hit_node which is in the queue (Who has hit threshold greater than HT) encounters in the
                aggregation, remove/discard it from the queue.
        """

        gsom_node_map = gsom_node_map_wrapper.get_gsom_node_map()
        pathway_id = gsom_node_map_wrapper.get_pathway_id()

        # Select the hit nodes
        hit_nodes = []
        for node in gsom_node_map:
            if gsom_node_map[node].get_hit_count() > self.hit_threshold:
                hit_nodes.append(gsom_node_map[node])

        # Sort the hit nodes
        hit_nodes.sort(key=lambda nd: nd.get_hit_count(), reverse=True)

        # Process the queue nodes into aggregation.
        global PATHWAY_TRACKER
        aggregate_nodes = []
        discard_list = []
        generated_pathways = 0
        for hit_node in hit_nodes:

            if Utils.Utilities.generate_index(hit_node.x, hit_node.y) in discard_list:
                continue

            # Construct the pathway name/ id
            if generated_pathways == 0:
                current_pathway_id = pathway_id
            else:
                current_pathway_id = PATHWAY_TRACKER + 1
                PATHWAY_TRACKER += 1

            aggregate_nodes.append(self._aggregate_single_node(hit_node, gsom_node_map, pathway_id,
                                                               current_pathway_id, discard_list=discard_list))
            self.selected_pathway_ids.append(pathway_id)
            generated_pathways += 1

        return aggregate_nodes

    def _aggregate_nodemap(self, gsom_node_map_wrapper):

        global PATHWAY_TRACKER

        aggregate_nodes = []
        gsom_node_map = gsom_node_map_wrapper.get_gsom_node_map()
        pathway_id = gsom_node_map_wrapper.get_pathway_id()
        generated_pathways = 0
        for node in gsom_node_map:
            if gsom_node_map[node].get_hit_count() > self.hit_threshold:

                # Construct the pathway name/ id
                if generated_pathways == 0:
                    current_pathway_id = pathway_id
                else:
                    current_pathway_id = PATHWAY_TRACKER + 1
                    PATHWAY_TRACKER += 1

                aggregate_nodes.append(self._aggregate_single_node(gsom_node_map[node], gsom_node_map, pathway_id,
                                                                   current_pathway_id))
                self.selected_pathway_ids.append(pathway_id)
                generated_pathways += 1

        return aggregate_nodes

    def _aggregate_single_node(self, selected_node, gsom_node_map, parent_pathway_id, current_pathway_id,
                               discard_list=None):

        x, y = selected_node.x, selected_node.y

        if self.neighbour_radius > 4:
            print('IKASL aggregate proximity over 4 not implemented.')
            sys.exit(-1)

        if self.ikasl_params.get_aggregation_function() == Elements.AggregateFunction.AVERAGE:
            # Average Aggregation function
            count = 0
            neighbourhood_weight = np.zeros(self.dimensions)
            for neighbour in Utils.Utilities.neighbors(x, y, self.neighbour_radius):
                node_index = Utils.Utilities.generate_index(neighbour[0], neighbour[1])
                if node_index in gsom_node_map:
                    count += 1
                    neighbourhood_weight += gsom_node_map[node_index].weights
                    if discard_list is not None:
                        discard_list.append(node_index)  # Remove from the hit_nodes queue

            if count > 0:
                neighbourhood_weight /= count
                aggregated_weight = (selected_node.weights * 0.7) + (neighbourhood_weight * 0.3)
            else:
                aggregated_weight = selected_node.weights

        elif self.ikasl_params.get_aggregation_function() == Elements.AggregateFunction.MAX:
            # Max Aggregate function
            aggregated_weight = selected_node.weights
            for neighbour in Utils.Utilities.neighbors(x, y, self.neighbour_radius):
                node_index = Utils.Utilities.generate_index(neighbour[0], neighbour[1])
                if node_index in gsom_node_map:
                    aggregated_weight = np.maximum(aggregated_weight, gsom_node_map[node_index].weights)
                    if discard_list is not None:
                        discard_list.append(node_index)  # Remove from the hit_nodes queue

        elif self.ikasl_params.get_aggregation_function() == Elements.AggregateFunction.FUZZY:

            neighbours_ids = Utils.Utilities.neighbors(x, y, self.neighbour_radius)
            current_hit_node = Utils.Utilities.generate_index(x, y)

            # get distances for all the neighbours
            g_values = []
            for neighbour_id in neighbours_ids:
                neighbour_node_index = Utils.Utilities.generate_index(neighbour_id[0], neighbour_id[1])
                if neighbour_node_index in gsom_node_map:
                    g_values.append(1.0 - Utils.Utilities.get_distance(gsom_node_map[current_hit_node].weights,
                                                                       gsom_node_map[neighbour_node_index].weights,
                                                                       self.ikasl_params.get_gsom_parameters().DISTANCE_FUNCTION,
                                                                       self.ikasl_params.get_gsom_parameters().DISTANCE_DIVIDER))
                    if discard_list is not None:
                        discard_list.append(neighbour_node_index)  # Remove from the hit_nodes queue

            g_values = np.asarray(g_values)
            sum_normalised_g_values = [float(i) / sum(g_values) for i in g_values]

            aggregated_weight = []
            for i in range(0, len(gsom_node_map[current_hit_node].weights)):
                weights_i = []
                count = 0
                for neighbour_id in neighbours_ids:
                    neighbour_node_index = Utils.Utilities.generate_index(neighbour_id[0], neighbour_id[1])
                    if neighbour_node_index in gsom_node_map:
                        weights_i.append(gsom_node_map[neighbour_node_index].weights[i])
                        count += 1
                if count > 0:
                    aggregated_weight.append(
                        Utils.SugenoFuzzyIntregal.get_sugeno_fuzzy_integral(weights_i, sum_normalised_g_values,
                                                                            self.ikasl_params.get_sugeno_lambda()))

            if len(aggregated_weight) > 0:
                aggregated_weight = np.asarray(aggregated_weight)
            else:
                aggregated_weight = selected_node.weights

        elif self.ikasl_params.get_aggregation_function() == Elements.AggregateFunction.PROXIMITY_AVERAGE:
            # Average Aggregation function
            count = 0
            neighbourhood_1_weight = np.zeros(self.dimensions)
            neighbourhood_2_weight = np.zeros(self.dimensions)
            neighbourhood_3_weight = np.zeros(self.dimensions)
            neighbourhood_4_weight = np.zeros(self.dimensions)

            neigh_1_count = 0
            neigh_2_count = 0
            neigh_3_count = 0
            neigh_4_count = 0

            for neighbour in Utils.Utilities.neighbors(x, y, self.neighbour_radius):
                node_index = Utils.Utilities.generate_index(neighbour[0], neighbour[1])
                if node_index in gsom_node_map:
                    count += 1
                    distance = Utils.Utilities.get_max_node_distance_square(gsom_node_map[node_index], selected_node)
                    if distance < 2:
                        neighbourhood_1_weight += gsom_node_map[node_index].weights
                        neigh_1_count += 1
                    elif distance < 4:
                        neighbourhood_2_weight += gsom_node_map[node_index].weights
                        neigh_2_count += 1
                    elif distance < 10:
                        neighbourhood_3_weight += gsom_node_map[node_index].weights
                        neigh_3_count += 1
                    elif distance < 17:
                        neighbourhood_4_weight += gsom_node_map[node_index].weights
                        neigh_4_count += 1

                    if discard_list is not None:
                        discard_list.append(node_index)  # Remove from the hit_nodes queue

            if count > 0:
                if neigh_1_count > 0:
                    neighbourhood_1_weight /= neigh_1_count
                if neigh_2_count > 0:
                    neighbourhood_2_weight /= neigh_2_count
                if neigh_3_count > 0:
                    neighbourhood_3_weight /= neigh_3_count
                if neigh_4_count > 0:
                    neighbourhood_4_weight /= neigh_4_count

                aggregated_weight = (selected_node.weights * 0.5) + (neighbourhood_1_weight * 0.3) \
                                    + (neighbourhood_2_weight * 0.15) + (neighbourhood_3_weight * 0.04) \
                                    + (neighbourhood_4_weight * 0.01)
            else:
                aggregated_weight = selected_node.weights

        return Elements.AggregateNode(self.layer_id, aggregated_weight, parent_pathway_id, current_pathway_id)
