from core import elements as Elements
from core import gsom as GSOM_Core
from util import utilities as Utils

class GSOMLayer:
    def __init__(self, layer_id, prev_gen_layer, params, dimensions):
        self.gsom_node_maps = []
        self.ikasl_layer_id = layer_id
        self.prev_gen_layer = prev_gen_layer
        self.params = params
        self.dimensions = dimensions

    def generate(self, input_vector_weights):

        if self.prev_gen_layer is not None:

            aggregate_nodes = self.prev_gen_layer.get_generalised_nodemap()

            # Construct the input_weight_vectors relevant to each aggregate node using distance of input to aggr weight.
            itr = 0  # The ID of aggregate node of each batch.
            for input_vector in input_vector_weights:
                Utils.Utilities.select_input_to_closest_aggregate_node(aggregate_nodes,
                                                                       Elements.InputWeight(input_vector, itr),
                                                                       self.params.get_gsom_parameters().DISTANCE_FUNCTION,
                                                                       self.params.get_gsom_parameters().DISTANCE_DIVIDER)
                itr += 1

            # Generate GSOM node maps for each of the aggregate node.
            for aggregate_node in aggregate_nodes:
                # Create GSOM node map on the aggregate node only if input vectors have been selected to the aggr node.
                if aggregate_node.get_input_vector_count() > 0:
                    gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), aggregate_node.get_input_vectors(), self.dimensions,
                                aggregate_node)
                    gsom.grow()
                    gsom.smooth()
                    gsom_nodemap = Elements.GSOMNodeMap(gsom.evaluate_hits(aggregate_node.get_input_vectors()),
                                               aggregate_node.get_pathway_id())
                    self.gsom_node_maps.append(gsom_nodemap)

        else:
            # First GSOM node map will be initialize with four init nodes.
            gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), input_vector_weights, self.dimensions)
            gsom.grow()
            gsom.smooth()
            # gsom.assign_hits()
            # Name the first node map as the pathway 0
            gsom_nodemap = Elements.GSOMNodeMap(gsom.evaluate_hits(input_vector_weights), 0)
            self.gsom_node_maps.append(gsom_nodemap)

    def get_node_maps(self):
        return self.gsom_node_maps