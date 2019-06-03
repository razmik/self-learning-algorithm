from util import utilities as Utils
from core import elements as Elements
from core import gsom_layer as Core_GSOM_Layer
from core import generalisation_layer as Core_Generalisation_Layer

class IKASLLayer:

    def __init__(self, layer_id, params):
        self.layer_id = layer_id
        self.params = params
        self.gsom_layer = None
        self.generalisation_layer = None

    def build_gsom_layer(self, input_vector_weights, dimensions, prev_ikasl_layer=None):

        self.gsom_layer = Core_GSOM_Layer.GSOMLayer(self.layer_id, prev_ikasl_layer, self.params, dimensions)
        self.gsom_layer.generate(input_vector_weights)
        print('Sq:', self.layer_id, 'Learning Layer built with', len(self.gsom_layer.get_node_maps()), 'layers')

    def build_generalisation_layer(self, input_vector_weights, dimensions, prev_ikasl_layer=None):

        self.generalisation_layer = Core_Generalisation_Layer.GeneralisationLayer(self.layer_id, self.gsom_layer, self.params,
                                                        len(input_vector_weights), dimensions, prev_ikasl_layer)
        self.generalisation_layer.generalise()
        print('Sq:', self.layer_id, 'Generalisation Layer built with',
              len(self.generalisation_layer.get_generalised_nodemap()), 'nodes')

    def cluster_inputs_for_final_step(self, input_vector_weights, prev_ikasl_layer=None):
        if prev_ikasl_layer is not None:
            aggregate_nodes = prev_ikasl_layer.get_generalised_nodemap()

            # Construct the input_weight_vectors relevant to each aggregate node using distance of input to aggr weight.
            itr = 0  # The ID of aggregate node of each batch.
            for input_vector in input_vector_weights:
                Utils.Utilities.select_input_to_closest_aggregate_node(aggregate_nodes,
                        Elements.InputWeight(input_vector, itr), self.params.get_gsom_parameters().DISTANCE_FUNCTION,
                                                                       self.params.get_gsom_parameters().DISTANCE_DIVIDER)
                itr += 1

    def get_generalisation_layer(self):
        return self.generalisation_layer