import time
from core import ikasl_layer as Core_IKASL_Layer
import sys


class IKASL:

    def __init__(self, params):
        self.params = params

    def aggregate_gsom(self, ikasl_level, input_vector_db, ikasl_sequence):
        batch_vector_weights = input_vector_db[0]
        # Build generalisation layer
        ikasl_sequence[ikasl_level].generalisation_layer = None
        ikasl_sequence[ikasl_level].params = self.params
        ikasl_sequence[ikasl_level].build_generalisation_layer(batch_vector_weights, batch_vector_weights.shape[1])

        # Assign inputs to aggregated nodes
        if len(ikasl_sequence[ikasl_level].get_generalisation_layer().aggregated_nodemap) > 0:
            ikasl_sequence[ikasl_level].cluster_inputs_for_final_step(batch_vector_weights,
                                                                ikasl_sequence[ikasl_level].get_generalisation_layer())
        else:
            print('No aggregated nodes, due to high hit threshold of',
                  ikasl_sequence[ikasl_level].get_generalisation_layer().hit_threshold)
            sys.exit(0)

        return ikasl_sequence

    def run(self, input_vector_db):
        ikasl_sequence = {}
        for batch_key, batch_vector_weights in input_vector_db.items():

            batch_id = int(batch_key)
            ikasl_sequence[batch_id] = Core_IKASL_Layer.IKASLLayer(batch_id, self.params)

            start_time = time.time()

            # batch_vector_weights = batch_vector_weights[8:int((len(batch_vector_weights)-20)/2)]
            # batch_vector_weights = batch_vector_weights[:1010]

            if batch_id == 0:
                ikasl_sequence[batch_id].build_gsom_layer(batch_vector_weights, batch_vector_weights.shape[1])
                ikasl_sequence[batch_id].build_generalisation_layer(batch_vector_weights, batch_vector_weights.shape[1])

                # Update R value to comply for one node at deriving the learning rate
                self.params.get_gsom_parameters().update_R_for_one_starting_node()
            else:
                ikasl_sequence[batch_id].build_gsom_layer(batch_vector_weights, batch_vector_weights.shape[1],
                                                          ikasl_sequence[batch_id - 1].get_generalisation_layer())
                ikasl_sequence[batch_id].build_generalisation_layer(batch_vector_weights, batch_vector_weights.shape[1],
                                                          ikasl_sequence[batch_id - 1].get_generalisation_layer())

            if (batch_id == (len(input_vector_db)-1)) and \
                    (len(ikasl_sequence[batch_id].get_generalisation_layer().aggregated_nodemap) > 0):
                ikasl_sequence[batch_id].cluster_inputs_for_final_step(batch_vector_weights,
                                                        ikasl_sequence[batch_id].get_generalisation_layer())
            elif batch_id == (len(input_vector_db)-1):
                print('Final clusters not assigned/ calibrated, due to high hit threshold of',
                      ikasl_sequence[batch_id].get_generalisation_layer().hit_threshold)

            print('IKASL sequence', batch_id, 'completed in', round(time.time() - start_time, 2), '(s)\n')

        return ikasl_sequence
