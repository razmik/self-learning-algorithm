import time
from core4 import gsom as GSOM_Core


class Controller:

    def __init__(self, params):
        self.params = params
        self.gsom_nodemap = None

    def _grow_gsom(self, inputs, dimensions, plot_for_itr=0, classes=None, output_loc=None):
        gsom = GSOM_Core.GSOM(self.params.get_gsom_parameters(), inputs, dimensions, plot_for_itr=plot_for_itr, activity_classes=classes, output_loc=output_loc)
        gsom.grow()
        gsom.smooth()
        self.gsom_nodemap = gsom.assign_hits()

    def run(self, input_vector_db, plot_for_itr=0, classes=None, output_loc=None):

        results = []

        for batch_key, batch_vector_weights in input_vector_db.items():

            batch_id = int(batch_key)

            start_time = time.time()
            self._grow_gsom(batch_vector_weights, batch_vector_weights.shape[1], plot_for_itr=plot_for_itr, classes=classes, output_loc=output_loc)
            print('Batch', batch_id)
            print('Neurons:', len(self.gsom_nodemap))
            print('Duration:', round(time.time() - start_time, 2), '(s)\n')

            results.append({
                'gsom': self.gsom_nodemap,
                'aggregated': None
            })

        return results
