import time
import sys
sys.path.append('../../')

from util import input_parser as Parser
from util import utilities as Utils
from util import display as Display_Utils

from params import params as Params
from core import ikasl as Core_IKASL
from core import elements as Elements

# IKASL config
input_folder_path = ("input/combined/").replace('\\', '/')
labels_file = ("input/combined/frame0100.csv").replace('\\', '/')

# mode config
mode = 1  # IKASL Mode = 1, Display Mode = 2

# Construct graph view cofig
output_save_location = 'output/'
output_save_filename = 'collective-activity-bow'
output_image_title = 'IKASL for Collective Activity Dataset'
labels_to_show = 10
display_output_format = 'svg'
enable_style = True

# Display mode config
display_filename = 'collective-activity-bow_2017-10-26-15-52-50'

# View config
# image_files_root_folder = 'E:\Data\Collective Activity Dataset\ActivityDataset/'.replace('\\', '/')
# single_image_width = 720
# single_image_height = 480
# view_cluster_sequence = 38


def construct_output_title(ikasl_param):
    global output_image_title
    global input_folder_path
    output_image_title += '\nGSOM-SF - ' + str(ikasl_param.get_gsom_parameters().SPREAD_FACTOR)
    output_image_title += '\nGSOM-Learn and smooth itr - ' + str(ikasl_param.get_gsom_parameters().LEARNING_ITERATIONS)
    output_image_title += '\nGSOM-Neigh Radius - ' + str(ikasl_param.get_gsom_parameters().MAX_NEIGHBOURHOOD_RADIUS)
    output_image_title += '\nGSOM-Distance Func - ' + str(ikasl_param.get_gsom_parameters().DISTANCE_FUNCTION)
    output_image_title += '\nIKASL-HT Fraction - ' + str(ikasl_param.get_hit_threshold_fraction())
    output_image_title += '\nGSOM-Aggr Function - ' + str(ikasl_param.get_aggregation_function())
    output_image_title += '\nGSOM-Aggr Neigh Radius - ' + str(ikasl_param.get_aggregate_proximity())
    output_image_title += '\nInput file folder - ' + input_folder_path


if __name__ == '__main__':

    if mode == 1:

        print('Start running IKASL algorithm.')

        # Init GSOM Parameters
        """
        GSOMParameters(spread_factor, learning_itr, smooth_itr, max_neighbourhood_radius=4, start_learning_rate=0.3,
                 smooth_neighbourhood_radius_factor=0.5, smooth_learning_factor=0.5, distance='EUC', fd=0.1,
                 alpha=0.9, r=0.95)

        IKASLParameters(self, gsom_parameters, aggregate_proximity=2, hit_threshold_fraction=0.05, 
                        aggregate_function='AVG', aggregate_inside_hitnode_proximity=True)
        """
        gsom_params = Params.GSOMParameters(0.83, 100, 100, distance=Elements.DistanceFunction.COSINE)
        ikasl_params = Params.IKASLParameters(gsom_params, aggregate_proximity=2, hit_threshold_fraction=0.05,
                                              aggregate_function=Elements.AggregateFunction.PROXIMITY_AVERAGE,
                                              aggregate_inside_hitnode_proximity=False)

        construct_output_title(ikasl_params)

        # Process the input files
        input_vector_database, labels = Parser.InputParser.parse_input(input_folder_path)

        # Process the IKASL algorithm
        print('Starting IKASL ...')
        ikasl = Core_IKASL.IKASL(ikasl_params)
        ikasl_start = time.time()
        ikasl_sequence = ikasl.run(input_vector_database)
        print('IKASL algorithms completed in', round(time.time() - ikasl_start, 2), '(s)')

        saved_name = Utils.Utilities.save_object(ikasl_sequence, output_save_location + output_save_filename)
        display = Display_Utils.Display(ikasl_sequence)
        display.display_tree(output_image_title, saved_name, weight_labels=labels, labels_to_show=labels_to_show,
                             enable_style=enable_style, save_format=display_output_format)
        # Note: commenting the code due to large image data set. This is just for visualisation purposes.
        # display.view_clusters(image_files_root_folder, single_image_width, single_image_height, view_cluster_sequence)

        print('Visualisation saved in output folder.')

    elif mode == 2:

        # Process the input files
        labels = Parser.InputParser.get_labels(labels_file)

        # Only Display
        filename = output_save_location + display_filename
        ikasl_seq = Utils.Utilities.load_object(filename)
        display = Display_Utils.Display(ikasl_seq)
        display.display_tree(output_image_title, filename, weight_labels=labels, labels_to_show=labels_to_show,
                             enable_style=enable_style, save_format=display_output_format)
        # display.view_clusters(image_files_root_folder, single_image_width, single_image_height, view_cluster_sequence)

        print('Visualisation saved.')
