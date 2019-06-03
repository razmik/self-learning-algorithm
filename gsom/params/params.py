import math
from enum import Enum


class DistanceFunction(Enum):
    COSINE = 'cosine'
    EUCLIDEAN = 'euclidean'
    COMBINED = 'combined'


class AggregateFunction(Enum):
    AVERAGE = 'average'
    MAX = 'max'
    FUZZY = 'fuzzy'
    PROXIMITY_AVERAGE = 'proximity_average'


class GSOMParameters:
    def __init__(self, spread_factor, learning_itr, smooth_itr, age_threshold=200, max_neighbourhood_radius=4,
                 start_learning_rate=0.3, smooth_neighbourhood_radius_factor=0.5, smooth_learning_factor=0.5,
                 learn_smooth_sample_size=-1, distance=DistanceFunction.EUCLIDEAN, distance_divider=-1, fd=0.1,
                 alpha=0.9, r=3.8, tau_b=0.3, tau_n=0.1, temporal_context_count=2, beta=0.7, forget_itr_count=2):

        # Compulsory Parameters
        self.SPREAD_FACTOR = spread_factor
        self.LEARNING_ITERATIONS = learning_itr
        self.SMOOTHING_ITERATIONS = smooth_itr

        # Preset Parameters
        self.AGE_THRESHOLD = age_threshold
        self.FORGET_ITR_COUNT = forget_itr_count
        self.MAX_NEIGHBOURHOOD_RADIUS = max_neighbourhood_radius
        self.START_LEARNING_RATE = start_learning_rate
        self.SMOOTHING_LEARNING_RATE_FACTOR = smooth_learning_factor
        self.SMOOTHING_NEIGHBOURHOOD_RADIUS_FACTOR = smooth_neighbourhood_radius_factor
        self.LEARN_SMOOTH_SAMPLE_SIZE = learn_smooth_sample_size
        self.FD = fd
        self.R = r
        self.ALPHA = alpha

        # Distance Measures
        self.DISTANCE_FUNCTION = distance
        self.DISTANCE_DIVIDER = distance_divider

        # Habituation Parameters
        self.TAU_B = tau_b
        self.TAU_N = tau_n

        # Recurrent Self-organization Parameters
        self.NUMBER_OF_TEMPORAL_CONTEXTS = temporal_context_count  # [0] is default weight. i.e. min value should be 1.
        self.BETA = beta

    def get_gt(self, dimensions):
        return -dimensions * math.log(self.SPREAD_FACTOR)

    def update_R_for_one_starting_node(self):
        self.R = 0.95

    def get_learn_smooth_sample_size(self, input_length):
        return input_length if self.LEARN_SMOOTH_SAMPLE_SIZE == -1 else self.LEARN_SMOOTH_SAMPLE_SIZE

    def setup_age_threshold(self, input_data_len):
        self.AGE_THRESHOLD = input_data_len * self.FORGET_ITR_COUNT


class GeneraliseParameters:
    def __init__(self, gsom_parameters, aggregate_proximity=2, hit_threshold_fraction=0.05,
                 aggregate_function=AggregateFunction.AVERAGE,
                 aggregate_inside_hitnode_proximity=True, sugeno_lambda=0.05):
        self.gsom_parameters = gsom_parameters
        self.hit_threshold_fraction = hit_threshold_fraction
        self.aggregate_proximity = aggregate_proximity
        self.aggregate_function = aggregate_function
        self.aggregate_inside_hitnode_proximity = aggregate_inside_hitnode_proximity
        self.sugeno_lambda = sugeno_lambda

    def get_gsom_parameters(self):
        return self.gsom_parameters

    def get_hit_threshold_fraction(self):
        return self.hit_threshold_fraction

    def get_aggregate_proximity(self):
        return self.aggregate_proximity

    def get_aggregation_function(self):
        return self.aggregate_function

    def get_sugeno_lambda(self):
        return self.sugeno_lambda

    def is_aggregate_inside_hitnode_proximity(self):
        return self.aggregate_inside_hitnode_proximity

    def setup_age_threshold(self, input_data_len):
        self.gsom_parameters.setup_age_threshold(input_data_len)
