import math
from core import elements as Elements


class GSOMParameters:
    def __init__(self, spread_factor, learning_itr, smooth_itr, max_neighbourhood_radius=4, start_learning_rate=0.3,
                 smooth_neighbourhood_radius_factor=0.5, smooth_learning_factor=0.5, learn_smooth_sample_size=-1,
                 distance=Elements.DistanceFunction.EUCLIDEAN, distance_divider=-1, fd=0.1, alpha=0.9, r=3.8):
        self.SPREAD_FACTOR = spread_factor
        self.LEARNING_ITERATIONS = learning_itr
        self.SMOOTHING_ITERATIONS = smooth_itr
        self.MAX_NEIGHBOURHOOD_RADIUS = max_neighbourhood_radius
        self.START_LEARNING_RATE = start_learning_rate
        self.SMOOTHING_LEARNING_RATE_FACTOR = smooth_learning_factor
        self.SMOOTHING_NEIGHBOURHOOD_RADIUS_FACTOR = smooth_neighbourhood_radius_factor
        self.LEARN_SMOOTH_SAMPLE_SIZE = learn_smooth_sample_size
        self.FD = fd
        self.R = r
        self.ALPHA = alpha
        self.DISTANCE_FUNCTION = distance
        self.DISTANCE_DIVIDER = distance_divider

    def get_gt(self, dimensions):
        return -dimensions * math.log(self.SPREAD_FACTOR)

    def update_R_for_one_starting_node(self):
        self.R = 0.95

    def get_learn_smooth_sample_size(self, input_length):
        return input_length if self.LEARN_SMOOTH_SAMPLE_SIZE == -1 else self.LEARN_SMOOTH_SAMPLE_SIZE


class IKASLParameters:
    def __init__(self, gsom_parameters, aggregate_proximity=2, hit_threshold_fraction=0.05,
                 aggregate_function=Elements.AggregateFunction.AVERAGE,
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
