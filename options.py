class GaussianSplattingOptions:
    """
    Options for Gaussian Splatting
    """

    def __init__(self):
        # initial opacity value (sigmoid activation)
        self.initial_opacity_value = 0.5  # initial opacity value (sigmoid activation)
        # number of neighbors used to compute initial scale
        self.initial_scale_num_neighbors = 3
        # scaling factor for initial scale:  log(mean_neighbor_dist * factor = initial_scale)
        self.mean_neighbor_dist_to_initial_scale_factor = 0.4
