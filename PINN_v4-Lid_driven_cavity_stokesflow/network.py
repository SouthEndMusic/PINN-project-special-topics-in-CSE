import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from itertools import product
from custom_cbar import *

@tf.function
def swish(x):
    return x * tf.math.sigmoid(x)


class construct_network():
    """Class to construct the model for the lid driven cavity stokesflow PINN.
    Inspired by
    https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178.

    Parameters
    ----------
    stem_widths : list
        List of integers, determines the widths of the dense layers in the stem of the network.
    branch_widths : list
        List of integers, determines the widths of the dense branches of the network (all branches are the same).
    activation function : function 'float -> float' or 'swish'
        The activation function used for all nodes, defaults to 'swish'.


    Outputs
    -------
    Model
        Branched tensorflow model for the lid driven cavity stokesflow PINN.
    """

    def __init__(self,
                 stem_widths   : list = 2*[5],
                 branch_widths : list = 4*[3],
                 activation_function = 'swish'):

        self.branch_widths = branch_widths
        self.stem_widths   = stem_widths

        if activation_function == 'swish':
            self.activ_function = swish
        else:
            self.activ_function = activation_function

        self.assemble_full_network()



    def _construct_branch(self,
                          inputs,
                          layer_widths,
                          type_name,
                          layer_start = 1):
        """Construct a branch (or the stem) of the network.

        Parameters
        ----------
        Inputs : Keras layers
            What comes before this branch in the network
        layer_widths : list
            List of integers, determines the widths of the dense layers in the branch
        type_name : str
            Name of the branch, i.e. either the stem or one of the branches
        layer_start : int
            Index of the first layer in this branch, counting from the input layer having index 0
            
        Outputs
        -------
        branch : Keras sequential layers
            The dense sequential branch
        """

        branch = tf.keras.layers.Dense(
            layer_widths[0],
            activation = self.activ_function,
            kernel_initializer = 'he_normal',
            name = f'{type_name}_{layer_start}'
        )(inputs)

        for i, width in enumerate(layer_widths[1:]):

            branch = tf.keras.layers.Dense(
                width,
                activation = self.activ_function,
                kernel_initializer = 'he_normal',
                name = f'{type_name}_{layer_start+i+1}'
            )(branch)

        return branch



    def assemble_full_network(self):
        """Construct the full network."""

        input_layer = tf.keras.layers.Input(shape = (2,), name = 'Input_layer_0')

        # Construct stem of network or skip if it has length 0
        if len(self.stem_widths) > 0:

            stem = self._construct_branch(
                input_layer,
                self.stem_widths,
                'stem',
                layer_start = 1
            )
            
            branch_input = stem 

        else:
            branch_input = input_layer

        # Add layer of width 1, i.e. the output layer, to the branch widths
        branch_widths_with_output = self.branch_widths + [1]

        # The index of the first layer of each branch, counting from the input layer having index 0
        branch_layer_start = len(self.stem_widths) + 1

        v1_branch = self._construct_branch(
            branch_input,
            branch_widths_with_output,
            'v1-branch',
            layer_start = branch_layer_start
        )
        
        v2_branch = self._construct_branch(
            branch_input,
            branch_widths_with_output,
            'v2-branch',
            layer_start = branch_layer_start
        )
        
        p_branch = self._construct_branch(
            branch_input,
            branch_widths_with_output,
            'p-branch',
            layer_start = branch_layer_start
        )
        
        model = tf.keras.models.Model(
            inputs = input_layer,
            outputs = [v1_branch,v2_branch,p_branch],
            name = 'Stokesflow_branched_PINN'
        )

        self.model = model


        

class vizualize_network():
    """Class to vizualize the below network

    Parameters
    ----------
    model : Keras model
        A keras model as the one constructed above
    figsize : (float, float)
        Tuple of integers passed to matplotib to set the figure size
    spacing_x : float
        The distance between the nodes in the x direction in the plot
    spacing_y : float
        The distance between the nodes in the y direction in the plot
    linewidth : float
        With of the weight lines
    nodesize : float
        Size of the nodes
    show_axis : bool
        Whether the axes with numbers should be shown
    """

    def __init__(self,
                 figsize   = (5,5),
                 spacing_x = 1,
                 spacing_y = 1,
                 linewidth = 1,
                 nodesize  = 10,
                 show_axes = False):

        self.cmap_weights = cm.get_cmap('coolwarm')
        self.lw           = linewidth
        self.nodesize     = nodesize
        
        self.fig_network, self.ax_network = plt.subplots(figsize = figsize)
        self.__init_plotting(spacing_x,
                             spacing_y,
                             show_axes)


    def __init_plotting(self,
                        spacing_x,
                        spacing_y,
                        show_axes):
        """Initiate the matplotlib objects that can be updated."""

        branch_names_fancy = ['$v_2$', '$v_1$', '$p$']
        branch_names       = ['v2', 'v1', 'p']
        branch_colors      = ['r', 'g', 'b']

        # The connections between the nodes (in the order (right,left))
        connections = []

        # Dict with: node data -> node location
        # The node data is (type,layer_index,node_index_in_layer)
        node_dict = {('input',0,0) : np.array([0,  spacing_y/2]),
                     ('input',0,1) : np.array([0, -spacing_y/2])}

        # the Stem

        # The nodes of the stem layers
        for i,layer_width in enumerate(self.stem_widths):

            layer_index = i+1
            layer_top   = (layer_width-1)/2

            for node_index in range(layer_width):

                node_data = ('stem',layer_index,node_index)
                node_dict[node_data] = np.array([layer_index          *spacing_x,
                                                (layer_top-node_index)*spacing_y])

        # The connections of the stem layers
        for i,(layer_width, prev_layer_width) in enumerate(zip(self.stem_widths,
                                                               [2]+self.stem_widths)):

            if i == 0:
                prev_type = 'input'
            else:
                prev_type = 'stem'


            for index, index_prev in product(range(layer_width),
                                             range(prev_layer_width)):

                connections.append((
                    ('stem',i+1,index),
                    (prev_type,i,index_prev)
                ))

        # The branches
        branch_mids        = np.array([-1,0,1])*np.max(self.branch_widths)*spacing_y
        branch_start_index = len(self.stem_widths) + 1
        
        for branch_name, branch_mid in zip(branch_names, branch_mids):

            layer_index = branch_start_index - 1

            for branch_width in self.branch_widths + [1]:
                layer_index += 1
                layer_top    = (branch_width-1)/2

                for node_index in range(branch_width):

                    node_data = (branch_name, layer_index, node_index)
                    node_dict[node_data] = np.array([layer_index                        *spacing_x,
                                                     branch_mid + (layer_top-node_index)*spacing_y])

            # The connections of this branch
            for i, (layer_width, prev_layer_width) in enumerate(zip(self.branch_widths + [1],
                                                                    [self.stem_widths[-1]] + self.branch_widths)):


                if i == 0:
                    prev_type = 'stem'
                else:
                    prev_type = branch_name

                for index, index_prev in product(range(layer_width),
                                                 range(prev_layer_width)):

                    connections.append((
                        (branch_name, branch_start_index + i, index),
                        (prev_type, branch_start_index + i - 1, index_prev)
                    ))
        
        # Plotting stem nodes
        stem_nodes   = [node_data for node_data in node_dict.keys() if node_data[0] in ['stem','input']]
        coords_nodes = np.array([node_dict[node_data] for node_data in stem_nodes])
        self.ax_network.scatter(coords_nodes[:,0], coords_nodes[:,1], c = 'k', zorder = 1, s = self.nodesize)

        # Plotting branch nodes
        for branch_name, branch_color in zip(branch_names, branch_colors):
            branch_nodes = [node_data for node_data in node_dict.keys() if node_data[0] == branch_name]
            coords_nodes = np.array([node_dict[node_data] for node_data in branch_nodes])
            self.ax_network.scatter(coords_nodes[:,0], coords_nodes[:,1], c = branch_color, zorder = 1, s = self.nodesize)

        # Plotting connections
        weight_lines = dict()
        
        for connection in connections:

            xy = np.array([node_dict[node_data] for node_data in connection]).T
            weight_lines[connection], = self.ax_network.plot(xy[0],xy[1], zorder = 0, color = 'k', lw = self.lw)

        self.node_dict    = node_dict
        self.connections  = connections
        self.weight_lines = weight_lines

        # Adding annotations input layers
        input_layer_locs = [node_dict[name] for name in [('input',0,0),('input',0,1)]]
        for txt, loc in zip(['$x$','$y$'], input_layer_locs):
            self.ax_network.annotate(txt,loc, xytext = loc - np.array([spacing_x/4,0]))

        # Adding annotations output layers
        output_nodes = [(branch_name,layer_index,0) for branch_name in branch_names]
        coords_nodes = [node_dict[node_data] for node_data in output_nodes]
        
        for branch_name, loc in zip(branch_names_fancy, coords_nodes):
            self.ax_network.annotate(branch_name,loc, xytext = loc - np.array([-spacing_x/8,0]))

        self.network_cbar = custom_colorbar(self.ax_network, cmap = self.cmap_weights, label = 'Weights')

        if not show_axes:
            self.ax_network.axis('off')

        self.ax_network.set_title('Stokesflow network architecture')
        self.ax_network.set_aspect('equal','box')

    def update_weights_plot(self):
        """Update the colors in the plot that represent the weights."""

        # Compute minimum and maximum weight
        min_weight = 1e100
        max_weight = -1e100
        
        for W in self.model.weights:
            min_W = tf.math.reduce_min(W)
            max_W = tf.math.reduce_max(W)
            
            if min_W < min_weight:
                min_weight = min_W.numpy()

            if max_W > max_weight:
                max_weight = max_W.numpy()

        self.network_cbar.update_data([min_weight,max_weight])

        for connection in self.connections:

            layer_type = connection[0][0]

            if not layer_type == 'stem':
                layer_type = layer_type + '-branch'

            index_right = connection[0][2]
            index_left  = connection[1][2]

            layer  = self.model.get_layer(f'{layer_type}_{connection[0][1]}')
            weight = layer.get_weights()[0][index_left, index_right]

            normalized_weight = (weight-min_weight)/(max_weight-min_weight)

            self.weight_lines[connection].set_color(self.cmap_weights(normalized_weight))

        self.fig_network.canvas.draw()

        
        
