import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import tensorflow as tf

from itertools import product
from custom_cbar import *

def construct_model_warmup(hidden_layer_widths = np.array([40,40,40]),
                    activation_function = tf.nn.tanh):
    """Construct the sequential NN in keras."""

    # Sequential model
    model = tf.keras.Sequential()
    
    # Input layer (input: 2 spatial dimensions)
    model.add(tf.keras.layers.InputLayer(input_shape = 2))

    # Hidden layers
    for width in hidden_layer_widths:
        model.add(tf.keras.layers.Dense(
                width, activation = activation_function,
                kernel_initializer = 'glorot_normal'))

    # Output layer (1 value)
    model.add(tf.keras.layers.Dense(
              1, activation = activation_function,
              kernel_initializer = 'glorot_normal'))

    return model



def construct_model_stokesflow(hidden_layer_widths = np.array([40,40,40]),
                    activation_function = tf.nn.tanh):
    """Construct the sequential NN in keras."""

    # Sequential model
    model = tf.keras.Sequential()
    
    # Input layer (input: 2 spatial dimensions)
    model.add(tf.keras.layers.InputLayer(input_shape = 2))

    # Hidden layers
    for width in hidden_layer_widths:
        model.add(tf.keras.layers.Dense(
                width, activation = activation_function,
                kernel_initializer = 'glorot_normal'))

    # Output layer (3 values: pressure + 2 velocity components)
    model.add(tf.keras.layers.Dense(
              3, activation = activation_function,
              kernel_initializer = 'glorot_normal'))

    return model



class model_stokesflow_branched():

    def __init__(self,
                 branch_widths       = 10*[5],
                 stem_widths         =  2*[5],
                 activation_function = 'swish'):
        """Construct branched network, inspired by
        https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178"""

        self.branch_widths = branch_widths
        self.stem_widths   = stem_widths

        if activation_function == 'swish':
            self.activ_function = self.swish_activation
        else:
            self.activ_function = activation_function




    def plot_model(self,
                    branch_colors = ['r','g','b'],
                    spacing_x     = 1,
                    spacing_y     = 1,
                    lw            = 1,
                    show_axis     = False,
                    figsize       = (5,5),
                    weight_cmap   = 'coolwarm'):
        """Plot the neural network structure."""

        if not hasattr(self, 'model'):
            raise ValueError("The tf model must be constructed first.")

        branch_names_fancy = ['$v_1$', '$v_2$', '$p$']
        branch_names       = ['v1', 'v2', 'p']

        fig, ax       = plt.subplots(figsize = figsize)
        layer_counter = 0 

        # The connections between the nodes (in the order (right,left))
        connections  = [] 
        
        # Dict with: node name_layer -> location
        node_dict = {'input_0_0': np.array([0, spacing_y/2]),
                     'input_0_1': np.array([0,-spacing_y/2])}

        # The fully connected (stem) layers
        for layer_width in self.stem_widths:
            layer_counter += 1
            
            for i in range(layer_width):

                node_code = f'stem_{layer_counter}_{i}'
                node_dict[node_code] = np.array([layer_counter*spacing_x,
                                                 ((layer_width-1)/2-i)*spacing_y])


                for node_name in node_dict.keys():
                    if node_name.split('_')[1] == str(layer_counter-1):
                        connections.append((node_code,node_name))

        # The branches
        branch_mids  = np.array([1,0,-1])*np.max(self.branch_widths)*spacing_y
        for branch_width in self.branch_widths + [1]:
            layer_counter += 1
            
            for branch_name, branch_mid in zip(branch_names,branch_mids):

                for i in range(branch_width):

                    node_code = f'{branch_name}-branch_{layer_counter}_{i}'
                    node_dict[node_code] = np.array([layer_counter*spacing_x,
                                                     branch_mid + ((branch_width-1)/2-i)*spacing_y])

                    # First layer of branch
                    if layer_counter == len(self.stem_widths)+1:
                        for node_name in node_dict.keys():
                            if node_name.split('_')[1] == str(layer_counter-1):
                                connections.append((node_code,node_name))

                    # Further layer of branch
                    else:
                        for node_name in node_dict.keys():
                            node_name_parts = node_name.split('_')
                            if node_name_parts[0].startswith(branch_name) and node_name_parts[1] == str(layer_counter-1):
                                connections.append((node_code,node_name))


        # Plotting branch nodes
        for branch_name,branch_color in zip(branch_names,branch_colors):
            branch_nodes = [name for name in node_dict.keys() if name.startswith(f'{branch_name}-branch')]
            coords_nodes = np.array([node_dict[name] for name in branch_nodes])
            ax.scatter(coords_nodes[:,0], coords_nodes[:,1], zorder = 1, c = branch_color)

        # Plotting stem and input nodes
        stem_nodes   = [name for name in node_dict.keys() if name.startswith('stem') or name.startswith('input')]
        coords_nodes = np.array([node_dict[name] for name in stem_nodes])
        ax.scatter(coords_nodes[:,0], coords_nodes[:,1], zorder = 1, c = 'k')

        # Weight cmap
        weight_cmap = cm.get_cmap(weight_cmap)

        # Computing minimum and maximum weight
        min_weight = 1e100
        max_weight = -1e100
        
        for W in self.model.weights:
            min_W = tf.math.reduce_min(W)
            max_W = tf.math.reduce_max(W)
            
            if min_W < min_weight:
                min_weight = min_W.numpy()

            if max_W > max_weight:
                max_weight = max_W.numpy()  
        
        # Plotting connections
        for connection in connections:

            # Get connection weight
            node_name_parts_right = connection[0].split('_')
            node_name_parts_left  = connection[1].split('_')

            index_right = int(node_name_parts_right[2])
            index_left  = int(node_name_parts_left[2])
            
            layer  = self.model.get_layer('_'.join(node_name_parts_right[:2]))
            weight = layer.get_weights()[0][index_left,index_right]
            
            xy = np.array([node_dict[node] for node in connection]).T
            ax.plot(xy[0],xy[1], zorder = 0, lw = lw,
                    color = weight_cmap(weight-min_weight/(max_weight-min_weight)))

        # Adding annotations input layers
        for txt, loc in zip(['$x$','$y$'], [node_dict[name] for name in ['input_0_0',
                                                                         'input_0_1']]):
            ax.annotate(txt,loc, xytext = loc - np.array([spacing_x/4,0]))

        # Adding annotations output layers
        output_node_codes = [f'{branch_names[i]}-branch_{layer_counter}_0' for i in range(3)]
        output_node_locs  = [node_dict[name] for name in output_node_codes]
        
        for branch_name, loc in zip(branch_names_fancy, output_node_locs):
            ax.annotate(branch_name,loc, xytext = loc - np.array([-spacing_x/8,0]))
        

        ax.set_title('Stokesflow network architecture')

        if not show_axis:
            ax.axis('off')

        self.cbar = custom_colorbar(ax, cmap = weight_cmap, label = 'Weights')
        self.cbar.update_data([min_weight,max_weight])

        ax.set_aspect('equal','box')
        plt.tight_layout()
        fig.canvas.draw()
        plt.show()
        
        return fig


    def swish_activation(self,x):
        """Swish activation function."""

        return x * tf.math.sigmoid(x)


    def _construct_branch(self, layer_widths, inputs,
                          type_name,
                          layer_start = 0):
        """Construct a single branch of the network."""

        x = tf.keras.layers.Dense(
                layer_widths[0], activation = self.activ_function,
                kernel_initializer = 'he_normal',
                name = f'{type_name}_{layer_start}')(inputs)

        for i,width in enumerate(layer_widths[1:]):

            x = tf.keras.layers.Dense(
                width, activation = self.activ_function,
                kernel_initializer = 'he_normal',
                name = f'{type_name}_{layer_start+i+1}')(x)
        
        return x

        
    def assemble_full_model(self):
        """Assemble the full network."""        

        # model
        inputs = tf.keras.layers.Input(shape = 2, name = 'Input_layer_0')

        if len(self.stem_widths) > 0:
            stem         = self._construct_branch(self.stem_widths,inputs,'stem', layer_start = 1)
            branch_input = stem
        else:
            branch_input = inputs

        branch_widths_with_end = self.branch_widths+[1]
        branch_layer_start     = len(self.stem_widths) + 1

            
        v1_branch = self._construct_branch(branch_widths_with_end,branch_input,'v1-branch', layer_start = branch_layer_start)
        v2_branch = self._construct_branch(branch_widths_with_end,branch_input,'v2-branch', layer_start = branch_layer_start)
        p_branch  = self._construct_branch(branch_widths_with_end,branch_input,'p-branch',  layer_start = branch_layer_start)
        model     = tf.keras.models.Model(inputs  = inputs,
                                          outputs = [v1_branch,
                                                     v2_branch,
                                                     p_branch],
                                          name    = 'Stokesflow_branched')

        self.model = model
        return model

if __name__ == "__main__":

    C = model_stokesflow_branched(stem_widths = 3*[4], branch_widths = 3*[3])
    C.assemble_full_model()
    C.model.summary()
    W = C.model.get_weights()
    C.plot_model(spacing_x = 5, figsize = (7,5))
