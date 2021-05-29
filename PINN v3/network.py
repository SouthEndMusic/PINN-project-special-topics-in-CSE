import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from itertools import product

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


    def _construct_branch(self, layer_widths, inputs):
        """Construct a single branch of the network."""

        x = tf.keras.layers.Dense(
                layer_widths[0], activation = self.activ_function,
                kernel_initializer = 'he_normal')(inputs)

        for width in layer_widths[1:]:

            x = tf.keras.layers.Dense(
                width, activation = self.activ_function,
                kernel_initializer = 'he_normal')(x)

            
        x = tf.keras.layers.Dense(
            1, activation = self.activ_function,
            kernel_initializer = 'he_normal')(x)
        
        return x

    def plot_model(self,
                    branch_colors = ['r','g','b'],
                    spacing_x     = 1,
                    spacing_y     = 1,
                    lw            = 1,
                    show_axis     = False,
                    figsize       = (5,5)):
        """Plot the neural network structure."""

        fig, ax       = plt.subplots(figsize = figsize)
        layer_counter = 0 

        # The connections between the nodes
        connections  = [] 
        
        # Dict with: node name_layer -> location
        node_dict = {'input_x_0': np.array([0, spacing_y/2]),
                     'input_y_0': np.array([0,-spacing_y/2])}

        # The fully connected (stem) layers
        for layer_width in self.stem_widths:
            layer_counter += 1
            
            for i in range(layer_width):

                node_code = f'fully_connected_{i}_{layer_counter}'
                node_dict[node_code] = np.array([layer_counter*spacing_x,
                                                 ((layer_width-1)/2-i)*spacing_y])


                for node_name in node_dict.keys():
                    if node_name.split('_')[-1] == str(layer_counter-1):
                        connections.append((node_code,node_name))

        # The branches
        branch_mids  = np.array([1,0,-1])*np.max(self.branch_widths)*spacing_y
        for branch_width in self.branch_widths + [1]:
            layer_counter += 1
            
            for branch, branch_mid in enumerate(branch_mids):

                for i in range(branch_width):

                    node_code = f'branch_{branch}_{i}_{layer_counter}'
                    node_dict[node_code] = np.array([layer_counter*spacing_x,
                                                     branch_mid + ((branch_width-1)/2-i)*spacing_y])

                    # First layer of branch
                    if layer_counter == len(self.stem_widths)+1:
                        for node_name in node_dict.keys():
                            if node_name.split('_')[-1] == str(layer_counter-1):
                                connections.append((node_code,node_name))

                    # Further layer of branch
                    else:
                        for node_name in node_dict.keys():
                            node_name_parts = node_name.split('_')
                            if node_name_parts[1] == str(branch) and node_name_parts[-1] == str(layer_counter-1):
                                connections.append((node_code,node_name))


        # Plotting nodes
        for branch,branch_color in enumerate(branch_colors):
            branch_nodes = [name for name in node_dict.keys() if name.startswith(f'branch_{branch}')]
            coords_nodes = np.array([node_dict[name] for name in branch_nodes])
            ax.scatter(coords_nodes[:,0], coords_nodes[:,1], zorder = 1, c = branch_color)

        branch_nodes = [name for name in node_dict.keys() if not name.startswith('branch')]
        coords_nodes = np.array([node_dict[name] for name in branch_nodes])
        ax.scatter(coords_nodes[:,0], coords_nodes[:,1], zorder = 1, c = 'k')
        
        # Plotting connections
        for connection in connections:
            xy = np.array([node_dict[node] for node in connection]).T
            ax.plot(xy[0],xy[1], c = 'k', zorder = 0, lw = lw)

        # Adding annotations
        for txt, loc in zip(['$x$','$y$'], [node_dict[name] for name in ['input_x_0',
                                                                         'input_y_0']]):
            ax.annotate(txt,loc, xytext = loc - np.array([spacing_x/4,0]))

        for txt, loc in zip(['$v_1$', '$v_2$', '$p$'], [node_dict[name] for name in [f'branch_{i}_0_{layer_counter}'
                                                                                     for i in range(3)]]):
            ax.annotate(txt,loc, xytext = loc - np.array([-spacing_x/8,0]))
        

        ax.set_title('Stokesflow network architecture')

        if not show_axis:
            ax.axis('off')

        ax.set_aspect('equal','box')
        plt.tight_layout()
        fig.canvas.draw()
        plt.show()


    def swish_activation(self,x):
        """Swish activation function."""

        return x * tf.math.sigmoid(x)
        
    def assemble_full_model(self):
        """Assemble the full network."""        

        # model
        inputs = tf.keras.layers.Input(shape = 2)

        if len(self.stem_widths) > 0:
            stem         = self._construct_branch(self.stem_widths,inputs)
            branch_input = stem
        else:
            branch_input = inputs
            
        v1_branch = self._construct_branch(self.branch_widths,branch_input)
        v2_branch = self._construct_branch(self.branch_widths,branch_input)
        p_branch  = self._construct_branch(self.branch_widths,branch_input)
        model     = tf.keras.models.Model(inputs  = inputs,
                                          outputs = [v1_branch,
                                                     v2_branch,
                                                     p_branch],
                                          name    = 'stokesflow_branched')

        return model

if __name__ == "__main__":

    C = model_stokesflow_branched()
    C.plot_model(show_axis = True, spacing_x = 5)
