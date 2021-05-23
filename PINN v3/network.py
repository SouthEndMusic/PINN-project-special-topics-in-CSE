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
                 activation_function = tf.nn.tanh):
        """Construct branched network, inspired by
        https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178"""

        self.branch_widths  = branch_widths
        self.activ_function = activation_function


    def _construct_branch(self, layer_widths, inputs):
        """Construct a single branch of the network."""

        x = tf.keras.layers.Dense(
                layer_widths[0], activation = self.activ_function,
                kernel_initializer = 'glorot_normal')(inputs)

        for width in layer_widths[1:]:

            x = tf.keras.layers.Dense(
                width, activation = self.activ_function,
                kernel_initializer = 'glorot_normal')(x)

            
        x = tf.keras.layers.Dense(
            1, activation = self.activ_function,
            kernel_initializer = 'glorot_normal')(x)
        
        return x

    def plot_model(self,
                    branch_colors = ['r','g','b'],
                    lw = 1):
        """Plot the neural network structure."""

        fig, ax = plt.subplots()

        # The middle (y) of each branch
        branch_mids  = np.array([-1,0,1])*max(self.branch_widths)
        branch_names = ['$p$','$v_2$','$v_1$']

        for branch,(color,branch_name) in enumerate(zip(branch_colors,branch_names)):

            # The input layer
            x_values = np.array([0,0])
            y_values = np.array([-1,1])/2

            if branch == 0:
                ax.scatter(x_values,y_values, color = 'k')

                for x,y,txt in zip(x_values,y_values,['$y$','$x$']):
                    ax.annotate(txt, (x,y), xytext = (-1/4,y))

            prev_x_values = x_values
            prev_y_values = y_values
            branch_mid    = branch_mids[branch]

            for layer,width in enumerate(self.branch_widths):

                x_values = np.full((width,),layer+1)
                y_values = np.arange(width) + branch_mid - (width-1)/2

                # Plotting the nodes
                ax.scatter(x_values,y_values, color = color, zorder = 1)

                # Plotting the connections
                for j,j_prev in product(range(len(x_values)),range(len(prev_x_values))):
                    ax.plot([prev_x_values[j_prev],x_values[j]],
                            [prev_y_values[j_prev],y_values[j]], c = 'k', zorder = 0, lw = lw)

                prev_x_values = x_values
                prev_y_values = y_values

            output_x = layer+2
            output_y = branch_mid

            ax.scatter([output_x],[output_y], color = color, zorder = 1)
            ax.annotate(branch_name, (output_x,output_y), xytext = (output_x+1/8,output_y))

            for j_prev in range(len(prev_x_values)):
                ax.plot([prev_x_values[j_prev],layer+2],
                        [prev_y_values[j_prev],branch_mid], c = 'k', zorder = 0, lw = lw)

        ax.set_title('Stokesflow network architecture')
        ax.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        plt.show()
        
    def assemble_full_model(self):
        """Assemble the full network."""        

        # model
        inputs    = tf.keras.layers.Input(shape = 2)
        v1_branch = self._construct_branch(self.branch_widths,inputs)
        v2_branch = self._construct_branch(self.branch_widths,inputs)
        p_branch  = self._construct_branch(self.branch_widths,inputs)
        model     = tf.keras.models.Model(inputs  = inputs,
                                          outputs = [v1_branch,
                                                     v2_branch,
                                                     p_branch],
                                          name    = 'stokesflow_branched')

        return model

if __name__ == "__main__":

    C = model_stokesflow_branched(branch_widths = 5*[5])
    C.plot_model()
