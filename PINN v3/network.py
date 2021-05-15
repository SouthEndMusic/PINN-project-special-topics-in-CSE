import numpy as np
import tensorflow as tf

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
