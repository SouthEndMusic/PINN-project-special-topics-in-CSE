import numpy as np
import tensorflow as tf
import tqdm.notebook as tq

from collections import namedtuple

loss_types = ['total',
              'incompressibility',
              'pressure_average',
              'interior_1',
              'interior_2',
              'boundaries_v1',
              'boundaries_v2']

loss_weight_tuple = namedtuple('loss_weights', loss_types[1:])
loss_lists_tuple  = namedtuple('loss_values',  loss_types)

output_types       = ['p', 'v1', 'v2']
output_types_tuple = namedtuple('outputs', ' '.join(output_types))

class training():
    """Class to perform the training of the PINN.


    Parameters
    ----------
    optimizer
        Keras optimizer
    loss_weights
        The weights of the various terms in the loss function. Make sure to normalize so these sum to 1
    """

    def __init__(self,
                 optimizer    = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08),
                 loss_weights = np.ones((6,))/6,
                 ):

        self.optimizer    = optimizer
        self.loss_weights = loss_weight_tuple(*loss_weights)
        self.loss_values  = loss_lists_tuple(*[[] for loss_type in loss_types])

        # Epoch counter
        self.epoch = 0



    def print_training_config(self):

        print(f'loss weights:\n{self.loss_weights}')
        print(f'\nOptimizer:\n{self.optimizer.get_config()}')

    

    def evaluate(self, inputs, training = False):
        """Evaluate the neural network in the given inputs."""

        u_NN = self.model(inputs, training = training)

        return output_types_tuple(*u_NN)


    @staticmethod
    def partition_outputs(outputs, partition):
        """Combine multiple outputs of the network in the named tuple form into one tuple."""
        partitioned_p  = tf.split(outputs.p,  partition, axis = 0)
        partitioned_v1 = tf.split(outputs.v1, partition, axis = 0)
        partitioned_v2 = tf.split(outputs.v2, partition, axis = 0)

        output_partition = []
        for data_p, data_v1, data_v2 in zip(partitioned_p, partitioned_v1, partitioned_v2):
            output_partition.append(output_types_tuple(data_p, data_v1, data_v2))

        return output_partition
        

    def __loss(self):
        """Compute the loss in one epoch for the generated sample points."""

        loss = 0
        
        samples_boundary_left   = tf.convert_to_tensor(self.samples_boundary[0])
        samples_boundary_right  = tf.convert_to_tensor(self.samples_boundary[1])
        samples_boundary_top    = tf.convert_to_tensor(self.samples_boundary[2])
        samples_boundary_bottom = tf.convert_to_tensor(self.samples_boundary[3])
        samples_interior        = tf.convert_to_tensor(self.samples_interior)
        samples_all             = tf.concat([
            samples_boundary_left,
            samples_boundary_right,
            samples_boundary_top,
            samples_boundary_bottom,
            samples_interior
            ], axis = 0)

        # Using GradientTape for computing the derivatives of the output of the model
        # with respect to the inputs
        with tf.GradientTape(persistent = True) as tape:
        
            tape.watch(samples_all)

            # Output of the network
            output_all             = self.evaluate(samples_all, training = True)
            output_partitioned     = self.partition_outputs(output_all, self.samples_per_boundary + [samples_interior.shape[0]])
            output_boundary_left   = output_partitioned[0]
            output_boundary_right  = output_partitioned[1]
            output_boundary_top    = output_partitioned[2]
            output_boundary_bottom = output_partitioned[3]
            output_interior        = output_partitioned[4]

            # Get the first order velocity derivatives (on the tape)
            grad_v1 = tape.gradient(output_all.v1, samples_all)
            grad_v2 = tape.gradient(output_all.v2, samples_all)

            dv1_dx = grad_v1[:,0]
            dv1_dy = grad_v1[:,1]
            dv2_dx = grad_v2[:,0]
            dv2_dy = grad_v2[:,1]

        # Incompressibility loss
        v_divergence = dv1_dx + dv2_dy
        loss_incompr = tf.reduce_mean(tf.square(v_divergence))
        loss        += self.loss_weights.incompressibility * loss_incompr

        # Pressure average loss
        loss_pressure_mean = tf.square(tf.reduce_mean(output_all.p))
        loss              += self.loss_weights.pressure_average * loss_pressure_mean

        # Interior loss
        p_gradient = tape.gradient(output_all.p, samples_all)
        dv1_dxx = tape.gradient(dv1_dx, samples_all)[:,0]
        dv1_dyy = tape.gradient(dv1_dy, samples_all)[:,1]
        dv2_dxx = tape.gradient(dv2_dx, samples_all)[:,0]
        dv2_dyy = tape.gradient(dv2_dy, samples_all)[:,1]
        laplace_v = tf.stack([dv1_dxx + dv1_dyy,
                              dv2_dxx + dv2_dyy], axis = 1)
        PDE_LHS_values  = -p_gradient + laplace_v
        loss_interior_1 = tf.reduce_mean(tf.square(PDE_LHS_values[:,0]))
        loss_interior_2 = tf.reduce_mean(tf.square(PDE_LHS_values[:,1]))
        loss           += self.loss_weights.interior_1 * loss_interior_1
        loss           += self.loss_weights.interior_2 * loss_interior_2

        # Boundary loss
        loss_boundaries_v1 = 0
        loss_boundaries_v2 = 0

        for output_boundary in [output_boundary_left,
                                output_boundary_right,
                                output_boundary_top,
                                output_boundary_bottom]:

            # Special case for the top boundary of v1
            if output_boundary is output_boundary_top:
                loss_boundaries_v1 += tf.reduce_sum(tf.square(output_boundary.v1-1))
            else:
                loss_boundaries_v1 += tf.reduce_sum(tf.square(output_boundary.v1))

            loss_boundaries_v2 += tf.reduce_sum(tf.square(output_boundary.v2))

        loss_boundaries_v1 /= self.num_boundary_samples
        loss_boundaries_v2 /= self.num_boundary_samples
        loss             += self.loss_weights.boundaries_v1 * loss_boundaries_v1
        loss             += self.loss_weights.boundaries_v2 * loss_boundaries_v2

        # Letting the tape go
        del tape

        self.loss_values.incompressibility.append(loss_incompr.numpy())
        self.loss_values.pressure_average.append(loss_pressure_mean.numpy())
        self.loss_values.interior_1.append(loss_interior_1.numpy())
        self.loss_values.interior_2.append(loss_interior_2.numpy())
        self.loss_values.boundaries_v1.append(loss_boundaries_v1.numpy())
        self.loss_values.boundaries_v2.append(loss_boundaries_v2.numpy())
        self.loss_values.total.append(loss.numpy())

        return loss



    def __grad(self):
        """Compute the gradient in one epoch of the u values w.r.t. the sample points."""
        with tf.GradientTape() as tape:
            loss = self.__loss()

        return tape.gradient(loss, self.model.trainable_variables)       

        

    def fit(self,
            n_epochs : int = 1000,
            plot_update_interval : int = 50,
            show_progress : bool = True):
        """Train the neural network."""

        # Main training loop
        for i in (tq.tqdm(range(n_epochs), desc = 'epochs') if show_progress else range(n_epochs)):

            self.epoch += 1

            grad = self.__grad()

            self.optimizer.apply_gradients(zip(grad,self.model.trainable_variables))

            if (i+1) % plot_update_interval == 0:   

                if self.show_training_plots:
                    self.update_training_plots()
                if self.show_network_plot:
                    self.update_weights_plot()
