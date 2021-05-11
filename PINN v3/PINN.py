import numpy as np
import tensorflow as tf
import boundary
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import custom_cbar

class PINN_object():
    """Class to handle all neural network operations."""

    def __init__(self,
                 domain,
                 model,
                 optimizer,
                 kappa = 1e-2,
                 beta  = np.array([0.5,1])*np.sqrt(0.8),
                 RHS_f = lambda x,y: 1,
                 alpha = 100,
                 ):

        # Domain (see domain.py)
        self.domain = domain

        # Neural network model in keras
        self.model = model

        # Optimizer
        self.optimizer = optimizer

        # PDE parameters
        self.kappa = kappa
        self.beta  = beta
        self.RHS_f = RHS_f

        # Training parameters
        self.alpha = alpha

        # training data
        self.losses = []
        self.epoch  = 0

        # Generate the sample points
        self.domain.update_samples()



    def evaluate(self,inputs, training = False):
        """Evaluate the neural network in combination with the modifications for the boundary
        conditions."""

        u_NN = self.model(inputs, training = training)
        phi  = boundary.hom_dirichlet_2D(inputs[:,0],inputs[:,1],
                                         self.alpha,
                                         self.domain.L,self.domain.H)[:,None]

        return u_NN*phi



    def __epoch_loss(self):
        """Compute the loss in one epoch for the generated sample points."""

        sample_points = tf.convert_to_tensor(self.domain.samples)

        # Using GradientTape for computing the derivatives of the output of the model
        # with respect to the inputs
        with tf.GradientTape(persistent = True) as tape:

            tape.watch(sample_points)

            # Get the output of the model + BC adaptation for the generated sample points
            u_values = self.evaluate(sample_points, training = True)

            # Calculate the gradient of the u values
            grad_u_values = tape.gradient(u_values, sample_points)
            
            u_values_x = grad_u_values[:,0]
            u_values_y = grad_u_values[:,1]

        # Calculate the second order derivatives
        u_values_xx = tape.gradient(u_values_x, sample_points)[:,0]
        u_values_yy = tape.gradient(u_values_y, sample_points)[:,1]

        # Calculate the value of the LHS and RHS of the PDE
        diffusion = -self.kappa*(u_values_xx + u_values_yy)
        advection = self.beta[0] * u_values_x + self.beta[1] * u_values_y
        f_values  = self.RHS_f(sample_points[:,0],sample_points[:,1])

        # Calculate loss from the PDE
        loss = tf.reduce_mean(tf.square(diffusion + advection - f_values))

        # Letting the tape go
        del tape

        return loss



    def __epoch_grads(self):
        """Compute the gradient in one epoch of the u values w.r.t. the sample points."""
        with tf.GradientTape() as tape:
            loss = self.__epoch_loss()

        return tape.gradient(loss, self.model.trainable_variables), loss



    def _init_plotting(self):
        """Initiate attributes for plotting."""

        fig, axs      = plt.subplots(1,2, figsize = (10,5))
        ax_loss, ax_u = axs
        loss_line     = ax_loss.plot([],[],label = "loss")[0]

        ax_loss.set_yscale('log')
        ax_loss.set_xlabel('epochs')
        ax_loss.set_ylabel('loss')

        u       = np.array(self.evaluate(self.domain.plotting_grid)).reshape(self.domain.plotting_gridshape)
        u_image = ax_u.imshow(u, extent = [0,self.domain.L,
                                           0,self.domain.H])

        self.u_colorbar = custom_cbar.custom_colorbar(ax_u)

        ax_u.set_xlabel("$x$")
        ax_u.set_ylabel("$y$")
        ax_u.set_title("$u$")

        self.ax_u      = ax_u
        self.ax_loss   = ax_loss
        self.fig       = fig
        self.loss_line = loss_line
        self.u_image   = u_image

        

    def fit(self,
            n_epochs = 1000,
            plot_update_interval = 50,
            show_progress = True,
            ):
        """Train the neural network."""

        # Initiating plotting if desired
        if show_progress:
            if not hasattr(self, 'ax_u'):
                self._init_plotting()
            
            

        # Main train loop (show progress bar if desired)
        for i in (tq.tqdm(range(n_epochs), desc = 'epochs') if show_progress else range(n_epochs)):

            self.epoch += 1

            grads, loss = self.__epoch_grads()
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))

            self.losses.append(loss.numpy())

            if show_progress and (i+1) % plot_update_interval == 0:
                self.update_plots()



    def update_plots(self):
        """Update plots."""

        self.loss_line.set_xdata(range(1,self.epoch+1))
        self.loss_line.set_ydata(self.losses)

        self.ax_loss.set_xlim(0,self.epoch)
        self.ax_loss.set_ylim(1e-1*min(self.losses), 1e1*max(self.losses))

        u = np.array(self.evaluate(self.domain.plotting_grid)).reshape(self.domain.plotting_gridshape)
        u = np.rot90(u)

        self.u_image.set_data(u)
        self.u_image.set_clim(np.min(u),np.max(u))
        self.u_colorbar.update_data(u)

        self.fig.canvas.draw()
