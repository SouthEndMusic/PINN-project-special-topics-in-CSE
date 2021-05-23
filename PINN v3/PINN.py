import numpy as np
import tensorflow as tf
import tqdm.notebook as tq
import matplotlib.pyplot as plt
import custom_cbar

class PINN_object_warmup():
    """Class to handle all neural network operations."""

    def __init__(self,
                 domain,
                 model,
                 optimizer,
                 kappa = 1e-2,
                 beta  = np.array([0.5,1])*np.sqrt(0.8),
                 RHS_f = lambda x,y: 1):

        # Domain (see domain.py)
        self.domain = domain

        # Neural network model in keras
        self.model = model

        # Optimizer
        self.optimizer = optimizer

        # PDE parameters
        self.kappa = kappa
        self.beta  = beta
        self.RHS_f = np.vectorize(RHS_f)

        # training data
        self.losses = []
        self.epoch  = 0

        # Generate the sample points
        self.domain.update_samples()



    def evaluate(self,inputs, training = False):
        """Evaluate the neural network in combination with the modifications for the boundary
        conditions."""

        u_NN = self.model(inputs, training = training)

        return self.domain.enforce_BC(u_NN, inputs[:,0], inputs[:,1])



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



    def __epoch_grad(self):
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

        u       = np.array(self.evaluate(self.domain.plotting_grid, training = True))
        u       = u.reshape(self.domain.plotting_gridshape)
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

            grads, loss = self.__epoch_grad()
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





class PINN_object_stokesflow():
    """Class to handle all neural network operations."""

    def __init__(self,
                 domain,
                 model,
                 optimizer,
                 RHS_f = lambda x,y: np.zeros((x.shape[0],2)),
                 loss_weights = [1,1,1],
                 BCs_v1 = 4*[lambda x: 0*x],
                 BCs_v2 = 4*[lambda x: 0*x]
                 ):

        # Domain (see domain.py)
        self.domain = domain

        # Neural network model in keras (see network.py)
        self.model = model

        # Optimizer
        self.optimizer = optimizer

        # PDE parameters
        self.RHS_f = RHS_f

        # training data
        self.losses_total      = []
        self.losses_incompr    = []
        self.losses_interior   = []
        self.losses_boundaries = []
        self.epoch             = 0

        # Generate the sample points
        self.domain.update_samples()

        # (In)homogeneous Dirichlet boundary conditions (left, right, top, bottom)
        self.BCs_v1 = BCs_v1
        self.BCs_v2 = BCs_v2

        # Loss weights (in order incompressibility, interior, boundary)
        self.loss_weights = loss_weights
        

    def evaluate(self,inputs, training = False):
        """Evaluate the neural network."""

        u_NN = self.model(inputs, training = training)

        if type(u_NN) == list:
            p,v1,v2 = u_NN

        else:
            p    = u_NN[:,0]
            v1   = u_NN[:,1]
            v2   = u_NN[:,2]

        return p,v1,v2



    def compute_f_and_BC_values(self):
        """Values of the function f in the RHS of the PDE"""
        
        sample_points = np.concatenate([self.domain.interior_samples,
                                        self.domain.boundary_samples])
        self.RHS_f_values  = self.RHS_f(sample_points[:,0],sample_points[:,1])

        BC_values_v1 = []
        BC_values_v2 = []

        for boundary_samples, BC_function, dim in zip(np.split(self.domain.boundary_samples,
                                                               self.domain.boundary_sample_starts, axis = 0),
                                                      self.BCs_v1,
                                                      [1,1,0,0]):

            BC_values_v1.append(BC_function(boundary_samples[:,dim])[:,np.newaxis])

        for boundary_samples, BC_function, dim in zip(np.split(self.domain.boundary_samples,
                                                               self.domain.boundary_sample_starts, axis = 0),
                                                       self.BCs_v2,
                                                       [1,1,0,0]):

            BC_values_v2.append(BC_function(boundary_samples[:,dim])[:,np.newaxis])

        self.BC_values_v1 = BC_values_v1
        self.BC_values_v2 = BC_values_v2

        



    def __epoch_loss(self):
        """Compute the loss in one epoch for the generated sample points."""

        # The sample points on the boundary also contribute to the interior losss
        sample_points = tf.convert_to_tensor(np.concatenate([self.domain.boundary_samples,
                                                             self.domain.interior_samples]))

        # Using GradientTape for computing the derivatives of the output of the model
        # with respect to the inputs
        with tf.GradientTape(persistent = True) as tape:

            tape.watch(sample_points)

            # Get the output of the model + BC adaptation for the interior sample points
            p_values, v1_values, v2_values = self.evaluate(sample_points, training = True)

            # Get the first order derivatives of the velocity components
            grad_v1_values = tape.gradient(v1_values, sample_points)
            grad_v2_values = tape.gradient(v2_values, sample_points)
            
            v1_values_x = grad_v1_values[:,0]
            v1_values_y = grad_v1_values[:,1]
            v2_values_x = grad_v2_values[:,0]
            v2_values_y = grad_v2_values[:,1]

        # Calculate the incompressibility loss
        v_divergence = v1_values_x + v2_values_y
        loss_incompr = tf.reduce_mean(tf.square(v_divergence))

        # Calculate the pressure gradient
        p_gradient = tape.gradient(p_values, sample_points)

        # Calculate nabla**2 v (Maybe this can be done more efficiently?)
        v1_values_xx = tape.gradient(v1_values_x, sample_points)[:,0]
        v1_values_yy = tape.gradient(v1_values_y, sample_points)[:,1]
        v2_values_xx = tape.gradient(v2_values_x, sample_points)[:,0]
        v2_values_yy = tape.gradient(v2_values_y, sample_points)[:,1]
        nablasq_v    = tf.stack([v1_values_xx + v1_values_yy,
                                 v2_values_xx + v2_values_yy], axis = 1)

        # Calculate the value of the LHS and RHS of the PDE
        diffusion = nablasq_v

        # Calculate loss from the PDE
        loss_vectors = -p_gradient + diffusion - self.RHS_f_values
        loss_PDE     = tf.reduce_mean(tf.square(loss_vectors[:,0])) + tf.reduce_mean(tf.square(loss_vectors[:,1]))

        loss_boundary = 0

        # These loops go over the boundary samples per boundary (left, right, top, bottom) and compute the difference to the
        # values of the functions that define the boundary conditions.
        for boundary_values, BC_values in zip(tf.split(v1_values[:self.domain.boundary_samples.shape[0]],
                                                       self.domain.samples_per_boundary, axis = 0),
                                              self.BC_values_v1):

            loss_boundary += tf.reduce_sum(tf.square(boundary_values - BC_values))

        for boundary_values, BC_values in zip(tf.split(v2_values[:self.domain.boundary_samples.shape[0]],
                                                       self.domain.samples_per_boundary, axis = 0),
                                              self.BC_values_v2):

            loss_boundary += tf.reduce_sum(tf.square(boundary_values - BC_values))

        loss_boundary /= 2*self.domain.boundary_samples.shape[0]

        # Weighted sum of losses
        loss = self.loss_weights[0]*loss_incompr + self.loss_weights[1]*loss_PDE + self.loss_weights[2]*loss_boundary

        # Record losses
        self.losses_total.append(loss.numpy())
        self.losses_incompr.append(loss_incompr.numpy())
        self.losses_interior.append(loss_PDE.numpy())
        self.losses_boundaries.append(loss_boundary.numpy())

        # Letting the tape go
        del tape

        return loss



    def __epoch_grad(self):
        """Compute the gradient in one epoch of the u values w.r.t. the sample points."""
        with tf.GradientTape() as tape:
            loss = self.__epoch_loss()

        return tape.gradient(loss, self.model.trainable_variables)



    def _init_plotting(self):
        """Initiate attributes for plotting."""

        fig, axs                    = plt.subplots(2,2, figsize = (10,10))
        ax_loss, ax_p, ax_v1, ax_v2 = axs.flat
        loss_lines                  = []

        for loss_type, zorder in zip(['total loss (weighted)','incompressibility loss','interior (PDE) loss','boundary loss'],
                                      [1,0,0,0]):
            loss_lines.append(ax_loss.plot([],[], label = loss_type, zorder = zorder)[0])
        
        ax_loss.set_yscale('log')
        ax_loss.set_xlabel('epochs')
        ax_loss.set_ylabel('loss')

        p, v1, v2 = self.evaluate(self.domain.plotting_grid, training = True)
        p         = np.array(p)       
        v1        = np.array(v1)        
        v2        = np.array(v2)
        p         = p.reshape(self.domain.plotting_gridshape)
        v1        = v1.reshape(self.domain.plotting_gridshape)
        v2        = v2.reshape(self.domain.plotting_gridshape)

        p_image = ax_p.imshow(p, extent = [0,self.domain.L,
                                           0,self.domain.H])
        v1_image = ax_v1.imshow(v1, extent = [0,self.domain.L,
                                              0,self.domain.H])
        v2_image = ax_v2.imshow(v2, extent = [0,self.domain.L,
                                              0,self.domain.H])

        self.p_colorbar  = custom_cbar.custom_colorbar(ax_p)
        self.v1_colorbar = custom_cbar.custom_colorbar(ax_v1)
        self.v2_colorbar = custom_cbar.custom_colorbar(ax_v2)

        for ax, label in zip([ax_p, ax_v1, ax_v2],['$p$','$v_1$','$v_2$']):
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_title(label)

        ax_loss.legend()

        self.ax_p       = ax_p
        self.ax_v1      = ax_v1
        self.ax_v2      = ax_v2
        self.ax_loss    = ax_loss
        self.fig        = fig
        self.loss_lines = loss_lines
        self.p_image    = p_image
        self.v1_image   = v1_image
        self.v2_image   = v2_image

        plt.tight_layout()

        

    def fit(self,
            n_epochs = 1000,
            plot_update_interval = 50,
            show_progress = True,
            ):
        """Train the neural network."""

        if not hasattr(self, 'RHS_f_values'):
            self.compute_f_and_BC_values()

        # Initiating plotting if desired
        if show_progress:
            if not hasattr(self, 'ax_u'):
                self._init_plotting()
            
            

        # Main train loop (show progress bar if desired)
        for i in (tq.tqdm(range(n_epochs), desc = 'epochs') if show_progress else range(n_epochs)):

            self.epoch += 1

            grads = self.__epoch_grad()
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))

            if show_progress and (i+1) % plot_update_interval == 0:
                self.update_plots()



    def update_plots(self):
        """Update plots."""

        for loss_line in self.loss_lines:
            loss_line.set_xdata(range(1,self.epoch+1))
            
        self.loss_lines[0].set_ydata(self.losses_total)
        self.loss_lines[1].set_ydata(self.losses_incompr)
        self.loss_lines[2].set_ydata(self.losses_interior)
        self.loss_lines[3].set_ydata(self.losses_boundaries)

        all_losses = self.losses_incompr + self.losses_interior + self.losses_boundaries + self.losses_total
        loss_min = min(all_losses)
        loss_max = max(all_losses)
        self.ax_loss.set_xlim(0,self.epoch)
        self.ax_loss.set_ylim(1e-1*loss_min, 1e1*loss_max)

        p, v1, v2 = self.evaluate(self.domain.plotting_grid)
        p         = np.array(p)
        v1        = np.array(v1)
        v2        = np.array(v2)
        p         = p.reshape(self.domain.plotting_gridshape)
        v1        = v1.reshape(self.domain.plotting_gridshape)
        v2        = v2.reshape(self.domain.plotting_gridshape)
        p         = np.rot90(p)
        v1        = np.rot90(v1)
        v2        = np.rot90(v2)

        self.p_image.set_data(p)
        self.p_image.set_clim(np.min(p),np.max(p))
        self.p_colorbar.update_data(p)

        self.v1_image.set_data(v1)
        self.v1_image.set_clim(np.min(v1),np.max(v1))
        self.v1_colorbar.update_data(v1)

        self.v2_image.set_data(v2)
        self.v2_image.set_clim(np.min(v2),np.max(v2))
        self.v2_colorbar.update_data(v2)

        self.fig.canvas.draw()
