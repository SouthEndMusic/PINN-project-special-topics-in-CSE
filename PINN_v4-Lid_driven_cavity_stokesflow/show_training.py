import numpy as np
import matplotlib.pyplot as plt
import custom_cbar

class show_training():
    """"Class to show plots of intermediate results and the loss
    progress during training."""

    def __init__(self, figsize = (10,10)):

        fig_training, ax_trainings = plt.subplots(2,2, figsize = figsize)

        ax_training_loss, ax_training_p, ax_training_v1, ax_training_v2 = ax_trainings.flat
        loss_lines                  = []

        for i,loss_type in enumerate(['total loss (weighted)','incompressibility loss','pressure average loss',
                                      'interior (PDE) loss 1','interior (PDE) loss 2',
                                      'boundary loss $v_1$', 'boundary loss $v_2$']):
            
            if i == 0:
                zorder = 1
            else:
                zorder = 0
  
            loss_lines.append(ax_training_loss.plot([],[], label = loss_type, zorder = zorder)[0])
        
        ax_training_loss.set_yscale('log')
        ax_training_loss.set_xlabel('epochs')
        ax_training_loss.set_ylabel('loss')

        output = self.evaluate(self.plotting_grid, training = True)
        p   = np.array(output.p)       
        v1  = np.array(output.v1)        
        v2  = np.array(output.v2)
        p   = p.reshape(self.plotting_gridshape)
        v1  = v1.reshape(self.plotting_gridshape)
        v2  = v2.reshape(self.plotting_gridshape)

        extent = [0,1,0,1]

        p_image  = ax_training_p.imshow(p,   extent = extent)
        v1_image = ax_training_v1.imshow(v1, extent = extent)
        v2_image = ax_training_v2.imshow(v2, extent = extent)

        self.p_colorbar  = custom_cbar.custom_colorbar(ax_training_p)
        self.v1_colorbar = custom_cbar.custom_colorbar(ax_training_v1)
        self.v2_colorbar = custom_cbar.custom_colorbar(ax_training_v2)

        for ax_training, label in zip([ax_training_p, ax_training_v1, ax_training_v2],['$p$','$v_1$','$v_2$']):
            ax_training.set_xlabel("$x$")
            ax_training.set_ylabel("$y$")
            ax_training.set_title(label)

        ax_training_loss.legend()

        self.ax_training_p       = ax_training_p
        self.ax_training_v1      = ax_training_v1
        self.ax_training_v2      = ax_training_v2
        self.ax_training_loss    = ax_training_loss
        self.fig_training        = fig_training
        self.loss_lines = loss_lines
        self.p_image    = p_image
        self.v1_image   = v1_image
        self.v2_image   = v2_image

        plt.tight_layout()


    def update_training_plots(self):
        """Update plots."""

        for loss_line in self.loss_lines:
            loss_line.set_xdata(range(1,self.epoch+1))
            
        all_losses = []

        for loss_line, loss_data in zip(self.loss_lines,self.loss_values):

            loss_line.set_ydata(loss_data)
            
            all_losses += loss_data

        loss_min = min(all_losses)
        loss_max = max(all_losses)
        self.ax_training_loss.set_xlim(0,self.epoch)
        self.ax_training_loss.set_ylim(1e-1*loss_min, 1e1*loss_max)

        output = self.evaluate(self.plotting_grid)
        p   = np.array(output.p)
        v1  = np.array(output.v1)
        v2  = np.array(output.v2)
        p   = p.reshape(self.plotting_gridshape)
        v1  = v1.reshape(self.plotting_gridshape)
        v2  = v2.reshape(self.plotting_gridshape)

        p  = np.flipud(p)
        v1 = np.flipud(v1)
        v2 = np.flipud(v2)

        self.p_image.set_data(p)
        self.p_image.set_clim(np.min(p),np.max(p))
        self.p_colorbar.update_data(p)

        self.v1_image.set_data(v1)
        self.v1_image.set_clim(np.min(v1),np.max(v1))
        self.v1_colorbar.update_data(v1)

        self.v2_image.set_data(v2)
        self.v2_image.set_clim(np.min(v2),np.max(v2))
        self.v2_colorbar.update_data(v2)

        self.fig_training.canvas.draw()

