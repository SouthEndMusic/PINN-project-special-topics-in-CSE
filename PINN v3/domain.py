import numpy as np

class rectangular_domain():

    def __init__(self,
                 dtype = np.float32,
                 L  = 1,
                 H  = 1,
                 plotting_pixdens = 50,
                 sampling_sizes   = (50,50),
                 sampling_dist_x  = lambda x: x,
                 sampling_dist_y  = lambda x: x):
        """The class to handle everything domain related for the PINN."""

        # The datatype that the PINN requires as input
        self.dtype = dtype

        # The width of the rectangle
        self.L = L

        # The height of the rectangle
        self.H = H

        # The number of samples in each direction
        self.sampling_sizes    = sampling_sizes
        self.sampling_gridsize = np.prod(sampling_sizes)

        # The distribution of the samples in each direction
        self.sampling_dist_x = sampling_dist_x
        self.sampling_dist_y = sampling_dist_y

        # Pixel density in plotting grid
        self.plotting_pixdens = plotting_pixdens

        # Generate grid
        self.generate_plotting_grid()

    def update_samples(self):
              
        self.samples = self.generate_samples()


    def generate_plotting_grid(self):
        """Construct a grid of points that fits the domain."""

        # Shape and size of grid for plotting
        self.plotting_gridshape = (int(self.L*self.plotting_pixdens), int(self.H*self.plotting_pixdens))
        self.plotting_gridsize  = np.prod(self.plotting_gridshape)

        # Construct arrays with shape sampling_grid_size with points in the domain in a grid
        x = np.linspace(0,self.L,self.plotting_gridshape[0]).astype(self.dtype)
        y = np.linspace(0,self.H,self.plotting_gridshape[1]).astype(self.dtype)

        self.y_grid, self.x_grid = np.meshgrid(y,x)

        # The grid in a format passable to the PINN (Nx*Ny,2)
        self.plotting_grid = np.stack([self.x_grid.ravel(), self.y_grid.ravel()]).T

    def generate_samples(self):
        """Prouce random samples in the domain."""

        x = self.sampling_dist_x(np.linspace(0,1,self.sampling_sizes[0],dtype=self.dtype))*self.L
        y = self.sampling_dist_y(np.linspace(0,1,self.sampling_sizes[1],dtype=self.dtype))*self.H

        x_sample_grid, y_sample_grid = np.meshgrid(x,y)

        return np.stack([x_sample_grid.ravel(),y_sample_grid.ravel()]).T

        
