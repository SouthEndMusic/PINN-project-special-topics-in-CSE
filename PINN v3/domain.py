import numpy as np

class rectangular_domain():

    def __init__(self,
                 dtype = np.float32,
                 L  = 1,
                 H  = 1,
                 alpha = 20,
                 plotting_pixdens = 50,
                 sampling_sizes   = (50,50),
                 sampling_dist_x  = lambda x: x,
                 sampling_dist_y  = lambda x: x,
                 samples_per_boundary = [100,100,100,100]):
        """The class to handle everything domain related for the PINN."""

        # The datatype that the PINN requires as input
        self.dtype = dtype

        # The width and height of the rectangle
        self.L = L
        self.H = H

        # Boundary enforcing parameter
        self.alpha = alpha

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

        # The number of samples on each boundary (left, right, top, bottom)
        self.samples_per_boundary = samples_per_boundary

        # The starting indices in an array of boundary samples of each new boundary
        self.boundary_sample_starts = [np.sum(samples_per_boundary[0:i]) for i in range(1,4)]



    def update_samples(self):
        """Update the training samples for loss calculation."""
              
        self.interior_samples = self.generate_interior_samples()
        self.boundary_samples = self.generate_boundary_samples()



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



    def generate_interior_samples(self):
        """Produce samples in the domain."""

        x = self.sampling_dist_x(np.linspace(0,1,self.sampling_sizes[0],dtype=self.dtype))*self.L
        y = self.sampling_dist_y(np.linspace(0,1,self.sampling_sizes[1],dtype=self.dtype))*self.H

        x_sample_grid, y_sample_grid = np.meshgrid(x,y)

        return np.stack([x_sample_grid.ravel(),y_sample_grid.ravel()]).T

    def generate_boundary_samples(self):
        """Produce samples on the boundary"""

        left_samples      = np.zeros((self.samples_per_boundary[0],2))
        left_samples[:,1] = np.linspace(0,1,self.samples_per_boundary[0])
        right_samples     = np.ones((self.samples_per_boundary[1],2))
        right_samples[:,1]= np.linspace(0,1,self.samples_per_boundary[1])
        bottom_samples    = np.zeros((self.samples_per_boundary[2],2))
        bottom_samples[:,0] = np.linspace(0,1,self.samples_per_boundary[2])
        top_samples       = np.ones((self.samples_per_boundary[3],2))
        top_samples[:,0]  = np.linspace(0,1,self.samples_per_boundary[3])
        
        return np.concatenate([left_samples,right_samples,top_samples,bottom_samples]).astype(self.dtype)


    def enforce_BC(self,u,x,y):
        """Enforce the Dirichlet boundary conditions on the input"""

        hom_dirichlet_1D_x   = (1-np.exp(-self.alpha*x/self.L))*(1-np.exp(-self.alpha*(self.L-x)/self.L))
        hom_dirichlet_1D_y   = (1-np.exp(-self.alpha*y/self.H))*(1-np.exp(-self.alpha*(self.H-y)/self.H))
        phi                  = (hom_dirichlet_1D_x*hom_dirichlet_1D_y)[:,None]

        return u*phi
        
