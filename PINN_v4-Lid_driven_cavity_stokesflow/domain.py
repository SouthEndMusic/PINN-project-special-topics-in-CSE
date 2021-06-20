import numpy as np

class square_domain():
    """Class to produce points in the domain (0,1)x(0,1) and on its boundary, for training and plotting.

    Parameters
    ----------
    plotting_pixdens : int
        The number of pixels in both the x and y direction when plotting the result of the PINN.
    sampling_sizes : (int, int)
        The number of sample points (collocation points) in the x and the y direction
    sampling_func_x : function float -> float
        Function that determines the distribution of the sample (collocation points) in the interior of the domain
        in the x direction
    sampling_func_y : function float -> float
        Same as above but in the y direction
    samples_per_boundary: : (int,int,int,int)
        The number of sample points on each boundary, in the order left,right,top,bottom
    """

    def __init__(self,
                 plotting_pixdens : int = 50,
                 sampling_sizes = (50,50),
                 samples_per_boundary = 4*[100],
                 sampling_func_x = lambda x: np.sin(np.pi*x)**2,
                 sampling_func_y = lambda y: np.sin(np.pi*y/2)**0.8):

        self.plotting_pixdens     = plotting_pixdens
        self.sampling_sizes       = sampling_sizes
        self.samples_per_boundary = samples_per_boundary
        self.sampling_func_x      = sampling_func_x
        self.sampling_func_y      = sampling_func_y

        self._generate_plotting_grid()
        self._generate_interior_samples()
        self._generate_boundary_samples()


    def _generate_plotting_grid(self):
        """Consruct a regular grid of points that fits the domain, usable for plotting and
        as an input into the model."""
        
        x = np.linspace(0,1,self.plotting_pixdens).astype(np.float32)
        y = np.linspace(0,1,self.plotting_pixdens).astype(np.float32)

        # For plotting
        self.x_grid, self.y_grid = np.meshgrid(x,y)


        # For input in the model
        self.plotting_grid = np.stack([self.x_grid.ravel(),
                                       self.y_grid.ravel()]).T

        self.plotting_gridshape = self.x_grid.shape


    def _generate_interior_samples(self):
        """Consutrct the interior samples (collocation points) for training."""
        
        x = self.sampling_func_x(np.linspace(0,1,self.sampling_sizes[0])).astype(np.float32)
        y = self.sampling_func_y(np.linspace(0,1,self.sampling_sizes[1])).astype(np.float32)

        x_sample_grid, y_sample_grid = np.meshgrid(x,y)

        self.samples_interior = np.stack([x_sample_grid.ravel(),
                                          y_sample_grid.ravel()]).T



    def _generate_boundary_samples(self):
        """Construct the boundary samples for training"""

        left_samples      = np.zeros((self.samples_per_boundary[0],2))
        left_samples[:,1] = np.linspace(0,1,self.samples_per_boundary[0])
        right_samples     = np.ones((self.samples_per_boundary[1],2))
        right_samples[:,1]= np.linspace(0,1,self.samples_per_boundary[1])
        bottom_samples    = np.zeros((self.samples_per_boundary[2],2))
        bottom_samples[:,0] = np.linspace(0,1,self.samples_per_boundary[2])
        top_samples       = np.ones((self.samples_per_boundary[3],2))
        top_samples[:,0]  = np.linspace(0,1,self.samples_per_boundary[3])

        self.samples_boundary     = [left_samples,right_samples,top_samples,bottom_samples]
        self.samples_boundary     = [sb.astype(np.float32) for sb in self.samples_boundary]
        self.num_boundary_samples = sum(self.samples_per_boundary) 
        
