import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class custom_colorbar():
    """Botched custom colorbar class, needed because the standard colorbars
    didn't want to update in the animation below."""
    
    def __init__(self, ax,
                 n_colors = 100,
                 n_ticks  = 5,
                 cmap     = 'viridis',
                 label    = ''):
        
        self.ax       = ax
        self.n_colors = n_colors
        self.n_ticks  = n_ticks
        divider       = make_axes_locatable(ax)
        self.cax      = divider.append_axes('right', size='5%', pad=0.05)
        
        self.cmap     = cmap
        
        self.cax.yaxis.set_label_position("right")
        self.cax.set_ylabel(label)
        
        self.cax.set_xticks([])
        self.cax.yaxis.tick_right()
        
    def update_data(self, data):
        
        data_min = np.min(data)
        data_max = np.max(data)
        
        self.cax.imshow(np.linspace(1,0,self.n_colors)[np.newaxis].T, extent = [data_min/20,data_max/20,data_min,data_max],
                        cmap = self.cmap)
        self.cax.set_yticks(np.linspace(data_min,data_max,self.n_ticks))
