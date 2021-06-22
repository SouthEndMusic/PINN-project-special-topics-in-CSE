import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class schematics:

    @staticmethod
    def BCs():

        x_range   = [-0.1,1.1]
        y_range   = [-0.1,1.1]
        N_vectors = 15
        fontsize  = 15

        fig, ax = plt.subplots(figsize = (5,5))

        boundary = Rectangle((0,0), 1,1, fill = False, linewidth = 1)
        ax.add_patch(boundary)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal','box')

        # Arrows at top
        ax.quiver(np.linspace(0,1,N_vectors),
                  np.ones(N_vectors),
                  np.ones(N_vectors),
                  np.zeros(N_vectors))

        plt.axis('off')
        plt.tight_layout()

        # BC indications
        ax.text(0.30, 1.05, '$(v_1,v_2) = (1,0)$', fontsize = fontsize)
        ax.text(0.30, -0.1, '$(v_1,v_2) = (0,0)$', fontsize = fontsize)

        ax.text(-0.10, 0.65, '$(v_1,v_2) = (0,0)$', rotation = 'vertical',
                fontsize = fontsize)
        ax.text( 1.05, 0.65, '$(v_1,v_2) = (0,0)$', rotation = 'vertical',
                fontsize = fontsize)

        # Corner points of square
        ax.text(-0.06, -0.075, '$(0,0)$', fontsize = fontsize/1.2)
        ax.text( 0.925,-0.075, '$(1,0)$', fontsize = fontsize/1.2)
        ax.text(-0.060, 1.040, '$(0,1)$', fontsize = fontsize/1.2)
        ax.text( 0.925,01.040, '$(1,1)$', fontsize = fontsize/1.2)

        plt.savefig('figures/BCs.png')
        plt.show()

    @staticmethod
    def swish_activation():

        x_range = [-5,3]
        N       = 100
        
        X = np.linspace(*x_range,N)

        def swish(x):
            return x/(1+np.exp(-x))

        Y       = swish(X)
        y_range = [np.min(Y) - 0.1, np.max(Y)]

        fig, ax = plt.subplots(figsize = (6,3))

        ax.plot(X,Y)
        ax.hlines(0,*x_range)
        ax.vlines(0,*y_range)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal','box')
        ax.set_title('Swish activation')

        plt.savefig('figures/Swish_activation.png')
        plt.show()

if __name__ == "__main__":

    schematics.swish_activation()

