import numpy as np

### Satisfying homogeneous Dirichlet boundary conditions on the unit square (going steeply to 0 away from the boundary)

def hom_dirichlet_1D(x,alpha,L):
    return (1-np.exp(-alpha*x/L))*(1-np.exp(-alpha*(L-x)/L))

def hom_dirichlet_2D(x,y,alpha,L,H):
    return hom_dirichlet_1D(x,alpha,L)*hom_dirichlet_1D(y,alpha,H)
