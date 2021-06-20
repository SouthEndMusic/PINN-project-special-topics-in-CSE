import network
import domain
import training
import show_training

from importlib import reload

reload(network)
reload(domain)
reload(training)
reload(show_training)

class PINN_model(network.construct_network,
                 network.vizualize_network,
                 domain.square_domain,
                 training.training,
                 show_training.show_training):

    def __init__(self,
                 show_network_plot : bool = False,
                 show_training_plots : bool = False,
                 network_construction_kwargs :  dict = dict(),
                 network_vizualization_kwargs : dict = dict(),
                 domain_kwargs : dict = dict(),
                 training_kwargs : dict = dict(),
                 show_training_kwargs : dict = dict()
                 ):
        """Combine all parent classes in one class for the lid driven cavity stokesfloww PINN.

        Parameters
        ----------
            Kwargs for the parent classes; see their definition
        """
        network.construct_network.__init__(self,**network_construction_kwargs)
        domain.square_domain.__init__(self,**domain_kwargs)
        training.training.__init__(self,**training_kwargs)

        self.show_network_plot   = show_network_plot
        self.show_training_plots = show_training_plots

        if show_network_plot:
            network.vizualize_network.__init__(self,**network_vizualization_kwargs)
        if show_training_plots:
            show_training.show_training.__init__(self,**show_training_kwargs)
