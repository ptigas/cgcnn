"""
CGCNN predictor for ASE Atoms objects
"""
import json
import os
import torch
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from torch.autograd import Variable

from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer, collate_pool
from cgcnn.model import CrystalGraphConvNet


class Normalizer(object):
    """Normalize a Tensor and restore it later."""
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class CGCNNPredictor:
    """
    A predictor class that uses a trained CGCNN model to predict properties 
    from ASE Atoms objects.
    """
    
    def __init__(self, model_path, atom_init_file=None, max_num_nbr=12, 
                 radius=8, dmin=0, step=0.2):
        """
        Initialize the CGCNN predictor.
        
        Parameters
        ----------
        model_path : str
            Path to the trained CGCNN model checkpoint
        atom_init_file : str, optional
            Path to atom_init.json file. If None, will look for it in the same
            directory as the model
        max_num_nbr : int
            Maximum number of neighbors for graph construction
        radius : float
            Cutoff radius for neighbor search (Angstroms)
        dmin : float
            Minimum distance for Gaussian distance filter
        step : float
            Step size for Gaussian distance filter
        """
        self.model_path = model_path
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        
        # Load model checkpoint
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model_args = checkpoint.get('args', {})
        
        # Set up atom initializer
        if atom_init_file is None:
            model_dir = os.path.dirname(model_path)
            atom_init_file = os.path.join(model_dir, 'atom_init.json')
            
        if not os.path.exists(atom_init_file):
            raise FileNotFoundError(f"Atom initialization file not found: {atom_init_file}")
            
        self.atom_initializer = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        
        # Initialize model
        # Get feature dimensions from atom_init file
        with open(atom_init_file, 'r') as f:
            atom_init_data = json.load(f)
        orig_atom_fea_len = len(list(atom_init_data.values())[0])
        nbr_fea_len = self.gdf.filter.shape[0]
        
        # Detect model architecture from checkpoint state_dict
        state_dict = checkpoint['state_dict']
        model_params = self._detect_model_params(state_dict, orig_atom_fea_len, nbr_fea_len)
        
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            **model_params
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Load normalizer if available
        self.normalizer = None
        if 'normalizer' in checkpoint:
            self.normalizer = Normalizer(torch.zeros(1))
            self.normalizer.load_state_dict(checkpoint['normalizer'])
            
        self.model.eval()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()
    
    def _detect_model_params(self, state_dict, orig_atom_fea_len, nbr_fea_len):
        """
        Detect model architecture parameters from the state_dict.
        
        Parameters
        ----------
        state_dict : dict
            Model state dictionary
        orig_atom_fea_len : int
            Original atom feature length
        nbr_fea_len : int
            Neighbor feature length
            
        Returns
        -------
        dict
            Model parameters for CrystalGraphConvNet
        """
        # Detect atom_fea_len from embedding layer
        atom_fea_len = state_dict['embedding.weight'].shape[0]
        
        # Detect n_conv from the number of convolutional layers
        n_conv = 0
        for key in state_dict.keys():
            if key.startswith('convs.') and key.endswith('.fc_full.weight'):
                layer_idx = int(key.split('.')[1])
                n_conv = max(n_conv, layer_idx + 1)
        
        # Detect h_fea_len from conv_to_fc layer
        h_fea_len = state_dict['conv_to_fc.weight'].shape[0]
        
        # Detect n_h from the number of fully connected layers after conv_to_fc
        n_h = 1  # Always at least 1 (the conv_to_fc layer)
        for key in state_dict.keys():
            if key.startswith('fcs.') and key.endswith('.weight'):
                layer_idx = int(key.split('.')[1])
                n_h = max(n_h, layer_idx + 2)  # +2 because we count conv_to_fc as first layer
        
        # Detect classification vs regression from output layer
        output_shape = state_dict['fc_out.weight'].shape[0]
        classification = output_shape > 1
        
        # Try to get task from model_args if available, otherwise infer from output shape
        if hasattr(self, 'model_args') and hasattr(self.model_args, 'task'):
            task = self.model_args.task
        else:
            task = 'classification' if classification else 'regression'
        
        return {
            'atom_fea_len': atom_fea_len,
            'n_conv': n_conv,
            'h_fea_len': h_fea_len,
            'n_h': n_h,
            'classification': classification
        }
    
    def _atoms_to_graph(self, atoms):
        """
        Convert ASE Atoms object to graph representation.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Input crystal structure
            
        Returns
        -------
        tuple
            (atom_fea, nbr_fea, nbr_fea_idx) graph representation
        """
        # Convert ASE atoms to pymatgen structure
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(atoms)
        
        # Get atom features
        atom_fea = np.vstack([
            self.atom_initializer.get_atom_fea(structure[i].specie.number)
            for i in range(len(structure))
        ])
        atom_fea = torch.Tensor(atom_fea)
        
        # Get neighbors
        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                # Pad with zeros if not enough neighbors
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + 
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) + 
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        
        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        
        # Apply Gaussian distance filter
        nbr_fea = self.gdf.expand(nbr_fea)
        
        return (torch.Tensor(atom_fea),
                torch.Tensor(nbr_fea),
                torch.LongTensor(nbr_fea_idx))
    
    def predict(self, atoms):
        """
        Predict property for a single ASE Atoms object.
        
        Parameters
        ----------
        atoms : ase.Atoms
            Input crystal structure
            
        Returns
        -------
        float or np.ndarray
            Predicted property value(s)
        """
        # Convert to graph representation
        graph_data = self._atoms_to_graph(atoms)
        
        # Prepare input for model
        atom_fea, nbr_fea, nbr_fea_idx = graph_data
        crystal_atom_idx = [torch.LongTensor(np.arange(len(atoms)))]
        
        # Move to GPU if available
        if self.cuda:
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()
            crystal_atom_idx = [idx.cuda() for idx in crystal_atom_idx]
        
        # Make prediction
        with torch.no_grad():
            input_var = (Variable(atom_fea),
                        Variable(nbr_fea),
                        nbr_fea_idx,
                        crystal_atom_idx)
            output = self.model(*input_var)
            
            # Apply normalizer if available and task is regression
            if (self.normalizer is not None and 
                getattr(self.model_args, 'task', 'regression') == 'regression'):
                prediction = self.normalizer.denorm(output.data.cpu()).numpy()
            else:
                prediction = output.data.cpu().numpy()
                
        return prediction.flatten()[0] if prediction.size == 1 else prediction.flatten()
    
    def predict_batch(self, atoms_list):
        """
        Predict properties for a list of ASE Atoms objects.
        
        Parameters
        ----------
        atoms_list : list of ase.Atoms
            List of input crystal structures
            
        Returns
        -------
        np.ndarray
            Array of predicted property values
        """
        predictions = []
        for atoms in atoms_list:
            pred = self.predict(atoms)
            predictions.append(pred)
        return np.array(predictions)


def predict_from_atoms(atoms, model_path, atom_init_file=None, **kwargs):
    """
    Convenience function to predict property from ASE Atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Input crystal structure
    model_path : str
        Path to trained CGCNN model
    atom_init_file : str, optional
        Path to atom_init.json file
    **kwargs
        Additional parameters for CGCNNPredictor
        
    Returns
    -------
    float or np.ndarray
        Predicted property value(s)
    """
    predictor = CGCNNPredictor(model_path, atom_init_file, **kwargs)
    return predictor.predict(atoms)